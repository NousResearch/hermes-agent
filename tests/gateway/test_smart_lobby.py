"""Smart Discord lobby routing into profile-owned channel threads."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.smart_lobby import GatewaySmartLobbyMixin


def _raw_config():
    return {
        "enabled": True,
        "platform": "discord",
        "chat_id": "1506873825197949029",
        "default_profile": "default",
        "min_confidence": 0.65,
        "timeout_seconds": 20,
        "candidates": {
            "work": {
                "channel_id": "1539743242227290123",
                "description": "Cisco SecDev and Meraki work",
            },
            "lab": {
                "channel_id": "1539743243044913193",
                "description": "Homelab and infrastructure",
            },
        },
    }


def _event(text: str = "Please investigate this Meraki pipeline") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="1506873825197949029",
            chat_name="Techunix / #hermes",
            chat_type="group",
            user_id="42",
            user_name="Fred",
            guild_id="1235077258151067738",
            message_id="99",
        ),
    )


def test_smart_lobby_store_reservation_is_unique(tmp_path):
    from gateway.smart_lobby import SmartLobbyStore

    first = SmartLobbyStore(tmp_path / "smart-lobby.db")
    second = SmartLobbyStore(tmp_path / "smart-lobby.db")

    created, row = first.reserve(
        source_key="discord:guild:lobby:message",
        profile="work",
        channel_id="1539743242227290123",
        title="Investigate CI",
    )
    created_again, same_row = second.reserve(
        source_key="discord:guild:lobby:message",
        profile="lab",
        channel_id="1539743243044913193",
        title="Different decision",
    )

    assert created is True
    assert created_again is False
    assert row.profile == "work"
    assert same_row.profile == "work"


def test_gateway_config_round_trips_smart_lobby():
    cfg = GatewayConfig.from_dict({"smart_lobby": _raw_config()})
    assert cfg.smart_lobby == _raw_config()
    assert cfg.to_dict()["smart_lobby"] == _raw_config()


def test_gateway_loader_reads_nested_smart_lobby(monkeypatch, tmp_path):
    import yaml

    from gateway.config import load_gateway_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"gateway": {"smart_lobby": _raw_config()}}),
        encoding="utf-8",
    )

    cfg = load_gateway_config()

    assert cfg.smart_lobby == _raw_config()


def test_parse_smart_lobby_config_rejects_invalid_candidates():
    from gateway.smart_lobby import parse_smart_lobby_config

    raw = _raw_config()
    raw["candidates"]["bad"] = {"channel_id": "not-a-snowflake", "description": "bad"}
    parsed = parse_smart_lobby_config(raw)

    assert parsed is not None
    assert set(parsed.candidates) == {"work", "lab"}


def test_classifier_decision_requires_known_profile_and_confidence():
    from gateway.smart_lobby import parse_classifier_decision

    candidates = {"work", "lab"}
    assert parse_classifier_decision('{"profile":"work","confidence":0.9,"title":"Fix CI"}', candidates, 0.65).profile == "work"
    assert parse_classifier_decision('{"profile":"family","confidence":0.99}', candidates, 0.65) is None
    assert parse_classifier_decision('{"profile":"work","confidence":0.2}', candidates, 0.65) is None
    assert parse_classifier_decision("not json", candidates, 0.65) is None


class FakeAdapter:
    def __init__(self, *, thread_id: str = "777", authorized: bool = True):
        self.thread_id = thread_id
        self.authorized = authorized
        self.create_handoff_thread = AsyncMock(return_value=thread_id)
        self.handle_message = AsyncMock()
        self.send = AsyncMock()
        self.sources = []

    def _is_sender_authorized(self, _user_id, _chat_type=None, _chat_id=None):
        return self.authorized

    def build_source(self, **kwargs):
        self.sources.append(kwargs)
        return SessionSource(platform=Platform.DISCORD, **kwargs)


class Harness(GatewaySmartLobbyMixin):
    def _adapter_for_source(self, _source):
        return self.source_adapter


@pytest.mark.asyncio
async def test_lobby_routes_to_profile_adapter_thread_and_consumes_original(tmp_path):
    from gateway.smart_lobby import SmartLobbyDecision, SmartLobbyStore

    source_adapter = FakeAdapter(thread_id="source-unused")
    target_adapter = FakeAdapter(thread_id="777")
    runner = Harness()
    runner.config = SimpleNamespace(smart_lobby=_raw_config())
    runner.adapters = {Platform.DISCORD: source_adapter}
    runner._profile_adapters = {"work": {Platform.DISCORD: target_adapter}}
    runner.source_adapter = source_adapter
    runner._smart_lobby_store = SmartLobbyStore(tmp_path / "smart-lobby.db")
    runner._classify_smart_lobby = AsyncMock(
        return_value=SmartLobbyDecision(profile="work", confidence=0.93, title="Investigate Meraki pipeline")
    )

    event = _event()
    event.source.role_authorized = True
    consumed = await runner._maybe_route_smart_lobby(event)

    assert consumed is True
    target_adapter.create_handoff_thread.assert_awaited_once_with(
        "1539743242227290123", "Investigate Meraki pipeline"
    )
    assert len(target_adapter.sources) == 1
    built = target_adapter.sources[0]
    assert built["chat_id"] == "777"
    assert built["thread_id"] == "777"
    assert built["parent_chat_id"] == "1539743242227290123"
    assert built["chat_type"] == "thread"
    assert built["user_id"] == "42"
    assert built["role_authorized"] is False
    target_adapter.handle_message.assert_awaited_once()
    routed_event = target_adapter.handle_message.await_args.args[0]
    assert routed_event.text == "Please investigate this Meraki pipeline"
    assert routed_event.source.chat_id == "777"
    target_adapter.send.assert_awaited_once()
    assert "Please investigate this Meraki pipeline" in target_adapter.send.await_args.args[1]
    source_adapter.send.assert_awaited_once()
    assert "<#777>" in source_adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_lobby_slash_commands_stay_in_lobby_without_classification():
    source_adapter = FakeAdapter()
    runner = Harness()
    runner.config = SimpleNamespace(smart_lobby=_raw_config())
    runner.adapters = {Platform.DISCORD: source_adapter}
    runner._profile_adapters = {}
    runner.source_adapter = source_adapter
    runner._classify_smart_lobby = AsyncMock()

    consumed = await runner._maybe_route_smart_lobby(_event("/help"))

    assert consumed is False
    runner._classify_smart_lobby.assert_not_awaited()


@pytest.mark.asyncio
async def test_lobby_falls_back_to_normal_processing_on_low_confidence():
    source_adapter = FakeAdapter()
    runner = Harness()
    runner.config = SimpleNamespace(smart_lobby=_raw_config())
    runner.adapters = {Platform.DISCORD: source_adapter}
    runner._profile_adapters = {}
    runner.source_adapter = source_adapter
    runner._classify_smart_lobby = AsyncMock(return_value=None)

    consumed = await runner._maybe_route_smart_lobby(_event("something ambiguous"))

    assert consumed is False
    source_adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_selected_profile_missing_adapter_fails_closed_without_lobby_execution():
    from gateway.smart_lobby import SmartLobbyDecision

    source_adapter = FakeAdapter()
    runner = Harness()
    runner.config = SimpleNamespace(smart_lobby=_raw_config())
    runner.adapters = {Platform.DISCORD: source_adapter}
    runner._profile_adapters = {}
    runner.source_adapter = source_adapter
    runner._classify_smart_lobby = AsyncMock(
        return_value=SmartLobbyDecision(profile="work", confidence=0.9, title="Investigate CI")
    )

    consumed = await runner._maybe_route_smart_lobby(_event())

    assert consumed is True
    source_adapter.send.assert_awaited_once()
    assert "not executed in the lobby" in source_adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_lobby_falls_back_when_target_thread_cannot_be_created(tmp_path):
    from gateway.smart_lobby import SmartLobbyDecision, SmartLobbyStore

    source_adapter = FakeAdapter()
    target_adapter = FakeAdapter(thread_id=None)
    runner = Harness()
    runner.config = SimpleNamespace(smart_lobby=_raw_config())
    runner.adapters = {Platform.DISCORD: source_adapter}
    runner._profile_adapters = {"lab": {Platform.DISCORD: target_adapter}}
    runner.source_adapter = source_adapter
    runner._smart_lobby_store = SmartLobbyStore(tmp_path / "smart-lobby.db")
    runner._classify_smart_lobby = AsyncMock(
        return_value=SmartLobbyDecision(profile="lab", confidence=0.9, title="Check k3s")
    )

    consumed = await runner._maybe_route_smart_lobby(_event("check k3s"))

    assert consumed is True
    target_adapter.handle_message.assert_not_awaited()
    source_adapter.send.assert_awaited_once()
    assert "could not create" in source_adapter.send.await_args.args[1].lower()


@pytest.mark.asyncio
async def test_lobby_fails_closed_when_target_profile_rejects_user(monkeypatch, tmp_path):
    from gateway.smart_lobby import SmartLobbyDecision, SmartLobbyStore

    scope = {"active": False}

    @contextmanager
    def fake_profile_scope(_profile_home):
        scope["active"] = True
        try:
            yield
        finally:
            scope["active"] = False

    monkeypatch.setattr("gateway.run._profile_runtime_scope", fake_profile_scope)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: tmp_path)

    source_adapter = FakeAdapter()
    target_adapter = FakeAdapter(authorized=False)

    def target_auth(_user_id, _chat_type=None, _chat_id=None):
        assert scope["active"] is True
        return False

    target_adapter._is_sender_authorized = target_auth
    runner = Harness()
    runner.config = SimpleNamespace(smart_lobby=_raw_config())
    runner.adapters = {Platform.DISCORD: source_adapter}
    runner._profile_adapters = {"work": {Platform.DISCORD: target_adapter}}
    runner.source_adapter = source_adapter
    runner._smart_lobby_store = SmartLobbyStore(tmp_path / "smart-lobby.db")
    runner._classify_smart_lobby = AsyncMock(
        return_value=SmartLobbyDecision(profile="work", confidence=0.9, title="Investigate CI")
    )

    assert await runner._maybe_route_smart_lobby(_event()) is True
    target_adapter.create_handoff_thread.assert_not_awaited()
    target_adapter.handle_message.assert_not_awaited()
    source_adapter.send.assert_awaited_once()
    assert "not authorized" in source_adapter.send.await_args.args[1].lower()


@pytest.mark.asyncio
async def test_lobby_marks_failed_when_target_processing_task_does_not_start(tmp_path):
    from gateway.smart_lobby import SmartLobbyDecision, SmartLobbyStore

    source_adapter = FakeAdapter()
    target_adapter = FakeAdapter(thread_id="777")
    target_adapter._session_tasks = {}
    target_adapter._text_batch_key = lambda _event: "target-session"
    runner = Harness()
    runner.config = SimpleNamespace(smart_lobby=_raw_config())
    runner.adapters = {Platform.DISCORD: source_adapter}
    runner._profile_adapters = {"work": {Platform.DISCORD: target_adapter}}
    runner.source_adapter = source_adapter
    store = SmartLobbyStore(tmp_path / "smart-lobby.db")
    runner._smart_lobby_store = store
    runner._classify_smart_lobby = AsyncMock(
        return_value=SmartLobbyDecision(profile="work", confidence=0.9, title="Investigate CI")
    )

    assert await runner._maybe_route_smart_lobby(_event()) is True

    target_adapter.create_handoff_thread.assert_awaited_once()
    target_adapter.handle_message.assert_awaited_once()
    assert "did not start" in source_adapter.send.await_args.args[1]
    _created, row = store.reserve(
        source_key="discord:1235077258151067738:1506873825197949029:99",
        profile="work",
        channel_id="1539743242227290123",
        title="ignored",
    )
    assert row.status == "failed"
    assert row.error_kind == "dispatch_start"


@pytest.mark.asyncio
async def test_duplicate_lobby_message_never_creates_or_dispatches_twice(tmp_path):
    from gateway.smart_lobby import SmartLobbyDecision, SmartLobbyStore

    source_adapter = FakeAdapter()
    target_adapter = FakeAdapter(thread_id="777")
    runner = Harness()
    runner.config = SimpleNamespace(smart_lobby=_raw_config())
    runner.adapters = {Platform.DISCORD: source_adapter}
    runner._profile_adapters = {"work": {Platform.DISCORD: target_adapter}}
    runner.source_adapter = source_adapter
    runner._smart_lobby_store = SmartLobbyStore(tmp_path / "smart-lobby.db")
    runner._classify_smart_lobby = AsyncMock(
        return_value=SmartLobbyDecision(profile="work", confidence=0.9, title="Investigate CI")
    )
    event = _event()

    assert await runner._maybe_route_smart_lobby(event) is True
    assert await runner._maybe_route_smart_lobby(event) is True

    target_adapter.create_handoff_thread.assert_awaited_once()
    target_adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_lobby_message_without_stable_message_id_is_not_routed():
    source_adapter = FakeAdapter()
    runner = Harness()
    runner.config = SimpleNamespace(smart_lobby=_raw_config())
    runner.adapters = {Platform.DISCORD: source_adapter}
    runner._profile_adapters = {}
    runner.source_adapter = source_adapter
    runner._classify_smart_lobby = AsyncMock()
    event = _event()
    event.source.message_id = None

    assert await runner._maybe_route_smart_lobby(event) is False
    runner._classify_smart_lobby.assert_not_awaited()
