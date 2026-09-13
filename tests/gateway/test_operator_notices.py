"""Behavior contracts for operator-notice routing (issue #110246)."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.operator_notices import deliver_operator_notice
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


class _RecordingAdapter:
    def __init__(self, admins=()):
        self.config = SimpleNamespace(extra={"allow_admin_from": list(admins)})
        self.chat_sends = []
        self.direct_sends = []

    async def send(self, chat_id, content, metadata=None):
        self.chat_sends.append((chat_id, content, metadata))
        return SendResult(success=True)

    async def send_direct_notice(self, user_id, content, metadata=None):
        self.direct_sends.append((user_id, content, metadata))
        return SendResult(success=True)


def _source():
    return SessionSource(
        platform=Platform.DISCORD, chat_id="public-channel", chat_type="channel",
        user_id="participant", profile="work",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [None, "invalid", "chat"])
async def test_missing_or_invalid_operator_notice_mode_preserves_chat_delivery(configured):
    adapter = _RecordingAdapter()
    notices = {} if configured is None else {"session_reset": configured}

    mode = await deliver_operator_notice(
        adapter=adapter, source=_source(), kind="session_reset", content="reset",
        user_config={"gateway": {"notices": notices}}, chat_metadata={"thread_id": "thread"},
    )

    assert mode == "chat"
    assert adapter.chat_sends == [("public-channel", "reset", {"thread_id": "thread"})]
    assert adapter.direct_sends == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured_mode", "admins", "expected_targets"),
    [
        ("admin_dm", (), []),
        ("admin_dm", ("admin-2", "admin-1"), ["admin-1", "admin-2"]),
        ("log", ("admin-1",), []),
    ],
)
async def test_nonchat_modes_never_fall_back_to_originating_chat(
    configured_mode, admins, expected_targets, caplog,
):
    adapter = _RecordingAdapter(admins)

    mode = await deliver_operator_notice(
        adapter=adapter, source=_source(), kind="provider_error", content="provider failed",
        user_config={"gateway": {"notices": {"provider_error": configured_mode}}},
        chat_metadata={"thread_id": "thread"},
    )

    assert mode == configured_mode
    assert adapter.chat_sends == []
    assert [target for target, _content, _metadata in adapter.direct_sends] == expected_targets
    assert all(
        metadata == {"_interim_send": True, "hermes_profile": "work"}
        for _, _, metadata in adapter.direct_sends
    )
    if configured_mode == "admin_dm" and not admins:
        assert "no allow_admin_from identities" in caplog.text


def test_notice_config_is_loaded_inside_the_routed_profile(monkeypatch):
    runner = object.__new__(GatewayRunner)
    active_profile = []

    @contextmanager
    def profile_scope(source):
        active_profile.append(source.profile)
        try:
            yield
        finally:
            active_profile.pop()

    runner._profile_scope_for_source = profile_scope
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"profile": active_profile[-1] if active_profile else "default"},
    )

    assert runner._hmwa_user_config_for_source(_source()) == {"profile": "work"}
    assert active_profile == []


@pytest.mark.asyncio
async def test_reset_and_provider_paths_route_nonchat_notices(monkeypatch):
    runner = object.__new__(GatewayRunner)
    adapter = _RecordingAdapter(("admin",))
    runner._adapter_for_source = lambda _source: adapter
    runner._thread_metadata_for_source = lambda _source: {"thread_id": "thread"}
    runner._reset_notice_session_info = lambda _source: None
    runner._is_intentional_silence = lambda _result, _response: False
    monkeypatch.setattr("gateway.run_turn.build_channel_continuity_note", lambda *_args: None)
    config = {"gateway": {"notices": {"session_reset": "admin_dm", "provider_error": "log"}}}
    entry = SimpleNamespace(auto_reset_reason="suspended", session_id="session")
    sidecar = []

    await runner._hmwa_deliver_auto_reset_notice(entry, _source(), sidecar, config)
    response, intentional_silence, messages = await runner._hmwa_shape_agent_response(
        {"final_response": "API call failed after 3 retries: HTTP 401 Unauthorized", "messages": []},
        _source(), [], entry, "", "key", 1, "session", "discord", 0.0, config,
    )

    assert adapter.chat_sends == []
    assert [target for target, _content, _metadata in adapter.direct_sends] == ["admin"]
    assert response == ""
    assert intentional_silence is False
    assert messages == []
    assert sidecar
    assert entry.auto_reset_reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_mode", ["chat", "log"])
async def test_fallback_status_routes_by_policy_without_changing_legacy_chat_key(configured_mode):
    adapter = _RecordingAdapter(("admin",))
    adapter.status_sends = []

    async def send_or_update_status(chat_id, status_key, content, metadata=None):
        adapter.status_sends.append((chat_id, status_key, content, metadata))
        return SendResult(success=True)

    adapter.send_or_update_status = send_or_update_status
    ctx = TurnContext(
        source=_source(), _run_still_current=lambda: True,
        user_config={"gateway": {"notices": {"fallback_switch": configured_mode}}},
    )
    ctx._status_adapter = adapter
    ctx._status_chat_id = _source().chat_id
    ctx._status_thread_metadata = {"thread_id": "thread"}
    turn_runner = TurnRunner(SimpleNamespace(), ctx)
    scheduled = []
    turn_runner._schedule = lambda coro, _message: scheduled.append(asyncio.create_task(coro)) or scheduled[-1]

    turn_runner._status_callback_sync("fallback_switch", "🔄 Switched to fallback model")
    await scheduled[0]

    if configured_mode == "chat":
        assert adapter.status_sends == [(
            "public-channel", "lifecycle", "🔄 Switched to fallback model", {"thread_id": "thread"},
        )]
    else:
        assert adapter.status_sends == []
        assert adapter.chat_sends == []
