"""Profile-specific busy-input behavior for multiplexed gateways."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SessionSource,
    build_session_key,
)
from gateway.profile_routing import ProfileRoute
from gateway.run import GatewayRunner


class _ProfileAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {}


def _runner(*, default_mode: str = "interrupt") -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._busy_input_mode = default_mode
    runner._busy_text_mode = "queue" if default_mode == "queue" else "interrupt"
    runner._profile_adapters = {}
    runner.adapters = {}
    runner._sessions = {}
    runner._draining = False
    runner._restart_requested = False
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda source: True
    runner._session_has_compression_in_flight = AsyncMock(return_value=False)
    return runner


def _event(*, profile: str | None) -> MessageEvent:
    return MessageEvent(
        text="follow up",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            chat_type="dm",
            user_id="user-1",
            profile=profile,
        ),
        message_id="message-1",
    )


def _adapter() -> _ProfileAdapter:
    adapter = _ProfileAdapter(
        PlatformConfig(enabled=True, token="test-token"),
        Platform.TELEGRAM,
    )
    return adapter


async def _load_profile_snapshot(
    runner: GatewayRunner,
    profile_home,
    mode: str | None,
    *,
    legacy_text_mode: str | None = None,
) -> _ProfileAdapter:
    display = "display:\n"
    if mode is not None:
        display += f"  busy_input_mode: {mode}\n"
    if legacy_text_mode is not None:
        display += f"  busy_text_mode: {legacy_text_mode}\n"
    profile_home.mkdir()
    (profile_home / "config.yaml").write_text(display, encoding="utf-8")

    assert await runner._start_one_profile_adapters("research", profile_home, {}) == 0
    adapter = _adapter()
    runner._profile_adapters["research"][Platform.TELEGRAM] = adapter
    runner._configure_profile_adapter(adapter, "research", Platform.TELEGRAM)
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("secondary_mode", "handled", "expected_action", "expected_text_mode"),
    [
        ("queue", False, "queue", "queue"),
        ("steer", True, "steer", "interrupt"),
        ("interrupt", True, "interrupt", "interrupt"),
    ],
)
async def test_secondary_profile_busy_mode_controls_live_busy_behavior(
    tmp_path,
    monkeypatch,
    secondary_mode,
    handled,
    expected_action,
    expected_text_mode,
):
    """A routed profile chooses queue/steer/interrupt independently."""
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    runner = _runner(default_mode="interrupt")
    adapter = await _load_profile_snapshot(
        runner,
        tmp_path / "research",
        secondary_mode,
    )
    event = _event(profile="research")
    session_key = runner._session_key_for_source(event.source)
    agent = MagicMock()
    agent._active_children = []
    agent.steer.return_value = True
    runner._running_agents[session_key] = agent

    result = await runner._handle_active_session_busy_message(event, session_key)

    assert result is handled
    assert adapter._busy_text_mode == expected_text_mode
    if expected_action == "queue":
        agent.steer.assert_not_called()
        agent.interrupt.assert_not_called()
    elif expected_action == "steer":
        agent.steer.assert_called_once_with("follow up")
        agent.interrupt.assert_not_called()
    else:
        agent.steer.assert_not_called()
        agent.interrupt.assert_called_once_with("follow up")


@pytest.mark.asyncio
@pytest.mark.parametrize("secondary_mode", ["queue", "steer"])
async def test_secondary_profile_busy_mode_controls_priority_path(
    tmp_path,
    monkeypatch,
    secondary_mode,
):
    """The runner's early active-agent path uses the same routed policy."""
    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "0")
    runner = _runner(default_mode="interrupt")
    adapter = await _load_profile_snapshot(
        runner,
        tmp_path / "research",
        secondary_mode,
    )
    event = _event(profile="research")
    session_key = runner._session_key_for_source(event.source)
    agent = MagicMock()
    agent._active_children = []
    agent.steer.return_value = True
    runner._running_agents[session_key] = agent

    assert await runner._handle_message(event) is None

    agent.interrupt.assert_not_called()
    if secondary_mode == "queue":
        agent.steer.assert_not_called()
        assert adapter._pending_messages[session_key] is event
    else:
        agent.steer.assert_called_once_with("follow up")
        assert session_key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_busy_status_dispatches_through_active_session_path(tmp_path):
    """A running session still dispatches /busy through its normal handler."""
    runner = _runner(default_mode="interrupt")
    await _load_profile_snapshot(runner, tmp_path / "research", "queue")
    event = _event(profile="research")
    event.text = "/busy status"
    session_key = runner._session_key_for_source(event.source)
    runner._running_agents[session_key] = MagicMock()

    response = await runner._handle_message(event)

    assert "queue" in str(response).lower()


@pytest.mark.asyncio
async def test_busy_change_updates_only_routed_profile(tmp_path, monkeypatch):
    """A profile-scoped /busy change refreshes its snapshot and adapter."""
    runner = _runner(default_mode="interrupt")
    adapter = await _load_profile_snapshot(
        runner,
        tmp_path / "research",
        "queue",
    )
    event = _event(profile="research")
    event.text = "/busy steer"
    monkeypatch.setattr("cli.save_config_value", lambda _key, _value: True)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {"display": {"busy_input_mode": "steer"}},
    )

    response = await runner._handle_busy_command(event)

    assert "steer" in str(response).lower()
    assert runner._busy_input_mode == "interrupt"
    assert runner._effective_busy_input_mode(event.source) == "steer"
    assert adapter._busy_text_mode == "interrupt"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("default_mode", "secondary_mode", "queued"),
    [
        ("interrupt", "queue", True),
        ("queue", "interrupt", False),
    ],
)
async def test_secondary_profile_busy_mode_controls_busy_handler_restart_drain(
    tmp_path,
    default_mode,
    secondary_mode,
    queued,
):
    runner = _runner(default_mode=default_mode)
    adapter = await _load_profile_snapshot(
        runner,
        tmp_path / "research",
        secondary_mode,
    )
    runner._draining = True
    runner._restart_requested = True
    event = _event(profile="research")
    session_key = runner._session_key_for_source(event.source)

    assert await runner._handle_active_session_busy_message(event, session_key) is True
    assert (session_key in adapter._pending_messages) is queued


@pytest.mark.asyncio
async def test_secondary_profile_busy_mode_controls_priority_restart_drain(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "0")
    runner = _runner(default_mode="interrupt")
    adapter = await _load_profile_snapshot(
        runner,
        tmp_path / "research",
        "queue",
    )
    runner._draining = True
    runner._restart_requested = True
    event = _event(profile="research")
    session_key = runner._session_key_for_source(event.source)
    agent = MagicMock()
    agent._active_children = []
    runner._running_agents[session_key] = agent

    response = await runner._handle_message(event)

    assert isinstance(response, str)
    assert "queued" in response
    assert adapter._pending_messages[session_key] is event
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_secondary_adapter_busy_guard_stamps_profile_before_resolving_mode(
    tmp_path,
    monkeypatch,
):
    """Per-profile adapters route busy events before the message wrapper runs."""
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    runner = _runner(default_mode="interrupt")
    adapter = await _load_profile_snapshot(
        runner,
        tmp_path / "research",
        "steer",
    )
    event = _event(profile=None)
    # Seed the lane the adapter itself derives. A profile-owned adapter keys its
    # own _active_sessions in its own namespace (agent:research:...) — see
    # BasePlatformAdapter._session_key_profile. Seeding the unstamped
    # agent:main: key here asserted the pre-fix behaviour, where every profile's
    # adapter collapsed onto the default lane.
    adapter_session_key = build_session_key(event.source, profile="research")
    adapter._active_sessions[adapter_session_key] = asyncio.Event()

    routed_source = _event(profile="research").source
    routed_session_key = runner._session_key_for_source(routed_source)
    agent = MagicMock()
    agent._active_children = []
    agent.steer.return_value = True
    runner._running_agents[routed_session_key] = agent

    await adapter.handle_message(event)

    assert event.source.profile == "research"
    agent.steer.assert_called_once_with("follow up")
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_secondary_legacy_busy_text_mode_is_profile_specific(tmp_path):
    runner = _runner(default_mode="interrupt")
    adapter = await _load_profile_snapshot(
        runner,
        tmp_path / "research",
        "interrupt",
        legacy_text_mode="queue",
    )
    event = _event(profile="research")
    session_key = runner._session_key_for_source(event.source)
    agent = MagicMock()
    agent._active_children = []
    runner._running_agents[session_key] = agent

    assert await runner._handle_active_session_busy_message(event, session_key) is False
    assert adapter._busy_text_mode == "queue"
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_default_busy_mode_is_unchanged_by_secondary_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    runner = _runner(default_mode="interrupt")
    await _load_profile_snapshot(runner, tmp_path / "research", "steer")
    adapter = _adapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    event = _event(profile=None)
    session_key = runner._session_key_for_source(event.source)
    agent = MagicMock()
    agent._active_children = []
    runner._running_agents[session_key] = agent

    assert await runner._handle_active_session_busy_message(event, session_key) is True
    agent.interrupt.assert_called_once_with("follow up")
    agent.steer.assert_not_called()
    assert runner._busy_input_mode == "interrupt"


@pytest.mark.asyncio
@pytest.mark.parametrize("secondary_mode", [None, "not-a-mode"])
async def test_missing_or_invalid_secondary_mode_falls_back_to_gateway_default(
    tmp_path,
    secondary_mode,
):
    runner = _runner(default_mode="queue")
    await _load_profile_snapshot(runner, tmp_path / "research", secondary_mode)
    source = _event(profile="research").source

    assert runner._effective_busy_input_mode(source) == "queue"
    assert runner._effective_busy_text_mode(source) == "queue"
    assert runner._busy_input_mode == "queue"
    assert runner._busy_text_mode == "queue"


def test_profile_route_and_nonmultiplexed_resolution_preserve_boundaries(
    tmp_path,
    monkeypatch,
):
    runner = _runner(default_mode="interrupt")
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda **_: [("research", tmp_path / "research")],
    )
    runner._snapshot_profile_busy_modes(
        "research",
        {"display": {"busy_input_mode": "steer"}},
    )
    runner.config.profile_routes = [
        ProfileRoute(
            name="research-chat",
            platform="telegram",
            profile="research",
            chat_id="chat-1",
        )
    ]
    source = _event(profile=None).source

    # `_profile_name_for_source` rejects a route whose target profile is not in
    # the served set (`profiles_to_serve`). Without this patch the test reads
    # the runner's real on-disk profiles, so "research" is unserved on any
    # machine that does not happen to have it — and the route is rejected
    # before the busy-mode snapshot is consulted. Sibling coverage in
    # tests/gateway/test_profile_resolution.py patches the same seam.
    with patch(
        "hermes_cli.profiles.profiles_to_serve",
        return_value=[
            ("default", Path("/profiles/default")),
            ("research", Path("/profiles/research")),
        ],
    ):
        assert runner._effective_busy_input_mode(source) == "steer"

    runner.config.multiplex_profiles = False
    source.profile = "research"
    assert runner._effective_busy_input_mode(source) == "interrupt"


@pytest.mark.asyncio
async def test_effective_mode_uses_startup_snapshot_without_rereading_config(
    tmp_path,
    monkeypatch,
):
    import gateway.run as gateway_run

    runner = _runner(default_mode="interrupt")
    await _load_profile_snapshot(runner, tmp_path / "research", "steer")
    source = _event(profile="research").source

    def fail_config_read():
        raise AssertionError("busy-mode lookup reread config after startup")

    monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", fail_config_read)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", fail_config_read)

    assert runner._effective_busy_input_mode(source) == "steer"
    assert runner._effective_busy_input_mode(source) == "steer"
    assert runner._effective_busy_text_mode(source) == "interrupt"
