from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(),
        message_id="m1",
        internal=True,
    )


def _session_entry() -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._pending_messages = {}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = _session_entry()
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._queued_events = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._update_prompt_pending = {}
    runner._busy_input_mode = "interrupt"
    runner._draining = False
    runner._session_run_generation = {}
    runner._session_sources = {}
    runner._pending_native_image_paths_by_session = {}
    runner._background_tasks = {}
    runner._background_task_counter = 0
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._service_tier = None
    runner._fast_mode_by_session = {}
    runner._goal_state_by_session = {}
    runner._goal_runs_in_progress = set()
    runner._goal_queued_by_session = set()
    runner._is_telegram_topic_root_lobby = lambda _source: False
    runner._should_send_telegram_lobby_reminder = lambda _source: False
    runner._check_slash_access = lambda _source, _command: None
    runner._begin_session_run_generation = lambda _key: 1
    runner._release_running_agent_state = lambda key: runner._running_agents.pop(key, None)
    return runner, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("command_text", ["/queue do this next", "/q do this next"])
async def test_idle_queue_sends_payload_as_next_turn(command_text):
    runner, _adapter = _make_runner()
    captured = {}

    async def fake_handle_message_with_agent(event, source, key, generation):
        captured["text"] = event.text
        captured["command"] = event.get_command()
        captured["source"] = source
        captured["key"] = key
        captured["generation"] = generation
        return {"final_response": "", "messages": []}

    runner._handle_message_with_agent = fake_handle_message_with_agent

    result = await runner._handle_message(_make_event(command_text))

    assert result == {"final_response": "", "messages": []}
    assert captured["text"] == "do this next"
    assert captured["command"] is None
    assert captured["source"] == _make_source()
    assert captured["key"] == build_session_key(_make_source())
    assert captured["generation"] == 1
    assert runner._running_agents == {}


@pytest.mark.asyncio
async def test_multiplex_claimed_turn_completion_stays_in_routed_scope(
    tmp_path, monkeypatch
):
    """Post-turn hooks and cleanup must not fall back to the adapter owner."""
    from agent import secret_scope
    from gateway import run as run_mod
    from hermes_constants import get_hermes_home

    root = tmp_path / "hermes"
    profile = root / "profiles" / "fitness"
    root.mkdir(parents=True)
    profile.mkdir(parents=True)
    (root / ".env").write_text("ROUTE_SCOPE_TOKEN=owner\n", encoding="utf-8")
    (profile / ".env").write_text("ROUTE_SCOPE_TOKEN=routed\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(run_mod, "get_hermes_home", lambda: root)

    runner, _adapter = _make_runner()
    runner.config.multiplex_profiles = True
    runner._resolve_profile_home_for_source = lambda _source: profile
    event = _make_event("hello")
    event.source.profile = "fitness"
    observed = []

    def _assert_routed(stage):
        assert Path(get_hermes_home()) == profile
        assert secret_scope.get_secret("ROUTE_SCOPE_TOKEN") == "routed"
        observed.append(stage)

    async def _agent(_event, _source, _key, _generation):
        _assert_routed("agent")
        return {"final_response": "", "messages": []}

    async def _post_turn(**_kwargs):
        _assert_routed("post-turn")

    async def _clear(_event):
        _assert_routed("cleanup")
        return True

    runner._handle_message_with_agent = _agent
    runner._run_post_turn_hooks = _post_turn
    runner._clear_durable_active_turn = _clear

    secret_scope.set_multiplex_active(True)
    try:
        result = await runner._handle_message(event)
    finally:
        secret_scope.set_multiplex_active(False)

    assert result == {"final_response": "", "messages": []}
    assert observed == ["agent", "post-turn", "cleanup"]


@pytest.mark.asyncio
async def test_multiplex_scope_entry_failure_releases_claimed_turn(
    tmp_path, monkeypatch
):
    from gateway import run as run_mod

    runner, _adapter = _make_runner()
    runner.config.multiplex_profiles = True
    runner._resolve_profile_home_for_source = (
        lambda _source: tmp_path / "profiles" / "fitness"
    )
    event = _make_event("hello")
    event.source.profile = "fitness"

    @contextmanager
    def _failing_scope(_profile_home):
        raise RuntimeError("scope setup failed")
        yield  # pragma: no cover

    monkeypatch.setattr(run_mod, "_profile_runtime_scope", _failing_scope)

    with pytest.raises(RuntimeError, match="scope setup failed"):
        await runner._handle_message(event)

    assert runner._running_agents == {}


@pytest.mark.asyncio
async def test_multiplex_profile_resolution_failure_releases_claimed_turn():
    runner, _adapter = _make_runner()
    runner.config.multiplex_profiles = True

    def _fail_resolution(_source):
        raise RuntimeError("profile resolution failed")

    runner._resolve_profile_home_for_source = _fail_resolution
    event = _make_event("hello")
    event.source.profile = "fitness"

    with pytest.raises(RuntimeError, match="profile resolution failed"):
        await runner._handle_message(event)

    assert runner._running_agents == {}


def test_profile_runtime_scope_restores_home_when_secret_hydration_fails(
    tmp_path, monkeypatch
):
    from gateway import run as run_mod
    from hermes_constants import get_hermes_home

    root = tmp_path / "hermes"
    profile = root / "profiles" / "fitness"
    root.mkdir(parents=True)
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))

    def _fail_hydration(_profile_home):
        raise RuntimeError("hydration failed")

    monkeypatch.setattr(
        "hermes_cli.env_loader.hydrate_profile_secret_sources",
        _fail_hydration,
    )

    with pytest.raises(RuntimeError, match="hydration failed"):
        with run_mod._profile_runtime_scope(profile):
            pass

    assert Path(get_hermes_home()) == root


def test_profile_runtime_scope_restores_home_when_secret_cleanup_fails(
    tmp_path, monkeypatch
):
    from agent import secret_scope
    from gateway import run as run_mod
    from hermes_constants import get_hermes_home

    root = tmp_path / "hermes"
    profile = root / "profiles" / "fitness"
    root.mkdir(parents=True)
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    original_reset = secret_scope.reset_secret_scope

    def _reset_then_fail(token):
        original_reset(token)
        raise RuntimeError("secret cleanup failed")

    monkeypatch.setattr(secret_scope, "reset_secret_scope", _reset_then_fail)

    with pytest.raises(RuntimeError, match="secret cleanup failed"):
        with run_mod._profile_runtime_scope(profile):
            assert Path(get_hermes_home()) == profile

    assert Path(get_hermes_home()) == root


