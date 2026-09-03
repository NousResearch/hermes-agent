"""Tests for the ``pre_command`` observer hook (#64204).

The hook fires when a recognized slash command is about to be dispatched,
BEFORE the handler runs, on both the interactive CLI (cli.py
process_command) and the gateway canonical-command dispatch
(gateway/run.py _handle_message). Observer-only in v1: return values are
ignored (directives are logged at debug for discoverability).

The gateway running-agent intercept path (/stop, /approve, busy_policy
dispatch while a turn is live) is deliberately excluded — control-plane
commands on an in-flight run must not be observable/veto-able by plugins.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------


def test_pre_command_in_valid_hooks():
    from hermes_cli.plugins import VALID_HOOKS

    assert "pre_command" in VALID_HOOKS


# ---------------------------------------------------------------------------
# fire_pre_command_hook helper
# ---------------------------------------------------------------------------


def test_fire_helper_is_observer_only_and_never_raises(monkeypatch):
    """Directive-shaped returns are ignored; plugin exceptions don't escape."""
    from hermes_cli import plugins as plugins_mod

    calls = {}

    class _FakeManager:
        def has_hook(self, name):
            return name == "pre_command"

        def invoke_hook(self, name, **kwargs):
            calls["name"] = name
            calls["kwargs"] = kwargs
            # A directive-shaped return must be ignored (observer v1).
            return [{"action": "block", "message": "nope"}]

    monkeypatch.setattr(plugins_mod, "get_plugin_manager", _FakeManager)

    # Must not raise, must not return anything actionable.
    result = plugins_mod.fire_pre_command_hook(
        surface="cli",
        command="model",
        alias_used="model",
        args_raw="gpt-x",
    )
    assert result is None
    assert calls["name"] == "pre_command"
    assert calls["kwargs"]["surface"] == "cli"
    assert calls["kwargs"]["command"] == "model"


def test_fire_helper_skips_when_no_plugin_listens(monkeypatch):
    from hermes_cli import plugins as plugins_mod

    class _FakeManager:
        def has_hook(self, name):
            return False

        def invoke_hook(self, name, **kwargs):  # pragma: no cover
            raise AssertionError("invoke_hook must not be called")

    monkeypatch.setattr(plugins_mod, "get_plugin_manager", _FakeManager)
    plugins_mod.fire_pre_command_hook(
        surface="cli", command="help", alias_used="help", args_raw="",
    )


def test_fire_helper_swallows_manager_errors(monkeypatch):
    from hermes_cli import plugins as plugins_mod

    def _boom():
        raise RuntimeError("plugin discovery exploded")

    monkeypatch.setattr(plugins_mod, "get_plugin_manager", _boom)
    # Never raises.
    plugins_mod.fire_pre_command_hook(
        surface="gateway", command="new", alias_used="reset", args_raw="",
    )


# ---------------------------------------------------------------------------
# CLI surface (cli.py process_command)
# ---------------------------------------------------------------------------


def _make_cli():
    import cli as cli_mod

    inst = object.__new__(cli_mod.HermesCLI)
    inst.session_id = "sess-cli-1"
    inst._pending_resume_sessions = None
    return inst


def test_cli_fires_for_recognized_command(monkeypatch):
    from hermes_cli import plugins as plugins_mod

    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(plugins_mod, "fire_pre_command_hook", _capture)

    inst = _make_cli()
    inst.show_help = lambda *a, **k: None
    assert inst.process_command("/help") is True

    assert captured["surface"] == "cli"
    assert captured["command"] == "help"
    assert captured["alias_used"] == "help"
    assert captured["args_raw"] == ""
    assert captured["session_key"] == "sess-cli-1"


def test_cli_reports_canonical_name_for_alias(monkeypatch):
    """/exit is an alias of /quit — hook payload reports canonical 'quit'."""
    from hermes_cli import plugins as plugins_mod

    captured = {}
    monkeypatch.setattr(
        plugins_mod, "fire_pre_command_hook",
        lambda **kwargs: captured.update(kwargs),
    )

    inst = _make_cli()
    # /exit returns False (exit the REPL) — dispatch still happens after
    # the hook fires.
    assert inst.process_command("/exit") is False

    assert captured["command"] == "quit"
    assert captured["alias_used"] == "exit"
    assert captured["surface"] == "cli"


def test_cli_passes_raw_args(monkeypatch):
    from hermes_cli import plugins as plugins_mod

    captured = {}
    monkeypatch.setattr(
        plugins_mod, "fire_pre_command_hook",
        lambda **kwargs: captured.update(kwargs),
    )

    inst = _make_cli()
    inst.show_help = lambda *a, **k: None
    # /help ignores args but the payload must carry them raw (case kept).
    inst.process_command("/help Some RAW args")
    assert captured["args_raw"] == "Some RAW args"


def test_cli_hook_before_handler(monkeypatch):
    """The hook fires BEFORE the command handler runs."""
    from hermes_cli import plugins as plugins_mod

    order = []
    monkeypatch.setattr(
        plugins_mod, "fire_pre_command_hook",
        lambda **kwargs: order.append("hook"),
    )

    inst = _make_cli()
    inst.show_help = lambda *a, **k: order.append("handler")
    inst.process_command("/help")
    assert order == ["hook", "handler"]


# ---------------------------------------------------------------------------
# Gateway surface (gateway/run.py _handle_message)
# ---------------------------------------------------------------------------


def _make_source():
    from gateway.config import Platform
    from gateway.session import SessionSource

    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str):
    from gateway.platforms.base import MessageEvent, MessageType

    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(),
        message_id="m1",
        internal=True,
    )


def _session_entry():
    from gateway.config import Platform
    from gateway.session import SessionEntry, build_session_key

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
    from gateway.config import GatewayConfig, Platform, PlatformConfig
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
    runner._should_send_voice_reply = lambda *_a, **_k: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *a, **k: None
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
    runner._release_running_agent_state = (
        lambda key: runner._running_agents.pop(key, None)
    )
    return runner, adapter


@pytest.mark.asyncio
async def test_gateway_fires_for_recognized_command(monkeypatch):
    from hermes_cli import plugins as plugins_mod

    captured = {}
    monkeypatch.setattr(
        plugins_mod, "fire_pre_command_hook",
        lambda **kwargs: captured.update(kwargs),
    )

    runner, _adapter = _make_runner()

    async def _fake_agent(event, source, key, generation):
        return {"final_response": "", "messages": []}

    runner._handle_message_with_agent = _fake_agent

    await runner._handle_message(_make_event("/queue do it later"))

    assert captured["surface"] == "gateway"
    assert captured["command"] == "queue"
    assert captured["alias_used"] == "queue"
    assert captured["args_raw"] == "do it later"
    assert captured["platform"] == "telegram"
    assert captured["session_key"]


@pytest.mark.asyncio
async def test_gateway_reports_canonical_name_for_alias(monkeypatch):
    """/q is an alias of /queue — payload reports canonical name."""
    from hermes_cli import plugins as plugins_mod

    captured = {}
    monkeypatch.setattr(
        plugins_mod, "fire_pre_command_hook",
        lambda **kwargs: captured.update(kwargs),
    )

    runner, _adapter = _make_runner()

    async def _fake_agent(event, source, key, generation):
        return {"final_response": "", "messages": []}

    runner._handle_message_with_agent = _fake_agent

    await runner._handle_message(_make_event("/q later please"))

    assert captured["command"] == "queue"
    assert captured["alias_used"] == "q"


@pytest.mark.asyncio
async def test_gateway_does_not_fire_for_plain_text(monkeypatch):
    from hermes_cli import plugins as plugins_mod

    fired = []
    monkeypatch.setattr(
        plugins_mod, "fire_pre_command_hook",
        lambda **kwargs: fired.append(kwargs),
    )

    runner, _adapter = _make_runner()

    async def _fake_agent(event, source, key, generation):
        return {"final_response": "ok", "messages": []}

    runner._handle_message_with_agent = _fake_agent

    await runner._handle_message(_make_event("just a normal message"))
    assert fired == []


@pytest.mark.asyncio
async def test_gateway_control_plane_intercept_excluded(monkeypatch):
    """Commands hitting the running-agent intercept path must NOT fire the
    hook — /stop et al. during an active run are control-plane operations."""
    from hermes_cli import plugins as plugins_mod

    fired = []
    monkeypatch.setattr(
        plugins_mod, "fire_pre_command_hook",
        lambda **kwargs: fired.append(kwargs),
    )

    runner, _adapter = _make_runner()
    runner._peek_session_state = lambda _key: None
    runner._is_session_running = lambda _key: True
    runner._dispatch_busy_slash_command = AsyncMock(return_value="busy-handled")

    result = await runner._handle_message(_make_event("/stop"))

    assert result == "busy-handled"
    runner._dispatch_busy_slash_command.assert_awaited_once()
    assert fired == []


@pytest.mark.asyncio
async def test_gateway_hook_failure_is_non_fatal(monkeypatch):
    """A raising fire helper must not break command dispatch."""
    from hermes_cli import plugins as plugins_mod

    def _boom(**kwargs):
        raise RuntimeError("bad plugin infra")

    monkeypatch.setattr(plugins_mod, "fire_pre_command_hook", _boom)

    runner, _adapter = _make_runner()
    captured = {}

    async def _fake_agent(event, source, key, generation):
        captured["text"] = event.text
        return {"final_response": "", "messages": []}

    runner._handle_message_with_agent = _fake_agent

    result = await runner._handle_message(_make_event("/queue still works"))
    assert result == {"final_response": "", "messages": []}
    assert captured["text"] == "still works"
