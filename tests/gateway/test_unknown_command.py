"""Tests for gateway warning when an unrecognized /command is dispatched.

Without this warning, unknown slash commands get forwarded to the LLM as plain
text, which often leads to silent failure (e.g. the model inventing a bogus
delegate_task call instead of telling the user the command doesn't exist).
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class _FakeGatewayAdapter:
    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self):
        self.send = AsyncMock()
        self.send_typing = AsyncMock()
        self._pending_messages = {}
        self._active_sessions = {}
        self._post_delivery_callbacks = {}

    def has_pending_interrupt(self, _session_key):
        return False

    def get_pending_message(self, session_key):
        return self._pending_messages.pop(session_key, None)

    def extract_media(self, text):
        return [], text

    def extract_images(self, text):
        return [], text

    async def edit_message(self, *args, **kwargs):
        return None

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
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
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


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

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._draining = False
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
    return runner


@pytest.mark.asyncio
async def test_unknown_slash_command_returns_guidance(monkeypatch):
    """A genuinely unknown /foobar should return user-facing guidance, not
    silently drop through to the LLM."""
    import gateway.run as gateway_run

    runner = _make_runner()
    # If the LLM were called, this would fail: the guard must short-circuit
    # before _run_agent is invoked.
    runner._run_agent = AsyncMock(
        side_effect=AssertionError(
            "unknown slash command leaked through to the agent"
        )
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/definitely-not-a-command"))

    assert result is not None
    assert "Unknown command" in result
    assert "/definitely-not-a-command" in result
    assert "/commands" in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_slash_command_underscored_form_also_guarded(monkeypatch):
    """Telegram may send /foo_bar — same guard must trigger for underscored
    commands that normalize to unknown hyphenated names."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError(
            "unknown slash command leaked through to the agent"
        )
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/made_up_thing"))

    assert result is not None
    assert "Unknown command" in result
    assert "/made_up_thing" in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_known_slash_command_not_flagged_as_unknown(monkeypatch):
    """A real built-in like /status must NOT hit the unknown-command guard."""
    runner = _make_runner()
    # Make _handle_status_command exist via the normal path by running a real
    # dispatch. If the guard fires, the return string will mention "Unknown".
    runner._running_agents[build_session_key(_make_source())] = MagicMock()

    result = await runner._handle_message(_make_event("/status"))

    assert result is not None
    assert "Unknown command" not in result


@pytest.mark.asyncio
async def test_underscored_alias_for_hyphenated_builtin_not_flagged(monkeypatch):
    """Telegram autocomplete sends /reload_mcp for the /reload-mcp built-in.
    That must NOT be flagged as unknown."""
    import gateway.run as gateway_run

    runner = _make_runner()
    # Prevent real MCP work; we only care that the unknown guard doesn't fire.
    async def _noop_reload(*_a, **_kw):
        return "mcp reloaded"

    runner._handle_reload_mcp_command = _noop_reload  # type: ignore[attr-defined]

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/reload_mcp"))

    # Whatever /reload_mcp returns, it must not be the unknown-command guard.
    if result is not None:
        assert "Unknown command" not in result


@pytest.mark.asyncio
async def test_handoff_rewrites_to_agent_prompt_when_idle(monkeypatch):
    """Gateway /handoff should run as an agent turn, not as an unknown command."""
    runner = _make_runner()
    captured = {}

    async def _capture(event, source, quick_key, run_generation):
        captured["text"] = event.text
        captured["quick_key"] = quick_key
        return {"final_response": "handoff ready"}

    runner._handle_message_with_agent = _capture  # type: ignore[method-assign]

    result = await runner._handle_message(_make_event("/handoff focus on tests"))

    assert result == {"final_response": "handoff ready"}
    assert "Create a concise but complete SESSION HANDOFF" in captured["text"]
    assert "Focus especially on: focus on tests" in captured["text"]
    assert captured["quick_key"] == build_session_key(_make_source())


@pytest.mark.asyncio
async def test_handoff_queues_next_turn_when_agent_is_running():
    """Gateway /handoff should not interrupt an active agent."""
    runner = _make_runner()
    source = _make_source()
    key = build_session_key(source)
    runner._running_agents[key] = SimpleNamespace(
        get_activity_summary=lambda: {"seconds_since_activity": 0}
    )
    runner._running_agents_ts[key] = datetime.now().timestamp()

    result = await runner._handle_message(_make_event("/handoff-new focus on restart"))

    assert "Queued /handoff-new" in result
    adapter = runner.adapters[Platform.TELEGRAM]
    assert key in adapter._pending_messages
    queued = adapter._pending_messages[key]
    assert "Create a concise but complete SESSION HANDOFF" in queued.text
    assert "Focus especially on: focus on restart" in queued.text
    assert "推奨コマンド:" in queued.text
    assert "- /new" in queued.text


@pytest.mark.asyncio
async def test_handoff_save_already_sent_sends_trailing_save_notice(monkeypatch, tmp_path):
    """When streaming already sent the body, gateway should still notify save path."""
    from hermes_cli.handoff import build_handoff_prompt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "c1")
    runner = _make_runner()
    source = _make_source()
    key = build_session_key(source)
    runner._session_run_generation = {key: 1}
    runner._running_agents_ts = {}
    runner._session_model_overrides = {}
    runner._set_session_reasoning_override = lambda *_args, **_kwargs: None
    runner._pending_model_notes = {}
    runner._prepare_inbound_message_text = AsyncMock()
    runner._bind_adapter_run_generation = MagicMock()
    runner._deliver_media_from_response = AsyncMock()

    prompt = build_handoff_prompt("handoff-save", hermes_home=tmp_path)
    runner._prepare_inbound_message_text.return_value = prompt
    handoff_response = "SESSION HANDOFF\n\n目的:\n- preserve context"
    runner._run_agent = AsyncMock(return_value={
        "final_response": handoff_response,
        "messages": [],
        "already_sent": True,
        "failed": False,
    })

    result = await runner._handle_message_with_agent(
        _make_event("/handoff-save"), source, key, 1,
    )

    assert result is None
    saved_files = list((tmp_path / "handoffs").glob("handoff_*.md"))
    assert len(saved_files) == 1
    assert saved_files[0].read_text(encoding="utf-8") == handoff_response + "\n"
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.send.assert_awaited_once()
    assert "Handoff saved:" in adapter.send.await_args.args[1]
    assert str(saved_files[0]) in adapter.send.await_args.args[1]
    assert adapter.send.await_args.kwargs["metadata"] is None


@pytest.mark.asyncio
async def test_handoff_save_helper_appends_notice_before_queued_followup_send(monkeypatch, tmp_path):
    """Queued follow-up recursion must save the current turn before recursing."""
    from gateway.run import _save_handoff_response_and_notice
    from hermes_cli.handoff import build_handoff_prompt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    prompt = build_handoff_prompt("handoff-save", hermes_home=tmp_path)
    handoff_response = "SESSION HANDOFF\n\n目的:\n- before queued follow-up"

    response, saved_path = await _save_handoff_response_and_notice(
        prompt,
        handoff_response,
        already_sent=False,
    )

    assert saved_path is not None
    assert saved_path.read_text(encoding="utf-8") == handoff_response + "\n"
    assert response.startswith(handoff_response)
    assert "Handoff saved:" in response
    assert str(saved_path) in response


@pytest.mark.asyncio
async def test_handoff_save_helper_sends_notice_when_queued_turn_was_streamed(monkeypatch, tmp_path):
    """If the first queued-turn response was streamed, only send a trailing save notice."""
    from gateway.run import _save_handoff_response_and_notice
    from hermes_cli.handoff import build_handoff_prompt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    prompt = build_handoff_prompt("handoff-save", hermes_home=tmp_path)
    handoff_response = "SESSION HANDOFF\n\n目的:\n- streamed before queued follow-up"
    adapter = MagicMock()
    adapter.send = AsyncMock()
    event = _make_event("/handoff-save")
    event.metadata = {"thread_id": "t1"}

    response, saved_path = await _save_handoff_response_and_notice(
        prompt,
        handoff_response,
        already_sent=True,
        adapter=adapter,
        chat_id="c1",
        event=event,
    )

    assert response == handoff_response
    assert saved_path is not None
    assert saved_path.read_text(encoding="utf-8") == handoff_response + "\n"
    adapter.send.assert_awaited_once()
    assert "Handoff saved:" in adapter.send.await_args.args[1]
    assert str(saved_path) in adapter.send.await_args.args[1]
    assert adapter.send.await_args.kwargs["metadata"] == {"thread_id": "t1"}


@pytest.mark.asyncio
async def test_handoff_save_run_agent_saves_before_queued_followup_recursion(monkeypatch, tmp_path):
    """Exercise the actual _run_agent recursion branch, not just the helper."""
    import run_agent
    from hermes_cli.handoff import build_handoff_prompt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")

    runner = _make_runner()
    source = _make_source()
    key = build_session_key(source)
    adapter = _FakeGatewayAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._session_run_generation = {key: 1}
    runner._running_agents_ts = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._pending_skills_reload_notes = {}
    runner._ephemeral_system_prompt = ""
    runner._MAX_INTERRUPT_DEPTH = 5
    runner._prefill_messages = []
    runner.session_store._entries = {}
    runner._resolve_session_agent_runtime = lambda **_kwargs: ("fake-model", {"provider": "fake"})
    runner._resolve_turn_agent_config = lambda _message, model, runtime: {
        "model": model,
        "runtime": runtime,
        "request_overrides": {},
    }
    runner._resolve_session_reasoning_config = lambda **_kwargs: None
    runner._load_service_tier = lambda: None
    runner._cleanup_agent_resources = lambda _agent: None

    async def _run_inline(fn):
        return fn()

    runner._run_in_executor_with_context = _run_inline
    runner._is_session_run_current = lambda _session_key, _generation: True
    runner._release_running_agent_state = lambda *_args, **_kwargs: None
    runner._promote_queued_event = lambda _session_key, _adapter, pending_event: pending_event
    runner._consume_pending_native_image_paths = lambda _session_key: []
    runner._prepare_inbound_message_text = AsyncMock(
        side_effect=lambda event, source, history: event.text
    )

    prompt = build_handoff_prompt("handoff-save", hermes_home=tmp_path)
    adapter._pending_messages[key] = _make_event("follow up after handoff")
    calls = []

    class _FakeAgent:
        def __init__(self, *args, **kwargs):
            self.tools = []
            self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
            self.session_prompt_tokens = 0
            self.session_completion_tokens = 0
            self.context_length = 0
            self.model = "fake-model"

        def run_conversation(self, user_message, conversation_history=None, task_id=None):
            calls.append(user_message)
            if len(calls) == 1:
                return {
                    "final_response": "SESSION HANDOFF\n\n目的:\n- actual recursion path",
                    "messages": [{"role": "assistant", "content": "handoff"}],
                    "api_calls": 1,
                    "tools": [],
                }
            return {
                "final_response": "follow-up complete",
                "messages": [{"role": "assistant", "content": "follow-up"}],
                "api_calls": 1,
                "tools": [],
            }

        def get_activity_summary(self):
            return {
                "seconds_since_activity": 0,
                "api_call_count": 1,
                "max_iterations": 90,
                "last_activity_desc": "done",
            }

        def interrupt(self, _message=None):
            return None

    monkeypatch.setattr(run_agent, "AIAgent", _FakeAgent)

    result = await runner._run_agent(
        prompt,
        "",
        [],
        source,
        "sess-1",
        key,
        1,
    )

    assert result["final_response"] == "follow-up complete"
    assert calls[0] == prompt
    assert calls[1] == "follow up after handoff"
    saved_files = list((tmp_path / "handoffs").glob("handoff_*.md"))
    assert len(saved_files) == 1
    assert saved_files[0].read_text(encoding="utf-8") == (
        "SESSION HANDOFF\n\n目的:\n- actual recursion path\n"
    )
    assert adapter.send.await_count == 1
    sent_text = adapter.send.await_args.args[1]
    assert "SESSION HANDOFF" in sent_text
    assert "Handoff saved:" in sent_text
    assert str(saved_files[0]) in sent_text


# ------------------------------------------------------------------
# command:<name> decision hook — deny / handled / rewrite
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_command_hook_can_deny_before_dispatch(monkeypatch):
    """A handler returning {"decision": "deny"} blocks a slash command early."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("denied slash command leaked to the agent")
    )
    runner._handle_status_command = AsyncMock(
        side_effect=AssertionError("denied slash command reached its handler")
    )
    runner.hooks.emit_collect = AsyncMock(
        return_value=[{"decision": "deny", "message": "Blocked by ACL"}]
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/status"))

    assert result == "Blocked by ACL"
    runner._run_agent.assert_not_called()
    # The emit_collect call should use the canonical command name.
    call_args = runner.hooks.emit_collect.await_args
    assert call_args.args[0] == "command:status"


@pytest.mark.asyncio
async def test_command_hook_deny_without_message_uses_default(monkeypatch):
    """A deny decision with no message falls back to a generic blocked string."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._handle_status_command = AsyncMock(
        side_effect=AssertionError("denied slash command reached its handler")
    )
    runner.hooks.emit_collect = AsyncMock(return_value=[{"decision": "deny"}])

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/status"))

    assert result is not None
    assert "blocked" in result.lower()


@pytest.mark.asyncio
async def test_command_hook_can_mark_command_as_handled(monkeypatch):
    """A handled decision short-circuits dispatch cleanly with a custom reply."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._handle_status_command = AsyncMock(
        side_effect=AssertionError("handled slash command reached its handler")
    )
    runner.hooks.emit_collect = AsyncMock(
        return_value=[{"decision": "handled", "message": "Already handled upstream"}]
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/status"))

    assert result == "Already handled upstream"


@pytest.mark.asyncio
async def test_command_hook_allow_decision_is_passthrough(monkeypatch):
    """A handler returning {"decision": "allow"} must NOT prevent normal dispatch."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._handle_status_command = AsyncMock(return_value="status: ok")
    runner.hooks.emit_collect = AsyncMock(
        return_value=[{"decision": "allow"}]
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/status"))

    assert result == "status: ok"
    runner._handle_status_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_command_hook_non_dict_return_values_ignored(monkeypatch):
    """Hook return values that aren't dicts must not break dispatch."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._handle_status_command = AsyncMock(return_value="status: ok")
    runner.hooks.emit_collect = AsyncMock(
        return_value=["some string", 42, None, {}]
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/status"))

    assert result == "status: ok"


@pytest.mark.asyncio
async def test_command_hook_fires_for_plugin_registered_command(monkeypatch):
    """Plugin-registered slash commands should also trigger command:<name> hooks."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("plugin command leaked to the agent")
    )
    runner.hooks.emit_collect = AsyncMock(
        return_value=[{"decision": "handled", "message": "intercepted"}]
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    # Stub plugin command lookup so is_gateway_known_command() recognizes /metricas.
    from hermes_cli import plugins as _plugins_mod

    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_commands",
        lambda: {"metricas": {"description": "Metrics", "args_hint": "dias:7"}},
    )

    result = await runner._handle_message(_make_event("/metricas dias:7"))

    assert result == "intercepted"
    # Hook event name uses the plugin command as canonical.
    call_args = runner.hooks.emit_collect.await_args
    assert call_args.args[0] == "command:metricas"
    # Args are passed through in both "args" and "raw_args" keys.
    ctx = call_args.args[1]
    assert ctx["raw_args"] == "dias:7"


@pytest.mark.asyncio
async def test_command_hook_rewrite_routes_to_plugin(monkeypatch):
    """A rewrite decision should re-resolve the command and route to the new one."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("rewritten command leaked to the agent")
    )

    call_log = []

    async def _emit_collect(event_type, ctx):
        call_log.append(event_type)
        if event_type == "command:status":
            return [
                {
                    "decision": "rewrite",
                    "command_name": "metricas",
                    "raw_args": "dias:7",
                }
            ]
        return []

    runner.hooks.emit_collect = AsyncMock(side_effect=_emit_collect)

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    from hermes_cli import plugins as _plugins_mod

    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_commands",
        lambda: {"metricas": {"description": "Metrics", "args_hint": "dias:7"}},
    )
    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_command_handler",
        lambda name: (lambda args: f"metrics {args}") if name == "metricas" else None,
    )

    result = await runner._handle_message(_make_event("/status"))

    assert result == "metrics dias:7"
    # First emit_collect fires on the original command; after rewrite the
    # dispatcher does NOT re-fire for the new command (one decision per turn).
    assert call_log == ["command:status"]
