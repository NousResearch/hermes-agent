"""Tests for gateway warning when an unrecognized /command is dispatched.

Without this warning, unknown slash commands get forwarded to the LLM as plain
text, which often leads to silent failure (e.g. the model inventing a bogus
delegate_task call instead of telling the user the command doesn't exist).
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
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


def _make_voice_event(text: str = "voice_message_1.ogg") -> MessageEvent:
    source = _make_source()
    return MessageEvent(
        text=text,
        message_type=MessageType.VOICE,
        source=source,
        message_id="m1",
        media_urls=["/tmp/voice_message_1.ogg"],
        media_types=["audio/ogg"],
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
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
async def test_egress_slash_command_reports_proxy_status(monkeypatch):
    runner = _make_runner()
    monkeypatch.setattr(
        "hermes_cli.proxy_cli.format_status_text",
        lambda: "Egress proxy status\nEnabled: no",
    )

    result = await runner._handle_message(_make_event("/egress"))

    assert result is not None
    assert "Egress proxy status" in result
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


# ------------------------------------------------------------------
# command:<name> decision hook — deny / handled / rewrite
# ------------------------------------------------------------------


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


@pytest.mark.asyncio
async def test_gateway_plugin_command_receives_authenticated_context(monkeypatch):
    """Gateway plugin slash commands receive immutable source metadata after auth."""
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("plugin slash command leaked to the agent")
    )
    runner.session_store.peek_session_id.return_value = "sess-1"

    received_contexts = []

    def _handler(raw_args, *, command_context=None):
        received_contexts.append(command_context)
        return (
            f"{raw_args}:"
            f"{command_context.platform}:"
            f"{command_context.user_id}:"
            f"{command_context.user_name}:"
            f"{command_context.chat_id}:"
            f"{command_context.chat_type}:"
            f"{command_context.message_id}:"
            f"{command_context.authorized}"
        )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    from hermes_cli import plugins as _plugins_mod

    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_commands",
        lambda: {"audit": {"description": "Audit command"}},
    )
    monkeypatch.setattr(
        _plugins_mod,
        "get_plugin_command_handler",
        lambda name: _handler if name == "audit" else None,
    )

    event = _make_event("/audit approve")
    result = await runner._handle_message(event)

    assert result == "approve:telegram:u1:tester:c1:dm:m1:True"
    assert received_contexts
    assert received_contexts[0].session_id == "sess-1"
    runner.session_store.peek_session_id.assert_called_once_with(
        build_session_key(event.source)
    )
    runner.session_store.get_or_create_session.assert_not_called()
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        received_contexts[0].user_id = "mutated"


@pytest.fixture
def discovered_command(tmp_path, monkeypatch):
    """Load an external plugin through the real profile-scoped discovery path."""
    from hermes_cli import plugins
    from gateway.session import SessionStore

    home = tmp_path / "home"
    plugin_dir = home / "plugins" / "context-probe"
    plugin_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [context-probe]\n", encoding="utf-8"
    )
    (plugin_dir / "plugin.yaml").write_text(
        "name: context-probe\nversion: 0.1.0\n", encoding="utf-8"
    )
    (plugin_dir / "__init__.py").write_text(
        "received = []\n"
        "async def command(raw_args, *, command_context):\n"
        "    received.append((raw_args, command_context))\n"
        "    return 'plugin ran'\n"
        "def register(ctx):\n"
        "    ctx.register_command('context-probe', command)\n",
        encoding="utf-8",
    )
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    manager.discover_and_load()
    handler = plugins.get_plugin_command_handler("context-probe")
    assert handler is not None
    runner = _make_runner()
    runner.session_store = SessionStore(home / "sessions", runner.config)
    runner._run_agent = AsyncMock(side_effect=AssertionError("plugin leaked to agent"))
    yield runner, handler.__globals__["received"]
    runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
async def test_discovered_plugin_context_uses_existing_session_only(discovered_command, existing):
    runner, received = discovered_command
    event = _make_event("/context_probe preserve  these args")
    event.source.chat_name = "Test chat"
    event.source.thread_id = "topic-1"
    event.source.guild_id = "guild-1"
    event.source.message_id = "source-message"
    entry = runner.session_store.get_or_create_session(event.source) if existing else None
    runner.session_store.get_or_create_session = MagicMock(
        side_effect=AssertionError("plugin command must not create or touch a session")
    )

    assert await runner._handle_message(event) == "plugin ran"
    assert len(received) == 1
    raw_args, context = received[0]
    assert raw_args == "preserve  these args"
    assert context.session_id == (entry.session_id if entry else None)
    assert (context.platform, context.user_id, context.chat_id) == ("telegram", "u1", "c1")
    assert (context.chat_name, context.thread_id, context.guild_id) == ("Test chat", "topic-1", "guild-1")
    assert context.message_id == "source-message"
    assert context.authorized is True
    runner.session_store.get_or_create_session.assert_not_called()
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("denial", ["sender", "slash"])
async def test_discovered_plugin_not_invoked_before_authorization(discovered_command, denial):
    runner, received = discovered_command
    if denial == "sender":
        runner._is_user_authorized = lambda _source: False
    else:
        runner.config.platforms[Platform.TELEGRAM].extra.update(
            allow_admin_from=["admin"], user_allowed_commands=[]
        )
    await runner._handle_message(_make_event("/context_probe status"))
    assert received == []
    runner._run_agent.assert_not_called()
