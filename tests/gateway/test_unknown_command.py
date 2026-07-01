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
async def test_mcpstatus_is_read_only_and_redacted(monkeypatch):
    """The mobile MCP status surface must not leak config, URLs, or secrets."""
    import gateway.run as gateway_run
    import tools.mcp_tool as mcp_tool

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/mcpstatus leaked through to the agent")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        mcp_tool,
        "get_mcp_status",
        lambda: [
            {
                "name": "linear",
                "transport": "http",
                "tools": 7,
                "connected": True,
                "disabled": False,
                "status": "connected",
                "url": "https://secret.example/mcp",
                "headers": {"Authorization": "Bearer sk-secret"},
                "env": {"TOKEN": "env-secret"},
            },
            {
                "name": "broken",
                "transport": "stdio",
                "tools": 0,
                "connected": False,
                "disabled": False,
                "status": "failed",
                "error": "failed with token sk-secret at https://secret.example",
            },
        ],
    )

    result = await runner._handle_message(_make_event("/mcp_status"))

    assert result is not None
    assert "MCP Status" in result
    assert "Configured servers: 2" in result
    assert "Connected servers: 1" in result
    assert "Tools available: 7" in result
    assert "- linear: connected, 7 tool(s), http" in result
    assert "- broken: failed, 0 tool(s), stdio" in result
    assert "Read-only" in result
    assert "secret.example" not in result
    assert "sk-secret" not in result
    assert "env-secret" not in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_mcpstatus_bypasses_active_session_as_read_only(monkeypatch):
    import gateway.run as gateway_run
    import tools.mcp_tool as mcp_tool

    runner = _make_runner()
    running_agent = MagicMock()
    runner._running_agents[build_session_key(_make_source())] = running_agent
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/mcpstatus leaked through to the agent")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        mcp_tool,
        "get_mcp_status",
        lambda: [
            {
                "name": "linear",
                "transport": "http",
                "tools": 7,
                "connected": True,
                "disabled": False,
                "status": "connected",
            }
        ],
    )

    result = await runner._handle_message(_make_event("/mcpstatus"))

    assert result is not None
    assert "MCP Status" in result
    assert "Agent is running" not in result
    running_agent.interrupt.assert_not_called()
    runner._run_agent.assert_not_called()


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


@pytest.mark.asyncio
async def test_mcpdoctor_is_read_only_redacted_and_actionable(monkeypatch):
    import gateway.run as gateway_run
    import tools.mcp_tool as mcp_tool

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/mcpdoctor leaked through to the agent")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        mcp_tool,
        "get_mcp_status",
        lambda: [
            {
                "name": "linear",
                "transport": "http",
                "tools": 7,
                "connected": True,
                "disabled": False,
                "status": "connected",
                "url": "https://secret.example/mcp",
                "headers": {"Authorization": "Bearer sk-secret"},
            },
            {
                "name": "github",
                "transport": "stdio",
                "tools": 0,
                "connected": False,
                "disabled": False,
                "status": "configured",
                "env": {"TOKEN": "env-secret"},
            },
            {
                "name": "broken",
                "transport": "stdio",
                "tools": 0,
                "connected": False,
                "disabled": False,
                "status": "failed",
                "error": "failed with token sk-secret at https://secret.example",
            },
            {
                "name": "off",
                "transport": "stdio",
                "tools": 0,
                "connected": False,
                "disabled": True,
                "status": "disabled",
            },
        ],
    )

    result = await runner._handle_message(_make_event("/mcp_doctor"))

    assert result is not None
    assert "MCP Doctor" in result
    assert "Configured servers: 4" in result
    assert "Connected servers: 1" in result
    assert "Configured-only servers: 1" in result
    assert "Failed servers: 1" in result
    assert "Disabled servers: 1" in result
    assert "- linear: connected; reason: connected; auto-fix: no; next: none" in result
    assert "github: configured; reason: configured but not loaded in this gateway process" in result
    assert "broken: failed; reason: last connection failed" in result
    assert "off: disabled; reason: disabled by config" in result
    assert "Read-only" in result
    assert "secret.example" not in result
    assert "sk-secret" not in result
    assert "env-secret" not in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_mcpdoctor_bypasses_active_session_as_read_only(monkeypatch):
    import gateway.run as gateway_run
    import tools.mcp_tool as mcp_tool

    runner = _make_runner()
    running_agent = MagicMock()
    runner._running_agents[build_session_key(_make_source())] = running_agent
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/mcpdoctor leaked through to the agent")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        mcp_tool,
        "get_mcp_status",
        lambda: [
            {
                "name": "github",
                "transport": "stdio",
                "tools": 0,
                "connected": False,
                "disabled": False,
                "status": "configured",
            }
        ],
    )

    result = await runner._handle_message(_make_event("/mcpdoctor"))

    assert result is not None
    assert "MCP Doctor" in result
    assert "Agent is running" not in result
    running_agent.interrupt.assert_not_called()
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_mcpreloadplan_is_read_only_redacted_and_classifies_risk(monkeypatch):
    import gateway.run as gateway_run
    import tools.mcp_tool as mcp_tool

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/mcpreloadplan leaked through to the agent")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        mcp_tool,
        "get_mcp_status",
        lambda: [
            {
                "name": "linear",
                "transport": "http",
                "tools": 7,
                "connected": True,
                "disabled": False,
                "status": "connected",
                "url": "https://secret.example/mcp",
                "headers": {"Authorization": "Bearer sk-secret"},
            },
            {
                "name": "github",
                "transport": "stdio",
                "tools": 0,
                "connected": False,
                "disabled": False,
                "status": "configured",
                "env": {"TOKEN": "env-secret"},
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
            },
            {
                "name": "broken",
                "transport": "sse",
                "tools": 0,
                "connected": False,
                "disabled": False,
                "status": "failed",
                "error": "failed with token sk-secret at https://secret.example",
            },
            {
                "name": "off",
                "transport": "stdio",
                "tools": 0,
                "connected": False,
                "disabled": True,
                "status": "disabled",
            },
            {
                "name": "mystery",
                "transport": "custom",
                "tools": 0,
                "connected": False,
                "disabled": False,
                "status": "configured",
            },
        ],
    )

    result = await runner._handle_message(_make_event("/mcp_reload_plan"))

    assert result is not None
    assert "MCP Reload Plan" in result
    assert "Configured servers: 5" in result
    assert "Connected servers: 1" in result
    assert "Planned reload candidates: 3" in result
    assert "Risk summary: low=1 / medium=1 / high=1 / parked=1 / unknown=1" in result
    assert "linear: state=connected; risk=low; surface=already-connected; safe-auto=no" in result
    assert "github: state=configured; risk=medium; surface=local-process; safe-auto=no" in result
    assert "broken: state=failed; risk=high; surface=sse; safe-auto=no" in result
    assert "off: state=disabled; risk=parked; surface=none; safe-auto=no" in result
    assert "mystery: state=configured; risk=unknown; surface=unknown; safe-auto=no" in result
    assert "Read-only" in result
    assert "secret.example" not in result
    assert "sk-secret" not in result
    assert "env-secret" not in result
    assert "npx" not in result
    assert "@modelcontextprotocol" not in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_mcpreloadplan_bypasses_active_session_as_read_only(monkeypatch):
    import gateway.run as gateway_run
    import tools.mcp_tool as mcp_tool

    runner = _make_runner()
    running_agent = MagicMock()
    runner._running_agents[build_session_key(_make_source())] = running_agent
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/mcpreloadplan leaked through to the agent")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        mcp_tool,
        "get_mcp_status",
        lambda: [
            {
                "name": "github",
                "transport": "stdio",
                "tools": 0,
                "connected": False,
                "disabled": False,
                "status": "configured",
            }
        ],
    )

    result = await runner._handle_message(_make_event("/mcpreloadplan"))

    assert result is not None
    assert "MCP Reload Plan" in result
    assert "Agent is running" not in result
    running_agent.interrupt.assert_not_called()
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_capabilitystatus_is_read_only_summary(monkeypatch):
    import gateway.run as gateway_run
    import tools.mcp_tool as mcp_tool

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/capabilitystatus leaked through to the agent")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        mcp_tool,
        "get_mcp_status",
        lambda: [
            {
                "name": "fetch",
                "transport": "stdio",
                "tools": 3,
                "connected": True,
                "disabled": False,
                "status": "connected",
                "env": {"TOKEN": "env-secret"},
                "url": "https://secret.example/mcp",
            },
            {
                "name": "github",
                "transport": "stdio",
                "tools": 0,
                "connected": False,
                "disabled": False,
                "status": "configured",
            },
        ],
    )

    result = await runner._handle_message(_make_event("/capability-status"))

    assert result is not None
    assert "Hermes Capability Status" in result
    assert "MoA: available" in result
    assert "MCP: 1/2 connected, 3 tool(s)" in result
    assert "MCP status tools: available / available / available" in result
    assert "DevFlow: available / available" in result
    assert "Kanban: available" in result
    assert "Delegation: available" in result
    assert "Learning loop:" in result
    assert "Automation:" in result
    assert "Browser & media tools:" in result
    assert "Search & vision tools:" in result
    assert "Execution & local tools:" in result
    assert "Context & session controls:" in result
    assert "Upstream freshness:" in result
    assert "Read-only" in result
    assert "tool availability probe" in result
    assert "secret.example" not in result
    assert "env-secret" not in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_capabilitystatus_bypasses_active_session_as_read_only(monkeypatch):
    import gateway.run as gateway_run
    import tools.mcp_tool as mcp_tool

    runner = _make_runner()
    running_agent = MagicMock()
    runner._running_agents[build_session_key(_make_source())] = running_agent
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/capabilitystatus leaked through to the agent")
    )

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        mcp_tool,
        "get_mcp_status",
        lambda: [
            {
                "name": "fetch",
                "transport": "stdio",
                "tools": 3,
                "connected": True,
                "disabled": False,
                "status": "connected",
            }
        ],
    )

    result = await runner._handle_message(_make_event("/capabilitystatus"))

    assert result is not None
    assert "Hermes Capability Status" in result
    assert "Agent is running" not in result
    running_agent.interrupt.assert_not_called()
    runner._run_agent.assert_not_called()
