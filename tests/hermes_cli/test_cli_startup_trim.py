"""Startup/exit work that belongs to the gateway never runs in the classic CLI's process.

Rewired, every chat turn executes in the gateway session, which owns MCP servers, shell hooks,
outbound webhooks, state.db / checkpoint maintenance and async delegations. The CLI process must
not spawn a second copy of an MCP server, register hooks that never fire, run VACUUM as a second
writer, or tear any of those down on exit.
"""
import sys
import types
from argparse import Namespace

import pytest


def _args(command):
    return Namespace(accept_hooks=True, command=command, cron_command=None, gateway_command=None,
                     mcp_action=None, tui=False, yolo=False, safe_mode=False, ignore_user_config=False)


@pytest.mark.parametrize("command", [None, "chat"])
def test_chat_launch_registers_no_shell_hooks_or_webhooks(monkeypatch, command):
    """... and ``--accept-hooks`` still reaches the gateway: exported as HERMES_ACCEPT_HOOKS, which
    rides session.create (env_launch_flags) and the classic CLI's consent check (CLI-5)."""
    from hermes_cli import main as main_mod
    calls = []
    monkeypatch.setenv("HERMES_ACCEPT_HOOKS", "")  # recorded, so the export below is undone at teardown
    monkeypatch.delenv("HERMES_ACCEPT_HOOKS")
    monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda _a: False)
    monkeypatch.setattr(main_mod, "_bypass_chat_launch", lambda _a: False, raising=False)
    monkeypatch.setattr(main_mod, "_guard_noninteractive_user_config", lambda _a: None)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", types.SimpleNamespace(
        discover_plugins=lambda: None, start_background_plugin_discovery=lambda: None))
    monkeypatch.setitem(sys.modules, "agent.shell_hooks", types.SimpleNamespace(
        register_from_config=lambda *a, **k: calls.append("hooks")))
    monkeypatch.setitem(sys.modules, "agent.outbound_webhooks", types.SimpleNamespace(
        register_from_config=lambda *a, **k: calls.append("webhooks")))
    main_mod._prepare_agent_startup(_args(command))
    assert calls == []
    assert main_mod.os.environ["HERMES_ACCEPT_HOOKS"] == "1"


def test_non_chat_agent_command_still_registers_hooks(monkeypatch):
    from hermes_cli import main as main_mod
    calls = []
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", types.SimpleNamespace(
        discover_plugins=lambda: None, start_background_plugin_discovery=lambda: None))
    monkeypatch.setitem(sys.modules, "hermes_cli.config", types.SimpleNamespace(
        load_config=dict, read_raw_config=dict))
    monkeypatch.setitem(sys.modules, "agent.shell_hooks", types.SimpleNamespace(
        register_from_config=lambda *a, **k: calls.append("hooks")))
    monkeypatch.setitem(sys.modules, "agent.outbound_webhooks", types.SimpleNamespace(
        register_from_config=lambda *a, **k: calls.append("webhooks")))
    from hermes_cli import mcp_startup
    monkeypatch.setattr(mcp_startup, "start_background_mcp_discovery", lambda **k: None)
    main_mod._prepare_agent_startup(_args("rl"))
    assert calls == ["hooks", "webhooks"]


@pytest.mark.parametrize(("command", "subcommand", "interactive"), [
    ("gateway", {"gateway_command": "run"}, False),  # a service: a hidden console is not a person (#127822)
    ("cron", {"cron_command": "tick"}, False),       # scheduled: nobody at the console
    ("acp", {}, False),                              # stdin carries the protocol stream
    ("rl", {}, True),                                # a foreground command may still ask at its terminal
])
def test_only_foreground_commands_may_prompt_for_hook_consent(monkeypatch, command, subcommand, interactive):
    from hermes_cli import main as main_mod
    seen = []
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", types.SimpleNamespace(
        discover_plugins=lambda: None, start_background_plugin_discovery=lambda: None))
    monkeypatch.setitem(sys.modules, "hermes_cli.config", types.SimpleNamespace(
        load_config=dict, read_raw_config=dict))
    monkeypatch.setitem(sys.modules, "agent.shell_hooks", types.SimpleNamespace(
        register_from_config=lambda *a, **k: seen.append(k)))
    monkeypatch.setitem(sys.modules, "agent.outbound_webhooks", types.SimpleNamespace(
        register_from_config=lambda *a, **k: None))
    from hermes_cli import mcp_startup
    monkeypatch.setattr(mcp_startup, "start_background_mcp_discovery", lambda **k: None)
    args = _args(command)
    args.accept_hooks = False
    for name, value in subcommand.items():
        setattr(args, name, value)
    main_mod._prepare_agent_startup(args)
    assert seen == [{"accept_hooks": False, "interactive": interactive}]


def test_deferred_first_turn_discovers_no_plugins_spawns_no_mcp_and_registers_no_hooks(monkeypatch):
    # Termux's deferred startup ran plugin discovery before the first turn; the gateway runs the turn
    # and loads the plugins (PLG-7).
    import cli as cli_mod
    from hermes_cli import gateway_agent
    from hermes_cli import mcp_startup
    calls = []
    monkeypatch.setenv("HERMES_DEFER_AGENT_STARTUP", "1")
    cli = cli_mod.HermesCLI(compact=True)
    import hermes_cli.plugins as plugins_mod
    monkeypatch.setattr(plugins_mod, "discover_plugins", lambda *a, **k: calls.append("plugins"))
    monkeypatch.setattr(mcp_startup, "start_background_mcp_discovery", lambda **k: calls.append("mcp"))
    monkeypatch.setattr(mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **k: calls.append("mcp"))
    monkeypatch.setitem(sys.modules, "agent.shell_hooks", types.SimpleNamespace(
        register_from_config=lambda *a, **k: calls.append("hooks")))
    monkeypatch.setitem(sys.modules, "agent.outbound_webhooks", types.SimpleNamespace(
        register_from_config=lambda *a, **k: calls.append("webhooks")))
    monkeypatch.setattr(gateway_agent, "GatewayAgent", lambda *_a, **_k: types.SimpleNamespace())
    cli._session_db, cli._resumed, cli.conversation_history = object(), False, []
    cli._install_tool_callbacks = lambda: None
    cli._ensure_runtime_credentials = lambda: True
    cli._ensure_gateway_session = lambda: None
    assert cli._init_agent() is True
    assert calls == []


def test_init_agent_runs_no_mcp_discovery(monkeypatch):
    import cli as cli_mod
    from hermes_cli import gateway_agent
    from hermes_cli import mcp_startup
    calls = []
    monkeypatch.setattr(mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **k: calls.append("mcp"))
    monkeypatch.setattr(mcp_startup, "start_background_mcp_discovery", lambda **k: calls.append("mcp-bg"))
    monkeypatch.setattr(gateway_agent, "GatewayAgent", lambda *_a, **_k: types.SimpleNamespace())
    cli = cli_mod.HermesCLI(compact=True)
    cli._session_db = object()
    cli._resumed = False
    cli.conversation_history = []
    cli._install_tool_callbacks = lambda: None
    cli._ensure_runtime_credentials = lambda: True
    cli._ensure_gateway_session = lambda: None
    assert cli._init_agent() is True
    assert calls == []


def test_session_store_open_runs_no_state_db_or_checkpoint_maintenance(monkeypatch):
    import cli as cli_mod
    from cli import HermesCLI
    calls = []
    monkeypatch.setattr(cli_mod, "_run_state_db_auto_maintenance", lambda db: calls.append("state.db"))
    monkeypatch.setattr(cli_mod, "_run_checkpoint_auto_maintenance", lambda: calls.append("checkpoints"))
    import tools.checkpoint_maintenance as ckpt
    monkeypatch.setattr(ckpt, "auto_prune_from_config", lambda *a, **k: calls.append("checkpoints"))
    cli = types.SimpleNamespace()
    HermesCLI._init_session_store(cli)
    assert calls == []


def test_exit_cleanup_leaves_gateway_owned_mcp_and_delegations_alone():
    from hermes_cli.cli_shutdown import _CLEANUP_STEPS
    names = {name for name, _exc in _CLEANUP_STEPS}
    assert "_shutdown_mcp_servers" not in names
    assert "_interrupt_async_delegations" not in names
    assert {"_cleanup_all_terminals", "_shutdown_cached_aux_clients"} <= names
