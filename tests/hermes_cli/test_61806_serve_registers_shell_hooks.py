"""
Regression test for issue #61806 - shell hooks (pre_tool_call) never
registered in `hermes serve`. Any `pre_tool_call` security policy is
silently bypassed on that surface.

The fix: add `"serve"` to the `_AGENT_COMMANDS` set in
`hermes_cli/main.py` so `_prepare_agent_startup` registers shell hooks
when running under `hermes serve` (Hermes Desktop remote-gateway mode,
dashboard /api/ws chat).

This test pins the gate's contents so a future refactor cannot silently
regress and remove "serve" from the set. It is a static-source tripwire
(verifying the literal in the source file), plus a behavioral test that
confirms `_prepare_agent_startup` does NOT early-return for
`args.command == "serve"`.
"""

import types
from pathlib import Path
from unittest import mock


def test_serve_in_agent_commands_set():
    """Static-source tripwire: the literal "serve" must be present in
    _AGENT_COMMANDS in hermes_cli/main.py. Fails on unfixed code (where
    the gate excludes "serve"), passes on fixed code."""
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    main_py = (worktree / "hermes_cli" / "main.py").read_text()

    # Find the _AGENT_COMMANDS assignment and check for "serve"
    import re
    m = re.search(r"_AGENT_COMMANDS\s*=\s*\{([^}]*)\}", main_py)
    assert m, "_AGENT_COMMANDS set not found in hermes_cli/main.py"
    set_content = m.group(1)
    assert '"serve"' in set_content, (
        f"#61806 regression: 'serve' is not in _AGENT_COMMANDS, so shell "
        f"hooks are not registered in `hermes serve`. Set content: {set_content!r}"
    )


def test_prepare_agent_startup_registers_hooks_for_serve(monkeypatch):
    """Behavioral test: when args.command == "serve",
    _prepare_agent_startup must register shell hooks."""
    from hermes_cli import main as hermes_main

    # Mock args
    args = types.SimpleNamespace(command="serve", accept_hooks=False)

    # Track whether register_from_config was called
    register_called = []

    def fake_register(config, accept_hooks=False):
        register_called.append((config, accept_hooks))

    fake_shell_hooks = types.SimpleNamespace(register_from_config=fake_register)
    fake_plugins = types.SimpleNamespace(discover_plugins=lambda: None)
    fake_mcp_tool = types.SimpleNamespace(discover_mcp_tools=lambda: None)

    with mock.patch.dict("sys.modules", {
        "agent.shell_hooks": fake_shell_hooks,
        "tools.mcp_tool": fake_mcp_tool,
    }), mock.patch("hermes_cli.config.load_config", return_value={"hooks": {"pre_tool_call": []}}), \
         mock.patch("hermes_cli.plugins.discover_plugins", fake_plugins.discover_plugins), \
         mock.patch.object(hermes_main, "_apply_safe_mode", lambda a: None), \
         mock.patch.object(hermes_main, "_is_tui_chat_launch", lambda a: False), \
         mock.patch.object(hermes_main, "_command_has_dedicated_mcp_startup", lambda a: False), \
         mock.patch.object(hermes_main, "_should_background_mcp_startup", lambda a: False):
        hermes_main._prepare_agent_startup(args)

    assert len(register_called) == 1, (
        f"#61806: shell hooks should be registered for 'serve' command, "
        f"but register_from_config was called {len(register_called)} times"
    )
