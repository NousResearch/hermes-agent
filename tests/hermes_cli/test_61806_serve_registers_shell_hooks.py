"""
Regression test for issue #61806 - shell hooks (pre_tool_call) never
registered in `hermes serve`.

The fix: add `"serve"` to the `_AGENT_COMMANDS` set in
`hermes_cli/main.py` so `_prepare_agent_startup` registers shell hooks
when running under `hermes serve`.

Engine-run note: also adds an INFO log line on successful hook
registration, and a WARNING log on registration failure. The
WARNING on failure is the security-visibility fix - on `hermes serve`,
a policy bypass is now visible in logs (operator can grep for the
warning) instead of being silent.

This test pins the gate's contents and the log behavior. Fails on
unfixed code.
"""

import logging
import re
import types
from pathlib import Path
from unittest import mock


def test_serve_in_agent_commands_set():
    """Static-source tripwire."""
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    main_py = (worktree / "hermes_cli" / "main.py").read_text()
    m = re.search(r"_AGENT_COMMANDS\s*=\s*\{([^}]*)\}", main_py)
    assert m, "_AGENT_COMMANDS set not found"
    set_content = m.group(1)
    assert '"serve"' in set_content, (
        f"#61806: 'serve' missing from _AGENT_COMMANDS: {set_content!r}"
    )


def test_registration_failure_logs_warning():
    """Static-source tripwire: the registration-failure path must log
    at WARNING (not DEBUG) so the operator can see policy bypasses.
    Engine-run addition: this is the security-visibility fix that the
    MCE pre-mortem 'verification skips a layer' warned about."""
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    main_py = (worktree / "hermes_cli" / "main.py").read_text()

    # Find the register_from_config block
    m = re.search(
        r"shell-hook registration failed at CLI startup\s*\(([^)]*)\)",
        main_py,
    )
    assert m, "registration-failure log statement not found"
    # The level is the first non-whitespace word before the message
    prefix = m.group(1)
    # Check that this is a logger.warning call, not logger.debug
    # (the logger.X call comes BEFORE the message in source)
    start = max(0, m.start() - 50)
    before = main_py[start:m.start()]
    assert "logger.warning" in before, (
        f"#61806: registration failure should log at WARNING, not DEBUG. "
        f"Found: ...{before!r}"
    )


def test_successful_registration_logs_info():
    """Static-source tripwire: successful hook registration logs at
    INFO so the operator can verify hooks are wired up. This is the
    engine's value-add for visibility."""
    worktree = Path("/tmp/hermes-pr-work-60859/hermes-agent")
    main_py = (worktree / "hermes_cli" / "main.py").read_text()
    assert "shell hooks registered" in main_py, (
        "#61806: should log 'shell hooks registered' on successful registration"
    )


def test_prepare_agent_startup_registers_hooks_for_serve(monkeypatch):
    """Behavioral test."""
    from hermes_cli import main as hermes_main
    args = types.SimpleNamespace(command="serve", accept_hooks=False)
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
    assert len(register_called) == 1
