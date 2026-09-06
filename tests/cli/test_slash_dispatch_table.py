"""Parity guard for the slash-command dispatch table in cli.HermesCLI.

Every canonical command that had a branch in the old if/elif chain must
resolve to a handler in ``_SLASH_DISPATCH``, and the pre-dispatch side effects
(pre_command hook, pending-resume reset, unknown-command fallthrough) must
keep their old semantics.
"""
from unittest.mock import MagicMock, patch

from cli import HermesCLI

# Command names that had an explicit branch in the pre-dispatch-table chain.
OLD_CHAIN_COMMANDS = [
    "exit", "quit", "help", "palette", "whoami", "profile", "tools", "toolsets",
    "config", "redraw", "clear", "history", "title", "handoff", "new", "resume",
    "sessions", "model", "codex-runtime", "personality", "pet", "hatch", "retry",
    "prompt", "undo", "branch", "worktree", "save", "cron", "suggestions",
    "blueprint", "curator", "kanban", "skills", "learn", "init", "memory",
    "platforms", "status", "context", "egress", "statusbar", "diff", "battery",
    "timestamps", "verbose", "focus", "footer", "yolo", "approvals", "reasoning",
    "fast", "compress", "usage", "subscription", "topup", "insights", "copy",
    "debug", "update", "version", "paste", "image", "reload", "reload-mcp",
    "reload-skills", "bundles", "browser", "plugins", "rollback", "snapshot",
    "export", "import", "stop", "agents", "journey", "bg", "btw", "queue",
    "steer", "goal", "heartbeat", "refine", "review", "loop", "plan", "moa",
    "subgoal", "skin", "voice", "wake", "busy", "indicator",
]


def test_every_old_branch_resolves_to_a_handler():
    for name in OLD_CHAIN_COMMANDS:
        entry = HermesCLI._slash_handler(name)
        assert entry is not None, name
        method_name, pass_arg = entry
        assert callable(getattr(HermesCLI, method_name)), name
        assert isinstance(pass_arg, bool)
    # explicit table entries are only the ones the naming convention can't cover
    for name, (method_name, pass_arg) in HermesCLI._SLASH_DISPATCH.items():
        assert name in OLD_CHAIN_COMMANDS
        assert (method_name, pass_arg) != (f"_handle_{name.replace('-', '_')}_command", True), name


def test_registry_names_resolve_into_the_table():
    from hermes_cli.commands import COMMAND_REGISTRY, resolve_command

    for name in HermesCLI._SLASH_DISPATCH:
        cmd = resolve_command(name)
        assert cmd is not None and HermesCLI._slash_handler(cmd.name) is not None, name
    # registry commands the CLI never handled inline must still fall through
    dispatched = {c.name for c in COMMAND_REGISTRY if HermesCLI._slash_handler(c.name)}
    assert dispatched == set(OLD_CHAIN_COMMANDS) - {"exit"} | {"quit"}


def _cli():
    c = HermesCLI.__new__(HermesCLI)
    c._pending_resume_sessions = ["x"]
    c.session_id = "s1"
    c.config = {}
    return c


def test_dispatch_return_semantics_and_side_effects():
    c = _cli()
    with patch.object(HermesCLI, "_toggle_yolo", return_value=None) as m, \
            patch("hermes_cli.plugins.fire_pre_command_hook") as hook:
        assert c.process_command("/yolo") is True
        m.assert_called_once_with()
        hook.assert_called_once()
        assert hook.call_args.kwargs["command"] == "yolo"
    assert c._pending_resume_sessions is None  # non-resume command disarms it

    c = _cli()
    with patch.object(HermesCLI, "_handle_resume_command") as m:
        assert c.process_command("/resume 2") is True
        m.assert_called_once_with("/resume 2")
    assert c._pending_resume_sessions == ["x"]

    c = _cli()
    assert c.process_command("/exit") is False
    c = _cli()
    with patch.object(HermesCLI, "_handle_handoff_command", return_value=False):
        assert c.process_command("/handoff telegram") is False
    with patch.object(HermesCLI, "_handle_handoff_command", return_value=True):
        assert c.process_command("/handoff telegram") is True
    with patch.object(HermesCLI, "_handle_update_command", return_value=True):
        assert c.process_command("/update") is False
    with patch.object(HermesCLI, "_handle_update_command", return_value=False):
        assert c.process_command("/update") is True


def test_unknown_command_falls_through():
    c = _cli()
    c._console_print = MagicMock()
    with patch.object(HermesCLI, "_process_unregistered_slash", return_value=True) as m:
        assert c.process_command("/definitely-not-a-command x") is True
        m.assert_called_once_with("/definitely-not-a-command x", "/definitely-not-a-command x")
