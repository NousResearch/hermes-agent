"""Regression coverage for the delegated-child env-marker leak (t_799fe7a8).

A top-level agent session process (``hermes chat``, ``-q``/oneshot, ACP, RL,
``gateway run``, ``cron run/tick``, ``mcp serve``) is a ROOT delegation
context: delegate_task marks children with a ContextVar *in-process*
(agent/delegation_context.py), never by environment heredity.  But the env
marker HERMES_DELEGATED_CHILD_CONTEXT crosses fork/exec boundaries, so a
session launched by an ancestor that carried the marker (e.g. a Kanban worker
shelling out to ``hermes chat -q``, or an execute_code kernel spawned under
such a CLI) inherited a *false-positive* delegated-child identity.  Every one
of that session's own Kanban writes then failed closed with "delegate_task
child contexts cannot mutate Kanban tasks or boards"
(kanban_db._assert_not_delegated_child_mutation).

The fix clears the inherited marker in ``_prepare_agent_startup`` — the
chokepoint all three launch paths (fast chat, Termux fast-CLI, full dispatch)
call before any agent turn.  Plain CLI subcommands (``hermes kanban ...``,
``boards``, ``cron list``, ...) never reach that gate, so a *genuine* delegate
child shelling out to the CLI keeps its lineage and the DB-layer wall holds
(covered by test_kanban_cli_exit_status.py).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).parents[2]

MARKER = "HERMES_DELEGATED_CHILD_CONTEXT"


def _startup_args(command=None, **extra):
    """An argparse.Namespace shaped like what the agent-session gate reads."""
    ns = argparse.Namespace(command=command)
    for key, value in extra.items():
        setattr(ns, key, value)
    return ns


# ---------------------------------------------------------------------------
# Unit: the agent-session chokepoint clears the marker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(command="chat"),
        dict(command=None),               # bare `hermes` interactive
        dict(command="acp"),
        dict(command="rl"),
        dict(command="gateway", gateway_command="run"),
        dict(command="cron", cron_command="run"),
        dict(command="mcp", mcp_action="serve"),
    ],
    ids=[
        "chat", "bare", "acp", "rl", "gateway-run", "cron-run", "mcp-serve",
    ],
)
def test_agent_session_gate_clears_inherited_marker(kwargs, monkeypatch):
    from hermes_cli.main import _prepare_agent_startup

    monkeypatch.setenv(MARKER, "1")
    with mock.patch("hermes_cli.main._apply_safe_mode"), mock.patch(
        "hermes_cli.main._is_tui_chat_launch", return_value=True
    ):
        _prepare_agent_startup(_startup_args(**kwargs))
    assert MARKER not in os.environ


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(command="kanban"),
        dict(command="boards"),
        dict(command="version"),
        dict(command="cron", cron_command="list"),
        dict(command="gateway", gateway_command="status"),
    ],
    ids=["kanban", "boards", "version", "cron-list", "gateway-status"],
)
def test_plain_cli_subcommands_keep_marker(kwargs, monkeypatch):
    """A genuine delegate child shelling out to the CLI must keep its env
    lineage — that is what kanban_db's mutation wall reads."""
    from hermes_cli.main import _prepare_agent_startup

    monkeypatch.setenv(MARKER, "1")
    with mock.patch("hermes_cli.main._apply_safe_mode"):
        _prepare_agent_startup(_startup_args(**kwargs))
    assert os.environ.get(MARKER) == "1"


def test_noop_when_marker_absent(monkeypatch):
    from hermes_cli.main import _prepare_agent_startup

    monkeypatch.delenv(MARKER, raising=False)
    with mock.patch("hermes_cli.main._apply_safe_mode"), mock.patch(
        "hermes_cli.main._is_tui_chat_launch", return_value=True
    ):
        _prepare_agent_startup(_startup_args(command="chat"))
    assert MARKER not in os.environ


@pytest.mark.parametrize(
    "command", ["serve", "dashboard", "desktop", "gui"]
)
def test_agent_host_servers_clear_marker_without_discovery(monkeypatch, command):
    """serve/dashboard/desktop/gui host agent turns and spawn kernels, so the
    inherited marker must go — but they run their own plugin/MCP discovery
    later, so _prepare_agent_startup must not run that discovery for them."""
    from hermes_cli.main import _prepare_agent_startup

    monkeypatch.setenv(MARKER, "1")
    discovery_ran = {"plugins": False, "tui_probe": False}

    def _fail_discovery():
        discovery_ran["plugins"] = True

    with mock.patch(
        "hermes_cli.main._apply_safe_mode"
    ), mock.patch(
        "hermes_cli.main.start_background_plugin_discovery",
        _fail_discovery,
        create=True,
    ):
        import hermes_cli.main as main_mod

        with mock.patch.object(
            main_mod, "_is_tui_chat_launch",
            side_effect=lambda a: discovery_ran.__setitem__("tui_probe", True),
        ):
            _prepare_agent_startup(_startup_args(command=command))
    assert MARKER not in os.environ, f"{command} kept the inherited marker"
    assert not discovery_ran["plugins"], (
        f"{command} must not trigger _prepare_agent_startup's plugin discovery"
    )
    assert not discovery_ran["tui_probe"], (
        f"{command} must return before the agent-session startup body"
    )


# ---------------------------------------------------------------------------
# End-to-end: real main() dispatch, both launch paths, child process.
# cmd_chat is stubbed INSIDE the child, so no agent turn and no gateway
# subprocess ever starts; we observe the env after main() dispatches.
# ---------------------------------------------------------------------------


def _child_main_probe(fast_launch: bool) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env[MARKER] = "1"
    if fast_launch:
        env.pop("HERMES_DISABLE_FAST_CHAT_LAUNCH", None)
    else:
        env["HERMES_DISABLE_FAST_CHAT_LAUNCH"] = "1"
    env["HERMES_GATEWAY_AUTOSTART"] = "0"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import os, sys;"
        "import hermes_cli.main as m;"
        "m.cmd_chat = lambda a: print('MARKER_AFTER_MAIN=' + str("
        "os.environ.get('HERMES_DELEGATED_CHILD_CONTEXT')));"
        "sys.argv = ['hermes', 'chat', '-q', 'hi'];"
        "m.main()"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.mark.parametrize("fast_launch", [True, False], ids=["fast", "full"])
def test_main_chat_launch_ends_with_clean_marker(fast_launch):
    proc = _child_main_probe(fast_launch)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "MARKER_AFTER_MAIN=None" in proc.stdout, (
        f"inherited marker survived a top-level chat launch: "
        f"{proc.stdout[-500:]}"
    )
