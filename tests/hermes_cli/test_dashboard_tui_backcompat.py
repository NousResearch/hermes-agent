"""Regression test: `hermes dashboard --tui` must not hard-crash.

Older Hermes desktop app shells (<= 0.15.x) spawn the backend as::

    hermes dashboard --no-open --tui --host 127.0.0.1 --port <PORT>

The ``--tui`` flag was removed from the ``dashboard`` subcommand in cae6b5486
(embedded chat is always on now). When a user's CLI updates past that commit
but their desktop app binary has not, argparse used to reject the unknown flag
with ``error: unrecognized arguments: --tui`` and ``exit(2)`` — the backend
died before it became ready and the desktop GUI showed only "Hermes couldn't
start" with no actionable cause.

The fix adds a hidden, deprecated, accepted-and-ignored ``--tui`` flag to the
dashboard subparser so an old app shell + new CLI degrades gracefully instead
of bricking. These tests pin that contract.
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)


def _run_cli(args, timeout=60):
    """Invoke the real hermes_cli.main parser in a subprocess.

    Uses ``--status`` so the dashboard command exits immediately after parsing
    (it scans the process table and returns) instead of starting a server.
    Returns the CompletedProcess.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_dashboard_tui_flag_is_accepted_not_rejected():
    """The exact argv an old desktop app sends must parse without argparse error."""
    result = _run_cli(
        ["dashboard", "--no-open", "--tui", "--host", "127.0.0.1",
         "--port", "39997", "--status"]
    )
    combined = (result.stdout or "") + (result.stderr or "")
    # The pre-fix failure signature.
    assert "unrecognized arguments" not in combined, combined
    assert "--tui" not in (result.stderr or ""), result.stderr
    # argparse usage errors exit 2; the parse itself must not be that error.
    assert result.returncode != 2, combined


def test_dashboard_tui_flag_is_hidden_from_help():
    """The deprecated shim must not re-advertise a removed feature in --help."""
    result = _run_cli(["dashboard", "--help"])
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, combined
    assert "--tui" not in combined, (
        "dashboard --tui is a deprecated back-compat shim and must stay "
        "hidden via argparse.SUPPRESS:\n" + combined
    )


def test_dashboard_without_tui_still_parses():
    """Sanity: the modern (no --tui) invocation is unaffected by the shim."""
    result = _run_cli(
        ["dashboard", "--no-open", "--host", "127.0.0.1",
         "--port", "39996", "--status"]
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "unrecognized arguments" not in combined, combined
    assert result.returncode != 2, combined
