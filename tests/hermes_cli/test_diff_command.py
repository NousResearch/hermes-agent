"""Tests for the CLI ``/diff`` command handler.

``/diff`` shows the cumulative diff of everything Hermes changed in the
working directory (earliest retained checkpoint to working tree), the
session-wide counterpart to ``/rollback diff <N>``. These assert the handler
renders the manager's ``session_diff`` result, honours ``--stat``, and
degrades gracefully when checkpoints are off / empty / no agent.
"""

import contextlib
import io

from hermes_cli.cli_commands_mixin import CLICommandsMixin


class _Mgr:
    def __init__(self, result, enabled=True):
        self.enabled = enabled
        self._result = result
        self.calls = []

    def session_diff(self, cwd):
        self.calls.append(cwd)
        return self._result


class _Agent:
    def __init__(self, mgr):
        self._checkpoint_mgr = mgr


class _Stub(CLICommandsMixin):
    def __init__(self, agent=None):
        self.agent = agent


def _run(stub, command):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        stub._handle_diff_command(command)
    return buf.getvalue()


def test_diff_prints_stat_and_diff():
    mgr = _Mgr({
        "success": True,
        "stat": " main.py | 2 +-",
        "diff": "--- a/main.py\n+++ b/main.py\n-print('hello')\n+print('v3')\n",
    })
    out = _run(_Stub(_Agent(mgr)), "/diff")
    assert " main.py | 2 +-" in out
    assert "+print('v3')" in out
    assert mgr.calls  # session_diff was consulted


def test_diff_stat_only_suppresses_body():
    mgr = _Mgr({
        "success": True,
        "stat": " main.py | 2 +-",
        "diff": "+print('v3')\n",
    })
    out = _run(_Stub(_Agent(mgr)), "/diff --stat")
    assert " main.py | 2 +-" in out
    assert "+print('v3')" not in out


def test_diff_empty_reports_no_changes():
    mgr = _Mgr({"success": True, "stat": "", "diff": "", "empty": True})
    out = _run(_Stub(_Agent(mgr)), "/diff")
    assert "No changes" in out


def test_diff_disabled_explains_how_to_enable():
    mgr = _Mgr({"success": True, "stat": "", "diff": ""}, enabled=False)
    out = _run(_Stub(_Agent(mgr)), "/diff")
    assert "not enabled" in out.lower()
    assert not mgr.calls  # short-circuits before touching the store


def test_diff_without_agent_is_graceful():
    out = _run(_Stub(agent=None), "/diff")
    assert "No active agent session" in out


def test_diff_failure_surfaces_error():
    mgr = _Mgr({"success": False, "error": "boom"})
    out = _run(_Stub(_Agent(mgr)), "/diff")
    assert "boom" in out
