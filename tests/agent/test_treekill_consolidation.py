"""#85125 Phase 4d — site-local tree-kills delegate to agent.deadline.

Each migrated wrapper keeps its caller-facing contract (signature, all
failures swallowed, ``None`` return) while routing the actual tree
termination through :func:`agent.deadline.kill_process_tree`:

* ``hermes_cli._subprocess_compat.kill_process_tree(proc)`` — also consumed
  by ``agent.shell_hooks`` by name; falls back to
  ``_legacy_kill_process_tree`` when delegation fails.
* ``tools.browser_tool_lifecycle._kill_process_tree(proc)`` — same pattern.
* ``tools.code_execution_tool._kill_process_group(proc, escalate=...)`` —
  SIGTERM tree first, then (escalate) bounded wait + SIGKILL tree.

Plus one real end-to-end probe: a setsid'd grandchild must die through the
compat wrapper (i.e. the psutil descendant sweep in agent.deadline is
actually reached from the delegating call site).
"""

from __future__ import annotations

import subprocess
import sys
import time
from unittest.mock import MagicMock

import pytest

import agent.deadline as deadline_mod
from tools import browser_tool_lifecycle as bt_lifecycle


class _FakeProc:
    def __init__(self, pid=54321):
        self.pid = pid
        self.kill_calls = 0

    def kill(self):
        self.kill_calls += 1


# ---------------------------------------------------------------------------
# (1) hermes_cli._subprocess_compat.kill_process_tree
# ---------------------------------------------------------------------------

class TestSubprocessCompatDelegation:
    def test_delegates_with_proc_pid(self, monkeypatch):
        from hermes_cli import _subprocess_compat

        calls = []
        monkeypatch.setattr(
            deadline_mod, "kill_process_tree", lambda pid, **kw: calls.append(pid) or True
        )
        proc = _FakeProc(pid=1111)
        assert _subprocess_compat.kill_process_tree(proc) is None
        assert calls == [1111]

    def test_swallows_delegation_raise_and_falls_back_to_legacy(self, monkeypatch):
        from hermes_cli import _subprocess_compat

        def _boom(pid, **kw):
            raise RuntimeError("delegation broken")

        monkeypatch.setattr(deadline_mod, "kill_process_tree", _boom)
        legacy_calls = []
        monkeypatch.setattr(
            _subprocess_compat, "_legacy_kill_process_tree", lambda proc: legacy_calls.append(proc)
        )
        proc = _FakeProc(pid=2222)
        _subprocess_compat.kill_process_tree(proc)  # must not raise
        assert legacy_calls == [proc]

    def test_shell_hooks_consumer_still_imports_by_name(self):
        import agent.shell_hooks as shell_hooks
        from hermes_cli import _subprocess_compat

        assert shell_hooks.kill_process_tree is _subprocess_compat.kill_process_tree


# ---------------------------------------------------------------------------
# (2) tools.browser_tool_lifecycle._kill_process_tree
# ---------------------------------------------------------------------------

class TestBrowserToolDelegation:
    def test_delegates_with_proc_pid(self, monkeypatch):
        from tools import browser_tool

        calls = []
        monkeypatch.setattr(
            deadline_mod, "kill_process_tree", lambda pid, **kw: calls.append(pid) or True
        )
        proc = _FakeProc(pid=3333)
        assert bt_lifecycle._kill_process_tree(proc) is None
        assert calls == [3333]

    def test_swallows_delegation_raise_and_falls_back_to_legacy(self, monkeypatch):
        from tools import browser_tool

        def _boom(pid, **kw):
            raise OSError("delegation broken")

        monkeypatch.setattr(deadline_mod, "kill_process_tree", _boom)
        legacy_calls = []
        monkeypatch.setattr(
            "tools.browser_tool_lifecycle._legacy_kill_process_tree", lambda proc: legacy_calls.append(proc)
        )
        proc = _FakeProc(pid=4444)
        bt_lifecycle._kill_process_tree(proc)  # must not raise
        assert legacy_calls == [proc]


# ---------------------------------------------------------------------------
# (3) tools.code_execution_tool._kill_process_group
# ---------------------------------------------------------------------------

class TestCodeExecutionDelegation:
    def test_delegates_sigterm_tree_first(self, monkeypatch):
        import signal as _signal

        from tools import code_execution_tool

        calls = []
        monkeypatch.setattr(
            deadline_mod,
            "kill_process_tree",
            lambda pid, sig=None: calls.append((pid, sig)) or True,
        )
        proc = _FakeProc(pid=5555)
        code_execution_tool._kill_process_group(proc)
        assert calls == [(5555, _signal.SIGTERM)]

    def test_escalate_waits_then_sigkills_tree(self, monkeypatch):
        import signal as _signal

        from tools import code_execution_tool

        calls = []
        monkeypatch.setattr(
            deadline_mod,
            "kill_process_tree",
            lambda pid, sig=None: calls.append((pid, sig)) or True,
        )
        proc = MagicMock()
        proc.pid = 6666
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=5)
        code_execution_tool._kill_process_group(proc, escalate=True)
        assert calls == [(6666, _signal.SIGTERM), (6666, _signal.SIGKILL)]
        proc.wait.assert_called_once_with(timeout=5)

    def test_swallows_delegation_raise_falls_back_to_plain_kill(self, monkeypatch):
        from tools import code_execution_tool

        def _boom(pid, **kw):
            raise PermissionError("nope")

        monkeypatch.setattr(deadline_mod, "kill_process_tree", _boom)
        proc = _FakeProc(pid=7777)
        code_execution_tool._kill_process_group(proc)  # must not raise
        assert proc.kill_calls == 1

    def test_even_proc_kill_raise_is_swallowed(self, monkeypatch):
        from tools import code_execution_tool

        def _boom(pid, **kw):
            raise PermissionError("nope")

        monkeypatch.setattr(deadline_mod, "kill_process_tree", _boom)
        proc = MagicMock()
        proc.pid = 8888
        proc.kill.side_effect = OSError("already reaped")
        code_execution_tool._kill_process_group(proc)  # must not raise


# ---------------------------------------------------------------------------
# End-to-end: setsid grandchild dies through the compat wrapper
# ---------------------------------------------------------------------------

@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX session semantics")
def test_e2e_setsid_grandchild_killed_via_compat_wrapper(tmp_path):
    """The delegating wrapper must reach descendants that setsid'd out of the
    child's process group — the exact capability the shared primitive adds
    over the old local killpg-only body (scaffold shape lifted from
    tests/agent/test_deadline.py::test_kills_descendant_in_its_own_session).
    """
    pytest.importorskip("psutil")
    import psutil

    from hermes_cli._subprocess_compat import kill_process_tree

    started = tmp_path / "grandchild_started"
    marker = tmp_path / "grandchild_survived"
    grandchild_py = tmp_path / "grandchild.py"
    grandchild_py.write_text(
        "import pathlib, time\n"
        f"pathlib.Path({str(started)!r}).write_text('x')\n"
        "time.sleep(10)\n"
        f"pathlib.Path({str(marker)!r}).write_text('x')\n"
    )
    parent_py = tmp_path / "parent.py"
    parent_py.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, {str(grandchild_py)!r}], start_new_session=True)\n"
        "time.sleep(10)\n"
    )
    proc = subprocess.Popen([sys.executable, str(parent_py)], start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert started.exists(), "grandchild never spawned — test harness broken"
        # Snapshot the tree BEFORE the kill so we can assert zero survivors.
        descendants = psutil.Process(proc.pid).children(recursive=True)

        kill_process_tree(proc)

        proc.wait(timeout=5)
        gone, alive = psutil.wait_procs(descendants, timeout=5)
        assert not alive, f"survivors after tree kill: {alive}"
        time.sleep(1.0)
        assert not marker.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
