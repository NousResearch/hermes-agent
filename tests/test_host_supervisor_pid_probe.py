"""Regression tests for the compute-host supervisor's pid handling.

``os.kill(pid, 0)`` is not a no-op on Windows.  ``signal.CTRL_C_EVENT`` is 0,
so that call reaches ``GenerateConsoleCtrlEvent``, whose second argument is a
process *group*.  A child started without ``CREATE_NEW_PROCESS_GROUP`` shares
its parent's group, so the Ctrl-C lands on everything attached to the same
console.  Measured on Windows 11, Python 3.12.10 — a parent that spawns a
sleeping child and then probes it with ``os.kill(child_pid, 0)``::

    alive before   : True
    os.kill(pid,0) : returned, no exception
    Traceback ... <string>, line 1        <- the child died
    Traceback ... probe.py, line 10       <- the parent died too
    KeyboardInterrupt

``KeyboardInterrupt`` is a ``BaseException``, so it also sails through the
``except Exception`` arms the probe is wrapped in.

``signal.SIGKILL`` does not exist on Windows at all; referencing it raises
``AttributeError``, which the surrounding blanket except swallows at debug
level and leaves the process running.

Both are flagged by ``scripts/check-windows-footguns.py``'s own rules.  Neither
was ever reported, because ``--all`` did not scan ``tui_gateway/``.
"""

from __future__ import annotations

import ast
import os

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HOST_SUPERVISOR = REPO_ROOT / "tui_gateway" / "host_supervisor.py"
FOOTGUN_SCRIPT = REPO_ROOT / "scripts" / "check-windows-footguns.py"


def _source() -> str:
    return HOST_SUPERVISOR.read_text(encoding="utf-8")


def test_liveness_probe_does_not_signal_the_process() -> None:
    """No executable ``os.kill(pid, 0)`` — signal 0 is CTRL_C_EVENT on Windows.

    Parsed rather than grepped: the explanation of why this is wrong lives in
    a docstring in the same file and would match a textual search.
    """
    tree = ast.parse(_source())
    offenders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "kill"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == 0
    ]
    assert not offenders, [n.lineno for n in offenders]


def test_hard_kill_signal_is_resolved_defensively() -> None:
    """``signal.SIGKILL`` is absent on Windows; it must not be referenced bare."""
    source = _source()
    assert "signal.SIGKILL" not in source or 'getattr(signal, "SIGKILL"' in source
    assert 'getattr(signal, "SIGKILL", signal.SIGTERM)' in source


def test_liveness_probe_uses_the_shared_cross_platform_helper() -> None:
    assert "from gateway.status import _pid_exists" in _source()


def test_pid_alive_answers_correctly_for_a_live_and_a_dead_pid() -> None:
    """Behavioural check, no mocks — and it does not signal anything.

    On the pre-fix source this test still passes on POSIX (``os.kill(pid, 0)``
    really is a no-op there); it is the Windows run that separates them.
    """
    from tui_gateway.host_supervisor import _pid_alive

    assert _pid_alive(os.getpid()) is True
    assert _pid_alive(0) is False
    assert _pid_alive(-1) is False

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    # A reaped child's pid is gone on POSIX; on Windows the handle is closed
    # by Popen.__del__/wait, so this is the "definitely not running" case.
    assert _pid_alive(child.pid) in (False, True)  # value is OS-dependent
    # What must hold everywhere: probing did not raise and did not kill us.
    assert _pid_alive(os.getpid()) is True


def test_footgun_scan_covers_tui_gateway_and_top_level_modules() -> None:
    """``--all`` has to reach the code these rules are about.

    Before this change the root list held eight package directories; neither
    ``tui_gateway/`` nor the modules at the repository root were among them,
    so the CI gate reported success across a strict subset of the tree.
    """
    script = FOOTGUN_SCRIPT.read_text(encoding="utf-8")
    assert 'REPO_ROOT / "tui_gateway"' in script
    assert 'REPO_ROOT.glob("*.py")' in script


def test_footgun_scan_is_clean_on_the_widened_scope() -> None:
    """The widened gate stays green — that is what makes it mergeable."""
    result = subprocess.run(
        [sys.executable, str(FOOTGUN_SCRIPT), "--all"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
        timeout=300,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
