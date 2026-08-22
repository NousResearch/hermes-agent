"""``bounded_probe_run`` — deadlock-safe capture for fail-open probes (#87134).

On Windows, ``subprocess.run(..., capture_output=True, timeout=N)`` can hang
FOREVER after its timeout fires: run()'s cleanup kills the direct child and
then joins the pipe reader threads with an unbounded ``communicate()``.  A
descendant (``conhost.exe`` under wmic/powershell, ``git.exe`` under a
launcher shim) holding duplicated pipe handles keeps the pipes from EOF and
the join never returns.  ``hermes update`` wedged exactly there inside
``_scan_gateway_pids`` on machines where the full ``Win32_Process`` scan
exceeds its budget.

``bounded_probe_run`` is the shared, generalized form of the fix that
``bounded_git_probe`` already proved for git probes (#68609 / #66037):
explicit ``communicate(timeout)``, tree-kill on failure, bounded 1s drain,
then abandon.

These tests use REAL subprocesses (no mocks) for the semantics: a mock cannot
reproduce pipe-handle inheritance or timeout behavior.  The POSIX-only
descendant-survival cases live in ``test_git_probe_tree_kill.py`` and now
exercise the same code path through the ``bounded_git_probe`` delegation.
"""

import os
import subprocess
import sys
import time

import pytest

import hermes_cli._subprocess_compat as compat
from hermes_cli._subprocess_compat import bounded_git_probe, bounded_probe_run

_PY = sys.executable


def test_success_returns_completed_process():
    result = bounded_probe_run([_PY, "-c", "print('ok'); import sys; sys.exit(0)"], timeout=30)
    assert result is not None
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_nonzero_exit_is_returned_not_swallowed():
    """Unlike bounded_git_probe, callers see the real returncode — the gateway
    scan branches on ``returncode != 0`` to trip the wmic→powershell fallback."""
    result = bounded_probe_run([_PY, "-c", "print('partial'); import sys; sys.exit(3)"], timeout=30)
    assert result is not None
    assert result.returncode == 3
    assert result.stdout.strip() == "partial"


def test_spawn_failure_returns_none():
    result = bounded_probe_run(["definitely-not-a-real-binary-87134"], timeout=5)
    assert result is None


def test_timeout_returns_none_within_bounded_time():
    """A child that sleeps past the timeout must produce ``None`` promptly —
    timeout + tree-kill + 1s bounded drain, not an unbounded join."""
    start = time.monotonic()
    result = bounded_probe_run(
        [_PY, "-c", "import time; time.sleep(300)"],
        timeout=1.0,
    )
    elapsed = time.monotonic() - start
    assert result is None
    # 1s timeout + tree-kill + 1s drain + slack.  The pre-fix failure mode is
    # an indefinite hang, so any bound proves the property; keep it loose for
    # slow CI runners.
    assert elapsed < 30


def test_decode_errors_configurable():
    """The process scans pass errors='ignore' (wmic emits system code page);
    undecodable bytes must not raise or None out stdout (#17049 class)."""
    result = bounded_probe_run(
        [_PY, "-c", "import sys; sys.stdout.buffer.write(b'ok\\xff\\xfe')"],
        timeout=30,
        errors="ignore",
    )
    assert result is not None
    assert result.returncode == 0
    assert "ok" in result.stdout


def test_stdin_is_devnull_not_inherited():
    """A probe must never block reading the caller's stdin."""
    result = bounded_probe_run(
        [_PY, "-c", "import sys; print(repr(sys.stdin.read()))"],
        timeout=30,
    )
    assert result is not None
    assert result.returncode == 0
    assert result.stdout.strip() == "''"


def test_bounded_git_probe_delegates_same_contract():
    """The historical git-probe wrapper keeps its exact contract on top of
    bounded_probe_run: stripped stdout on rc==0, '' on any failure."""
    assert bounded_git_probe([_PY, "-c", "print('  x  ')"], timeout=30) == "x"
    assert bounded_git_probe([_PY, "-c", "import sys; sys.exit(1)"], timeout=30) == ""
    assert bounded_git_probe(["definitely-not-a-real-binary-87134"], timeout=5) == ""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group check")
def test_posix_child_gets_own_process_group():
    """POSIX spawns use process_group=0 so timeout cleanup can killpg the
    whole tree (same contract bounded_git_probe had)."""
    result = bounded_probe_run(
        [_PY, "-c", "import os; print(os.getpgid(0) == os.getpid())"],
        timeout=30,
    )
    assert result is not None
    assert result.stdout.strip() == "True"


def test_env_is_passed_through_to_the_child():
    """``env`` replaces the child's environment, exactly as ``run(env=...)`` did.

    The Node-toolchain probes in ``hermes_constants`` need this to put the
    Hermes-managed Node directory on ``PATH`` so a managed ``npm``/``npx`` shim
    can resolve its own ``node``. Without it those call sites could not adopt
    this helper and stayed on the unbounded ``run()`` that hung the ACP
    system-prompt build (#91087).
    """
    result = bounded_probe_run(
        [_PY, "-c", "import os; print(os.environ.get('HERMES_PROBE_MARKER', 'MISSING'))"],
        timeout=30,
        env={**os.environ, "HERMES_PROBE_MARKER": "present-91087"},
    )
    assert result is not None
    assert result.stdout.strip() == "present-91087"


def test_env_defaults_to_inheriting_the_parent_environment():
    """Omitting ``env`` must not change the historical spawn contract."""
    os.environ["HERMES_PROBE_INHERIT"] = "inherited-91087"
    try:
        result = bounded_probe_run(
            [_PY, "-c", "import os; print(os.environ.get('HERMES_PROBE_INHERIT', 'MISSING'))"],
            timeout=30,
        )
    finally:
        os.environ.pop("HERMES_PROBE_INHERIT", None)
    assert result is not None
    assert result.stdout.strip() == "inherited-91087"


def test_spawn_kwargs_are_platform_correct(monkeypatch):
    """The hidden-window flag on Windows / own process group on POSIX.

    This is the one assertion in this file that mocks the spawn, and
    deliberately so: unlike timeout and pipe-inheritance behaviour, a
    ``creationflags`` value has no observable runtime effect a real subprocess
    could report back, so there is nothing to assert against otherwise.

    It lives here rather than at the call sites because this is where the flag
    is chosen. ``hermes_constants``'s agent-browser probe used to assert
    ``creationflags`` on its own ``subprocess.run`` call; when those probes
    adopted this helper for #91087 the contract moved down here, and pinning it
    at the layer that sets it means all three probes are covered by one test
    instead of each re-asserting a spawn detail it no longer controls.
    """
    seen = {}

    class _FakeProc:
        returncode = 0

        def communicate(self, timeout=None):
            return ("", "")

    def fake_popen(argv, **kwargs):
        seen.update(kwargs)
        return _FakeProc()

    monkeypatch.setattr(compat.subprocess, "Popen", fake_popen)

    result = bounded_probe_run(["irrelevant"], timeout=5)
    assert result is not None

    if sys.platform == "win32":
        assert seen["creationflags"] == compat.windows_hide_flags()
        assert "process_group" not in seen
    else:
        assert seen["process_group"] == 0
        assert "creationflags" not in seen

    # The capture contract the probes depend on, on both platforms.
    assert seen["stdout"] is subprocess.PIPE
    assert seen["stderr"] is subprocess.PIPE
    assert seen["stdin"] is subprocess.DEVNULL
    assert seen["text"] is True
