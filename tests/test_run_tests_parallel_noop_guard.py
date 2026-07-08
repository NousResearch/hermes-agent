"""E2E: the silent-no-op guard in scripts/run_tests_parallel.py fails LOUD.

Founding ask (`20260703_144008`): *"so the e2e suite can't silently no-op
again."* Before this guard the runner coerced pytest exit-5 ("no tests
collected") to exit-0 per file, so a whole zero-collect suite reported green —
"green because it didn't run." These tests prove the guard flips a zero-collect
suite from GREEN to RED, keeps a legitimate marker-filtered 0 GREEN, and is not
vacuous (mutation proof: turn the guard off and the same invocation goes green).

Worktree-bytes discipline (build brief §8f / PEP660 editable install): the
runner is a *script* (scripts/run_tests_parallel.py), so every subprocess
invocation below execs the worktree copy by absolute path — worktree bytes by
construction. `test_module_under_test_is_worktree_bytes` additionally imports
the module by file path and ASSERTS the worktree marker is in `module.__file__`
before we trust any green here.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNNER = _REPO_ROOT / "scripts" / "run_tests_parallel.py"
# Stable across any worktree name AND the merged main tree: assert the
# module resolves under THIS test file's repo, not a separate editable
# install copy elsewhere on disk.
import pathlib as _pl
_REPO_ROOT = str(_pl.Path(__file__).resolve().parents[1])


def _load_runner_module():
    """Import the runner by absolute file path (worktree bytes, no finder)."""
    spec = importlib.util.spec_from_file_location(
        "_a5_run_tests_parallel_under_test", _RUNNER
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_under_test_is_worktree_bytes() -> None:
    """Prove we are exercising WORKTREE bytes, not a deployed/installed copy.

    The build brief mandates asserting the worktree path is in the module's
    __file__ BEFORE trusting any green. The runner is also invoked as a
    subprocess in the other tests via _RUNNER, whose path we assert here.
    """
    assert _REPO_ROOT in str(_RUNNER), (
        f"runner path is not in the A5 worktree: {_RUNNER}"
    )
    mod = _load_runner_module()
    assert mod.__file__ and _REPO_ROOT in mod.__file__, (
        f"imported runner is not worktree bytes: {mod.__file__}"
    )
    # The guard itself must exist on these bytes (else every green below is a
    # test of the wrong file).
    assert hasattr(mod, "_noop_guard"), "no-op guard missing from worktree bytes"


def _run_runner(*args: str, timeout: int = 90) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_RUNNER), "-j", "1", "--file-timeout", "60", *args],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def _write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


# ── Probe fixtures ───────────────────────────────────────────────────────────

def _all_skip_file(d: Path) -> Path:
    """A file whose every test skips (simulates a missing-dep importorskip)."""
    return _write(
        d / "test_all_skip.py",
        "import pytest\n\n"
        "@pytest.mark.skip(reason='dep missing')\n"
        "def test_one():\n    assert True\n\n"
        "@pytest.mark.skip(reason='dep missing')\n"
        "def test_two():\n    assert True\n",
    )


def _all_integration_file(d: Path) -> Path:
    """A file whose every test is marked integration (marker-filtered lane)."""
    return _write(
        d / "test_all_integration.py",
        "import pytest\n\n"
        "@pytest.mark.integration\n"
        "def test_i_one():\n    assert True\n\n"
        "@pytest.mark.integration\n"
        "def test_i_two():\n    assert True\n",
    )


def _real_pass_file(d: Path) -> Path:
    """A file with real, un-filtered passing tests."""
    return _write(
        d / "test_real.py",
        "def test_a():\n    assert True\n\n"
        "def test_b():\n    assert True\n\n"
        "def test_c():\n    assert True\n",
    )


# ── A5-A: the founding case + mutation proof ───────────────────────────────────

def test_explicit_zero_collect_file_is_RED(tmp_path: Path) -> None:
    """Explicitly-requested file whose every test skips → RED + loud banner.

    This is the exact silent no-op: pre-guard the file exit-5→0-coerced to a
    pass and the run went green. Now it must exit non-zero.
    """
    probe = _all_skip_file(tmp_path)
    proc = _run_runner(str(probe))
    assert proc.returncode != 0, (
        f"zero-collect explicit file went GREEN — guard did not fire:\n{proc.stdout}"
    )
    assert "collected 0 tests (no-op)" in proc.stdout, proc.stdout


def test_mutation_proof_guard_off_goes_green(tmp_path: Path) -> None:
    """MUTATION PROOF (test-gate-honesty §2): flip the strict-noop gate off,
    same invocation goes GREEN — proving the gate is the load-bearing RED and
    not vacuous. --no-strict-noop is the one-line flip.

    We pair the zero-collect file with a real-pass file and request BOTH
    explicitly, so total_executed > 0 (the whole-suite-zero gate does NOT fire)
    and the explicit-0-collect gate is the ONLY thing deciding RED vs GREEN.
    That isolates the gate under proof.
    """
    skip = _all_skip_file(tmp_path)
    real = _real_pass_file(tmp_path)

    red = _run_runner(str(skip), str(real))
    assert red.returncode != 0, f"guard-on should be RED:\n{red.stdout}"
    assert "collected 0 tests (no-op)" in red.stdout, red.stdout

    green = _run_runner(str(skip), str(real), "--no-strict-noop")
    assert green.returncode == 0, (
        "guard-off (--no-strict-noop) should go GREEN, proving the strict-noop "
        f"gate is load-bearing:\n{green.stdout}"
    )
    # And the no-op is still SURFACED even when not gated (visibility ≠ gating).
    assert "no-op" in green.stdout, green.stdout


def test_whole_suite_zero_is_RED(tmp_path: Path) -> None:
    """A whole discovery run that executes 0 tests → hard RED, always.

    Uses a directory of only marker-filtered files with -m 'not integration':
    every file collects 0, total executed == 0. This trips the whole-suite-zero
    gate regardless of strict-noop.
    """
    _all_integration_file(tmp_path)
    proc = _run_runner(str(tmp_path), "-m", "not integration", "--no-strict-noop")
    assert proc.returncode != 0, (
        f"whole-suite-zero went GREEN:\n{proc.stdout}"
    )
    assert "NO TESTS EXECUTED" in proc.stdout, proc.stdout


def test_legit_marker_filter_stays_GREEN(tmp_path: Path) -> None:
    """FALSE-POSITIVE GUARD (the load-bearing test): a default-discovery run
    where SOME files are marker-emptied but others run must stay GREEN.

    This is CI's normal `-m 'not integration'` lane. The integration file is a
    discovered (not explicitly-requested) 0-collect → ⚠-surfaced, never RED.
    The real file runs, so total executed > 0 and the whole-suite gate is not
    tripped. If this reds, the guard wedges the normal unit lane.
    """
    _all_integration_file(tmp_path)
    _real_pass_file(tmp_path)
    proc = _run_runner(str(tmp_path), "-m", "not integration")
    assert proc.returncode == 0, (
        f"legit marker-filtered discovery run went RED — guard false-positives "
        f"on the normal unit lane:\n{proc.stdout}"
    )
    # The discovered marker-emptied file is NOT flagged as an explicit no-op
    # (it wasn't explicitly requested) and does NOT trip whole-suite-zero
    # (the real file executed). It stays green — the whole point.
    assert "NO TESTS EXECUTED" not in proc.stdout, proc.stdout


def test_min_tests_floor(tmp_path: Path) -> None:
    """--min-tests N: RED when executed < N, GREEN when executed >= N."""
    _real_pass_file(tmp_path)  # 3 real tests
    red = _run_runner(str(tmp_path), "--min-tests", "4")
    assert red.returncode != 0, f"3 < 4 should be RED:\n{red.stdout}"
    assert "below --min-tests floor" in red.stdout, red.stdout

    green = _run_runner(str(tmp_path), "--min-tests", "3")
    assert green.returncode == 0, f"3 >= 3 should be GREEN:\n{green.stdout}"


def test_normal_passing_run_stays_green(tmp_path: Path) -> None:
    """Sanity: a normal explicit file with real tests is GREEN and NOT flagged."""
    probe = _real_pass_file(tmp_path)
    proc = _run_runner(str(probe))
    assert proc.returncode == 0, proc.stdout
    assert "no-op" not in proc.stdout, proc.stdout
    assert "NO TESTS EXECUTED" not in proc.stdout, proc.stdout
