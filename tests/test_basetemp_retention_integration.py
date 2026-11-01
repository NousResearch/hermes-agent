"""End-to-end: a real sequence of test runs keeps a bounded number of sandboxes.

The unit tests in ``test_basetemp_custody.py`` pin the admission functions. They cannot
show that pytest *behaves* differently, because retention is not something this suite
configures - it is something pytest's tmpdir plugin does. So each test here spawns real
``python -m pytest`` runs in an isolated temp root and counts the directories that
survive, with the real ``tests/conftest.py`` loaded via ``-p`` (an outside-checkout probe
file does not load it by path).

Three scopes are proven separately, because they are three different mechanisms and only
one of them is retention:

1. default numbered basetemp  -> retention applies: keep 3 sessions, drop passing ones.
2. caller-named ``--basetemp`` -> retention never applies; that is why it must be refused
   rather than trusted, and why unbounded accumulation happened through this path.
3. relocated basetemp          -> runner-owned scratch, deleted unconditionally.

Everything runs in a throwaway root with ``HOME``/``HERMES_HOME`` pointed inside it, so no
run can reach the live install, and the disk reserve guard stays active.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_FAILING = "def test_probe(tmp_path):\n    (tmp_path / 'x').write_text('x')\n    assert 0\n"
_PASSING = "def test_probe(tmp_path):\n    (tmp_path / 'x').write_text('x')\n    assert 1\n"


def _run(sandbox: Path, probe: Path, *extra: str) -> subprocess.CompletedProcess:
    """One real pytest session against *probe*, isolated from the operator's installation."""
    root = sandbox / "temproot"
    home = sandbox / "home"
    root.mkdir(exist_ok=True)
    home.mkdir(exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(REPO_ROOT),
        "PYTEST_DEBUG_TEMPROOT": str(root),
        # Isolation: nothing resolves to the live install or the operator's real home.
        "HOME": str(home),
        "HERMES_HOME": str(home),
        # Guard 2 stays armed for these runs; it is satisfiable here, so they are admitted.
        "HERMES_TEST_DISK_RESERVE_MB": "2000",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
        "LANG": "C.UTF-8",
    }
    return subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "-p", "tests.conftest",
            "-p", "no:cacheprovider",
            str(probe), "-q", "--tb=no", *extra,
        ],
        cwd=str(sandbox), env=env, capture_output=True, text=True, timeout=300,
        check=False,
    )


def _basetemps(sandbox: Path) -> list[Path]:
    root = sandbox / "temproot"
    found = [p for p in root.glob("pytest-of-*/pytest-*") if p.is_dir() and p.name != "pytest-current"]
    return sorted(found, key=lambda p: int(p.name.rsplit("-", 1)[1]))


@pytest.fixture()
def sandbox(tmp_path):
    """A probe file per test, in a root outside the checkout."""
    box = tmp_path / "run"
    box.mkdir()
    (box / "probe_fail.py").write_text(_FAILING)
    (box / "probe_pass.py").write_text(_PASSING)
    return box


def test_a_failing_session_retains_its_sandbox(sandbox):
    result = _run(sandbox, sandbox / "probe_fail.py")
    assert result.returncode == 1, result.stdout + result.stderr
    kept = _basetemps(sandbox)
    assert len(kept) == 1, [p.name for p in kept]
    assert (kept[0] / "test_probe0" / "x").is_file(), "failed sandbox was not preserved"


def test_a_passing_session_leaves_no_basetemp_behind(sandbox):
    assert _run(sandbox, sandbox / "probe_fail.py").returncode == 1
    assert len(_basetemps(sandbox)) == 1

    result = _run(sandbox, sandbox / "probe_pass.py")
    assert result.returncode == 0, result.stdout + result.stderr
    kept = _basetemps(sandbox)
    # The passing session allocated pytest-1, then removed it at sessionfinish: only the
    # failing run's basetemp is left. This is the "keeps failures, prunes successes" rule
    # actually happening, not a config value being set.
    assert len(kept) == 1, [p.name for p in kept]
    assert (kept[0] / "test_probe0").is_dir()


def test_successive_failing_runs_keep_exactly_three(sandbox):
    for _ in range(5):
        assert _run(sandbox, sandbox / "probe_fail.py").returncode == 1

    kept = _basetemps(sandbox)
    assert len(kept) == 3, [p.name for p in kept]
    # Bounded history keeps the NEWEST sessions and prunes the surplus on the way in.
    assert [p.name for p in kept] == ["pytest-2", "pytest-3", "pytest-4"]
    for path in kept:
        assert (path / "test_probe0" / "x").is_file(), f"{path.name} lost its sandbox"


def test_a_caller_named_basetemp_is_outside_retention(sandbox):
    """The mechanism that produced 76.2 GB: pytest never prunes a basetemp it was handed.

    Distinct names accumulate without limit, so bounded history does not exist on this path
    - which is exactly why the entry point rejects the flag instead of trusting retention.
    """
    first = sandbox / "given-one"
    second = sandbox / "given-two"
    for target in (first, second):
        assert _run(sandbox, sandbox / "probe_fail.py", "--basetemp", str(target)).returncode == 1

    assert first.is_dir() and (first / "test_probe0").is_dir()
    assert second.is_dir() and (second / "test_probe0").is_dir()
    # Not numbered, not counted, not pruned: two runs, two survivors, forever.
    assert _basetemps(sandbox) == []

    # Reusing one name does not accumulate either - getbasetemp() wipes and recreates it, so
    # the third run leaves one sandbox, not a growing series of them.
    assert _run(sandbox, sandbox / "probe_fail.py", "--basetemp", str(first)).returncode == 1
    assert (first / "test_probe0" / "x").is_file()
    sandboxes = sorted(p.name for p in first.iterdir()
                       if p.is_dir() and not p.is_symlink() and p.name.startswith("test_probe"))
    assert sandboxes == ["test_probe0"], sandboxes


def test_the_runner_rejects_the_flag_that_escapes_cleanup(sandbox):
    """The supported entry point refuses the unbounded path at the argument parser."""
    runner = REPO_ROOT / "scripts" / "run_tests_parallel.py"
    result = subprocess.run(
        [sys.executable, str(runner), "tests/test_basetemp_custody.py",
         "--", "--basetemp", str(sandbox / "given-three")],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180, check=False,
    )
    assert result.returncode != 0
    assert "--basetemp escapes runner-owned cleanup" in (result.stderr + result.stdout)
    assert not (sandbox / "given-three").exists()
