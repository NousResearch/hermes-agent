"""``scripts/run_tests.sh`` pre-compiles bytecode only for tracked files.

The pre-compile step hands ``git ls-files '*.py'`` to ``compileall``. Outside a
git checkout (a ``cp -r`` copy, an unpacked archive) that list is empty, and
``compileall`` given no path compiles every ``sys.path`` entry instead: the
interpreter's own stdlib included, which can be the managed Python the installed
Hermes runs on. Its output and status are discarded, so the write was silent.

The interpreter is a fake that records its arguments and exits 0: no real
``compileall`` ever runs here, and the runner's last call (the parallel runner)
is recorded rather than executed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKIP_WARNING = "skipping bytecode pre-compile"


def _env(tmp_path: Path) -> dict[str, str]:
    """The caller's environment minus anything that selects a Hermes runtime or
    a git repository: an inherited GIT_DIR or GIT_INDEX_FILE would point both the
    fixture's ``git add`` and the script's ``git ls-files`` at another repo, and
    a git work tree above tmp_path would make the copy look tracked."""
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("HERMES_", "__HERMES_", "GIT_"))
    }
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
    return env


def _run(tmp_path: Path, *, git: bool) -> tuple[subprocess.CompletedProcess, list[str]]:
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    for name in ("run_tests.sh", "_activation.sh"):
        shutil.copy2(REPO_ROOT / "scripts" / name, root / "scripts" / name)
    (root / "pkg").mkdir()
    (root / "pkg" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    env = _env(tmp_path)
    if git:
        subprocess.run(["git", "init", "-q", str(root)], env=env, check=True)
        subprocess.run(["git", "-C", str(root), "add", "--", "pkg/mod.py"], env=env, check=True)

    log = tmp_path / "calls.log"
    fake = tmp_path / "bin" / "python"
    fake.parent.mkdir()
    fake.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{log}'\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)

    env.update({
        "HERMES_PYTHON": str(fake),
        # A named store skips the faked-HOME probe of the account's tool store.
        "HERMES_RUNTIME_DIR": str(tmp_path / "store"),
    })
    result = subprocess.run(
        ["bash", str(root / "scripts" / "run_tests.sh"), "-j", "1"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60, check=False,
    )
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result, calls


def _compileall_calls(calls: list[str]) -> list[str]:
    return [call for call in calls if call.startswith("-m compileall")]


@pytest.mark.platforms("posix")
def test_outside_git_skips_precompile_loudly_and_still_runs_the_suite(tmp_path):
    result, calls = _run(tmp_path, git=False)
    output = result.stdout + result.stderr

    assert _compileall_calls(calls) == [], output
    assert SKIP_WARNING in result.stderr, output
    # The reason is git's own message, not a guess.
    assert "not a git repository" in result.stderr, output
    assert result.returncode == 0, output
    assert any("run_tests_parallel.py" in call for call in calls), calls


@pytest.mark.platforms("posix")
def test_inside_git_precompiles_exactly_the_tracked_files(tmp_path):
    result, calls = _run(tmp_path, git=True)
    output = result.stdout + result.stderr

    assert _compileall_calls(calls) == ["-m compileall -q -j 0 -- pkg/mod.py"], output
    assert SKIP_WARNING not in output, output
    assert result.returncode == 0, output
