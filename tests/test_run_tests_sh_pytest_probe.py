"""Regression tests for scripts/run_tests.sh virtualenv selection."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash runner probe")


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _write_fake_python(path: Path, label: str, has_pytest: bool) -> None:
    _write_executable(
        path,
        f"""#!/usr/bin/env bash
set -euo pipefail

if [[ "${{1-}}" == "-c" && "${{2-}}" == "import pytest" ]]; then
  if [[ {1 if has_pytest else 0} -eq 1 ]]; then
    exit 0
  fi
  printf '%s\n' 'No module named pytest' >&2
  exit 1
fi

if [[ "${{1-}}" == "-m" && "${{2-}}" == "compileall" ]]; then
  exit 0
fi

if [[ "${{1-}}" == *"run_tests_parallel.py" ]]; then
  printf '%s\n' "fake-python:{label}:${{1-}}"
  exit 0
fi

printf '%s\n' "unexpected invocation: $*" >&2
exit 42
""",
    )


def _make_fake_repo(tmp_path: Path, *, local_venv_has_pytest: bool, local_dotvenv_has_pytest: bool) -> Path:
    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True)

    project_root = Path(__file__).resolve().parent.parent
    shutil.copy2(project_root / "scripts" / "run_tests.sh", scripts_dir / "run_tests.sh")
    shutil.copy2(
        project_root / "scripts" / "run_tests_parallel.py",
        scripts_dir / "run_tests_parallel.py",
    )

    (repo_root / "tracked.py").write_text("print('tracked')\n", encoding="utf-8")

    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(
        ["git", "add", "tracked.py", "scripts/run_tests.sh", "scripts/run_tests_parallel.py"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    for name, has_pytest in ((".venv", local_dotvenv_has_pytest), ("venv", local_venv_has_pytest)):
        venv_dir = repo_root / name / "bin"
        venv_dir.mkdir(parents=True)
        (venv_dir / "activate").write_text("# fake activate\n", encoding="utf-8")
        _write_fake_python(venv_dir / "python", name, has_pytest)

    return repo_root


def _run_run_tests(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    script = repo_root / "scripts" / "run_tests.sh"
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )


def _make_home_hermes_venv(home: Path, *, has_pytest: bool) -> Path:
    venv_dir = home / ".hermes" / "hermes-agent" / "venv" / "bin"
    venv_dir.mkdir(parents=True)
    (venv_dir / "activate").write_text("# fake activate\n", encoding="utf-8")
    _write_fake_python(venv_dir / "python", "home-hermes-venv", has_pytest)
    return venv_dir.parent


def test_run_tests_sh_skips_pytestless_dotvenv_and_uses_venv(tmp_path: Path) -> None:
    repo_root = _make_fake_repo(
        tmp_path,
        local_venv_has_pytest=True,
        local_dotvenv_has_pytest=False,
    )

    proc = _run_run_tests(repo_root, "tests/test_widget.py")

    assert proc.returncode == 0, proc.stdout
    assert "skipping .venv" in proc.stdout
    assert "fake-python:venv" in proc.stdout
    assert "fake-python:.venv" not in proc.stdout


def test_run_tests_sh_falls_back_to_home_hermes_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = _make_fake_repo(
        tmp_path,
        local_venv_has_pytest=False,
        local_dotvenv_has_pytest=False,
    )
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    _make_home_hermes_venv(home, has_pytest=True)

    proc = _run_run_tests(repo_root, "tests/test_widget.py")

    assert proc.returncode == 0, proc.stdout
    assert "skipping .venv" in proc.stdout
    assert "skipping venv" in proc.stdout
    assert "fake-python:home-hermes-venv" in proc.stdout


def test_run_tests_sh_prefers_dotvenv_when_both_are_ready(tmp_path: Path) -> None:
    repo_root = _make_fake_repo(
        tmp_path,
        local_venv_has_pytest=True,
        local_dotvenv_has_pytest=True,
    )

    proc = _run_run_tests(repo_root, "tests/test_widget.py")

    assert proc.returncode == 0, proc.stdout
    assert "skipping .venv" not in proc.stdout
    assert "fake-python:.venv" in proc.stdout
    assert "fake-python:venv" not in proc.stdout


def test_run_tests_sh_fails_cleanly_when_no_candidate_has_pytest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _make_fake_repo(
        tmp_path,
        local_venv_has_pytest=False,
        local_dotvenv_has_pytest=False,
    )
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))

    proc = _run_run_tests(repo_root, "tests/test_widget.py")

    assert proc.returncode == 1
    assert "skipping .venv" in proc.stdout
    assert "skipping venv" in proc.stdout
    assert "no test-ready virtualenv found" in proc.stdout
    assert "launching test runner" not in proc.stdout
