from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def _linked_worktree_with_shared_venv(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "main checkout with spaces"
    worktree = tmp_path / "linked worktree with spaces"
    home = tmp_path / "home"
    repo.mkdir()
    home.mkdir()

    _run("git", "init", "-b", "main", cwd=repo)
    _run("git", "config", "user.name", "Hermes Tests", cwd=repo)
    _run("git", "config", "user.email", "tests@example.invalid", cwd=repo)

    scripts = repo / "scripts"
    scripts.mkdir()
    shutil.copy2(project_root / "scripts" / "run_tests.sh", scripts / "run_tests.sh")
    (scripts / "run_tests_parallel.py").write_text("# test stub\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "example.py").write_text("# test stub\n", encoding="utf-8")
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-m", "test fixture", cwd=repo)
    _run("git", "worktree", "add", "-b", "test-worktree", str(worktree), cwd=repo)

    shared_venv = repo / "venv" / "bin"
    shared_venv.mkdir(parents=True)
    (shared_venv / "activate").write_text("# test stub\n", encoding="utf-8")
    fake_python = shared_venv / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = \"-c\" ]; then exit 0; fi\n"
        "if [ \"${1:-}\" = \"-m\" ]; then exit 0; fi\n"
        "printf 'SHARED_VENV_SELECTED:%s\\n' \"$0\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return worktree, home, fake_python


def _run_worktree_wrapper(
    worktree: Path,
    home: Path,
    *,
    path: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env.pop("HERMES_PYTHON", None)
    if path is not None:
        env["PATH"] = path
    return subprocess.run(
        ["bash", "scripts/run_tests.sh", "tests/example.py", "-q"],
        cwd=worktree,
        env=env,
        text=True,
        capture_output=True,
    )


def test_run_tests_uses_main_checkout_venv_from_git_worktree(tmp_path: Path) -> None:
    worktree, home, fake_python = _linked_worktree_with_shared_venv(tmp_path)

    result = _run_worktree_wrapper(worktree, home)

    assert result.returncode == 0, result.stderr
    assert f"SHARED_VENV_SELECTED:{fake_python}" in result.stdout


def test_run_tests_shared_venv_does_not_require_git_path_format(
    tmp_path: Path,
) -> None:
    worktree, home, fake_python = _linked_worktree_with_shared_venv(tmp_path)
    real_git = shutil.which("git")
    assert real_git is not None
    shim_dir = tmp_path / "old git shim"
    shim_dir.mkdir()
    git_shim = shim_dir / "git"
    git_shim.write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "unknown=\"\"\n"
        "args=()\n"
        "for arg in \"$@\"; do\n"
        "  if [ \"$arg\" = \"--path-format=absolute\" ]; then\n"
        "    unknown=\"$arg\"\n"
        "  else\n"
        "    args+=(\"$arg\")\n"
        "  fi\n"
        "done\n"
        "if [ -n \"$unknown\" ]; then\n"
        "  printf '%s\\n' \"$unknown\"\n"
        f"  {shlex.quote(real_git)} \"${{args[@]}}\"\n"
        "  exit 0\n"
        "fi\n"
        f"exec {shlex.quote(real_git)} \"$@\"\n",
        encoding="utf-8",
    )
    git_shim.chmod(0o755)

    result = _run_worktree_wrapper(
        worktree,
        home,
        path=f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
    )

    assert result.returncode == 0, result.stderr
    assert f"SHARED_VENV_SELECTED:{fake_python}" in result.stdout
