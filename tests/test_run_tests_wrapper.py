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


def _wrapper_checkouts(tmp_path: Path) -> tuple[Path, Path, Path]:
    project_root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "main checkout with spaces"
    worktree = tmp_path / "linked worktree with spaces"
    home = tmp_path / "home with spaces"
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

    return repo, worktree, home


def _fake_venv_python(
    venv: Path,
    *,
    windows_layout: bool = False,
    has_pytest: bool = True,
) -> Path:
    scripts_dir = venv / ("Scripts" if windows_layout else "bin")
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "activate").write_text("# test stub\n", encoding="utf-8")
    fake_python = scripts_dir / ("python.exe" if windows_layout else "python")
    pytest_probe_status = 0 if has_pytest else 1
    fake_python.write_text(
        "#!/bin/sh\n"
        f'if [ "${{1:-}}" = "-c" ]; then exit {pytest_probe_status}; fi\n'
        'if [ "${1:-}" = "-m" ]; then exit 0; fi\n'
        "printf 'WRAPPER_PYTHON_SELECTED:%s\\n' \"$0\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return fake_python


def _run_wrapper(
    checkout: Path,
    home: Path,
    *,
    path: str | None = None,
    hermes_python: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env.pop("HERMES_PYTHON", None)
    if path is not None:
        env["PATH"] = path
    if hermes_python is not None:
        env["HERMES_PYTHON"] = str(hermes_python)
    return subprocess.run(
        ["bash", "scripts/run_tests.sh", "tests/example.py", "-q"],
        cwd=checkout,
        env=env,
        text=True,
        capture_output=True,
    )


def _assert_selected(
    result: subprocess.CompletedProcess[str], fake_python: Path
) -> None:
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert f"WRAPPER_PYTHON_SELECTED:{fake_python}" in result.stdout


def test_run_tests_uses_posix_venv_from_main_checkout_in_worktree(
    tmp_path: Path,
) -> None:
    repo, worktree, home = _wrapper_checkouts(tmp_path)
    fake_python = _fake_venv_python(repo / "venv")

    result = _run_wrapper(worktree, home)

    _assert_selected(result, fake_python)


def test_run_tests_uses_windows_venv_from_main_checkout_in_worktree(
    tmp_path: Path,
) -> None:
    repo, worktree, home = _wrapper_checkouts(tmp_path)
    fake_python = _fake_venv_python(repo / "venv", windows_layout=True)

    result = _run_wrapper(worktree, home)

    _assert_selected(result, fake_python)


def test_run_tests_shared_venv_does_not_require_git_path_format(
    tmp_path: Path,
) -> None:
    repo, worktree, home = _wrapper_checkouts(tmp_path)
    fake_python = _fake_venv_python(repo / "venv")
    real_git = shutil.which("git")
    assert real_git is not None
    shim_dir = tmp_path / "old git shim"
    shim_dir.mkdir()
    git_shim = shim_dir / "git"
    git_shim.write_text(
        "#!/bin/sh\n"
        'for arg in "$@"; do\n'
        '  if [ "$arg" = "--path-format=absolute" ]; then\n'
        "    printf 'error: unknown option %s\\n' \"$arg\" >&2\n"
        "    exit 129\n"
        "  fi\n"
        "done\n"
        f'exec {shlex.quote(real_git)} "$@"\n',
        encoding="utf-8",
    )
    git_shim.chmod(0o755)

    result = _run_wrapper(
        worktree,
        home,
        path=f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
    )

    _assert_selected(result, fake_python)


def test_run_tests_skips_venv_without_pytest_before_shared_venv(
    tmp_path: Path,
) -> None:
    repo, worktree, home = _wrapper_checkouts(tmp_path)
    _fake_venv_python(worktree / ".venv", has_pytest=False)
    fake_python = _fake_venv_python(repo / "venv")

    result = _run_wrapper(worktree, home)

    _assert_selected(result, fake_python)
    assert "skipping venv without pytest" in result.stderr


def test_run_tests_falls_back_to_hermes_python(tmp_path: Path) -> None:
    _repo, worktree, home = _wrapper_checkouts(tmp_path)
    fake_python = _fake_venv_python(tmp_path / "nix dev venv with spaces")

    result = _run_wrapper(worktree, home, hermes_python=fake_python)

    _assert_selected(result, fake_python)
    assert "using Nix dev venv via HERMES_PYTHON" in result.stdout


def test_run_tests_uses_local_venv_in_non_worktree_checkout(tmp_path: Path) -> None:
    repo, _worktree, home = _wrapper_checkouts(tmp_path)
    fake_python = _fake_venv_python(repo / ".venv")

    result = _run_wrapper(repo, home)

    _assert_selected(result, fake_python)
