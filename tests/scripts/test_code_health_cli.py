"""scripts/check runner, git hooks, CLI output modes and regex-rule scope, on real git repos.

Fixture repos carry a copy of this checkout's engine (scripts/check, scripts/code_health, the
guard scripts, pyproject's ruff pin) so the runner and the hooks execute exactly as they would
in a clone, against tiny trees that keep every check fast.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from scripts.code_health import cli
from scripts.code_health.config import ENFORCEMENT

REPO = Path(__file__).resolve().parents[2]
_GUARD_SCRIPTS = (
    "scripts/check-windows-footguns.py", "scripts/check_bash_shebangs.py",
    "scripts/check_no_tmp_literals.py", "scripts/check_config_yaml_writers.py",
    "scripts/ci/check_os_marker_fakes.py", "scripts/check-case-collisions.py",
    "scripts/ci/check_lazy_deps_imports.py", "scripts/ci/check_profile_archive_boundary.py",
)
_ENGINE = ("scripts/check", "scripts/ci/profile_scope_patterns.json", *_GUARD_SCRIPTS)
_LEGACY = "def legacy(x):\n" + "".join(f"    if x == {i}:\n        return {i}\n" for i in range(21))
_GROWN = _LEGACY + "    if x == 99:\n        return 99\n"
_ENV_COPY = "import os\n\n\ndef child_env():\n    env = os.environ.copy()\n    return env\n"
_SWITCH = "scripts/code_health/config.py"


def _env() -> dict[str, str]:
    """The caller's env minus anything that would point git at another repo or config."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return env


def _sh(cwd: Path, *argv: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), cwd=cwd, env=env or _env(), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                          timeout=600, check=False)


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    proc = _sh(repo, "git", *args, env=env)
    assert proc.returncode == 0, (args, proc.stdout, proc.stderr)
    return proc.stdout.strip()


def _write(repo: Path, files: Mapping[str, str | None]) -> None:
    for rel, text in files.items():
        path = repo / rel
        if text is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


def _commit(repo: Path, files: Mapping[str, str | None], env: dict[str, str] | None = None) -> str:
    _write(repo, files)
    _git(repo, "add", "--all", "--", *files)
    _git(repo, "commit", "-q", "-m", "step", env=env)
    return _git(repo, "rev-parse", "HEAD")


def _init(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    (path / ".gitignore").write_text(".venv\n", encoding="utf-8")
    shutil.copy(REPO / "pyproject.toml", path / "pyproject.toml")
    return path


def _add_engine(repo: Path) -> list[str]:
    for rel in _ENGINE:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, repo / rel)
    shutil.copytree(REPO / "scripts/code_health", repo / "scripts/code_health",
                    ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    return ["scripts"]


def _engine_repo(tmp_path: Path) -> Path:
    """A repo whose first commit already carries the checker, with one clean module."""
    repo = _init(tmp_path / "repo")
    _write(repo, {"pkg/a.py": "def a():\n    return 1\n"})
    _add_engine(repo)
    _git(repo, "add", "--all", "--", ".gitignore", "pyproject.toml", "pkg", "scripts")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _check(repo: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return _sh(repo, sys.executable, str(repo / "scripts/check"), *args, env=env)


def _ratchet_repo(tmp_path: Path, switch: str | None = None) -> tuple[Path, str]:
    repo = _init(tmp_path / "repo")
    (repo / "scripts/ci").mkdir(parents=True)
    shutil.copy2(REPO / "scripts/ci/profile_scope_patterns.json", repo / "scripts/ci/")
    files: dict[str, str | None] = {"pkg/a.py": _LEGACY, **({_SWITCH: switch} if switch else {})}
    _write(repo, files)
    _git(repo, "add", "--all", "--", ".gitignore", "pyproject.toml", "scripts", "pkg")
    _git(repo, "commit", "-q", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


# --- F22: selectors are validated before anything runs ----------------------------------------


@pytest.mark.parametrize("selector", ["heath", ",", "", "heath,rof", "health,rof"])
def test_unknown_or_empty_selector_is_a_usage_error(selector):
    proc = _sh(REPO, sys.executable, str(REPO / "scripts/check"), "--only", selector)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "health" in proc.stderr and "shebangs" in proc.stderr  # lists the valid names
    assert "checks, ok" not in proc.stdout


def test_valid_selector_runs_its_check_and_keeps_its_status(tmp_path):
    repo = _engine_repo(tmp_path)
    _write(repo, {"pkg/c.py": _ENV_COPY})
    _git(repo, "add", "--", "pkg/c.py")
    proc = _check(repo, "--staged", "--only", "health,shebangs", "--base", "HEAD")
    assert proc.returncode == 1 and "2 checks, FAILED: health" in proc.stdout, proc.stdout
