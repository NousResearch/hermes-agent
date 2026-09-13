"""End-to-end contracts for install.sh's constrained-network fallbacks.

These tests execute the installer stages with tiny transport doubles. They do
not inspect install.sh's source: the assertions observe the actual clone and uv
attempt order, the exit status, and the resulting installer output.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.linux_only

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


_FAKE_GIT = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import sys

args = sys.argv[1:]
log = Path(os.environ["FAKE_GIT_LOG"])
with log.open("a", encoding="utf-8") as fh:
    fh.write(" ".join(args) + "\n")

if args == ["--version"]:
    print("git version 2.45.0")
    raise SystemExit(0)

cwd = None
if len(args) >= 2 and args[0] == "-C":
    cwd = Path(args[1])
    args = args[2:]

if not args:
    raise SystemExit(1)

command = args[0]
if command == "clone":
    url = args[-2]
    destination = Path(args[-1])
    if url != os.environ.get("FAKE_GIT_MIRROR"):
        raise SystemExit(1)
    if os.environ.get("FAKE_GIT_MIRROR_OK", "1") != "1":
        raise SystemExit(1)
    (destination / ".git").mkdir(parents=True)
    raise SystemExit(0)

if command == "rev-parse":
    if cwd is not None and not (cwd / ".git").exists():
        raise SystemExit(1)
    if "--verify" in args:
        print(os.environ.get("FAKE_GIT_HEAD_SHA", "0123456789abcdef0123456789abcdef01234567"))
        raise SystemExit(0)
    for arg in args[1:]:
        if not arg.startswith("-"):
            if "^{commit}" in arg:
                target = arg.replace("^{commit}", "")
                print(os.environ.get("FAKE_GIT_EXPECTED_SHA", target))
                raise SystemExit(0)
            print(os.environ.get("FAKE_GIT_HEAD_SHA", arg))
            raise SystemExit(0)
    raise SystemExit(0 if cwd is not None and (cwd / ".git").exists() else 1)

if command == "cat-file":
    if os.environ.get("FAKE_GIT_CAT_FILE_FAIL", "0") == "1":
        raise SystemExit(1)
    raise SystemExit(0)

if command == "remote":
    if len(args) >= 3 and args[1] == "get-url":
        print(os.environ.get("FAKE_GIT_ORIGIN_URL", "https://github.com/NousResearch/hermes-agent.git"))
        raise SystemExit(0)
    raise SystemExit(0)

if command == "fetch":
    if os.environ.get("FAKE_GIT_FETCH_FAIL", "0") == "1":
        raise SystemExit(1)
    if len(args) >= 2 and args[1] == "origin" and os.environ.get("FAKE_GIT_FETCH_ORIGIN_FAIL", "0") == "1":
        raise SystemExit(1)
    raise SystemExit(0)

if command == "checkout":
    if os.environ.get("FAKE_GIT_CHECKOUT_FAIL", "0") == "1":
        raise SystemExit(1)
    raise SystemExit(0)

if command in {"fsck", "reset", "merge", "pull", "status", "ls-files", "stash", "update-ref"}:
    raise SystemExit(0)

raise SystemExit(0)
'''


_FAKE_UV = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import sys

args = sys.argv[1:]
log = Path(os.environ["FAKE_UV_LOG"])
if args == ["--version"]:
    print("uv 0.9.0")
    raise SystemExit(0)
if args[:2] == ["python", "find"]:
    print(os.environ["REAL_PYTHON"])
    raise SystemExit(0)
if args and args[0] == "sync":
    with log.open("a", encoding="utf-8") as fh:
        fh.write(os.environ.get("UV_DEFAULT_INDEX", "<unset>") + "\n")
    raise SystemExit(0 if os.environ.get("UV_DEFAULT_INDEX") else 1)
raise SystemExit(0)
'''


_FAKE_DPKG = "#!/bin/sh\nexit 0\n"


def _executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _base_env(tmp_path: Path, fake_bin: Path) -> dict[str, str]:
    return {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PYTHONPATH": "",
        "PYTHONHOME": "",
    }


def _run_stage(
    stage: str,
    *,
    install_dir: Path,
    hermes_home: Path,
    env: dict[str, str],
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "/bin/bash",
        str(INSTALL_SH),
        "--stage",
        stage,
        "--non-interactive",
        "--dir",
        str(install_dir),
        "--hermes-home",
        str(hermes_home),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_git_fallback_is_opt_in_by_default(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = _executable(fake_bin / "git", _FAKE_GIT)
    mirror = "mirror://hermes-agent"

    home = tmp_path / "home"
    log = tmp_path / "git.log"
    env = _base_env(tmp_path, fake_bin)
    env.update(
        FAKE_GIT_LOG=str(log),
        FAKE_GIT_MIRROR=mirror,
    )
    env.pop("GIT_FALLBACK_REPO_URL", None)
    result = _run_stage(
        "repository",
        install_dir=tmp_path / "install",
        hermes_home=home,
        env=env,
    )

    assert result.returncode != 0
    assert "Failed to clone repository" in result.stdout
    assert "Cloned via fallback mirror" not in result.stdout
    calls = log.read_text(encoding="utf-8").splitlines()
    assert not any(mirror in line for line in calls)


def test_git_fallback_is_official_first_and_fails_closed(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = _executable(fake_bin / "git", _FAKE_GIT)
    mirror = "mirror://hermes-agent"

    successful_home = tmp_path / "success-home"
    success_log = tmp_path / "success-git.log"
    success_env = _base_env(tmp_path, fake_bin)
    success_env.update(
        FAKE_GIT_LOG=str(success_log),
        FAKE_GIT_MIRROR=mirror,
        GIT_FALLBACK_REPO_URL=mirror,
    )
    result = _run_stage(
        "repository",
        install_dir=tmp_path / "success-install",
        hermes_home=successful_home,
        env=success_env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = success_log.read_text(encoding="utf-8").splitlines()
    clone_calls = [line for line in calls if line.startswith("clone ")]
    assert clone_calls
    assert clone_calls[-1].endswith(f"{mirror} {tmp_path / 'success-install'}")
    assert calls.index(clone_calls[-1]) > 0
    assert "Cloned via fallback mirror (Git objects verified)" in result.stdout
    assert "Repository ready" in result.stdout
    assert (tmp_path / "success-install" / ".git").is_dir()

    failed_home = tmp_path / "failed-home"
    failed_log = tmp_path / "failed-git.log"
    failed_env = _base_env(tmp_path, fake_bin)
    failed_env.update(
        FAKE_GIT_LOG=str(failed_log),
        FAKE_GIT_MIRROR=mirror,
        FAKE_GIT_MIRROR_OK="0",
        GIT_FALLBACK_REPO_URL=mirror,
    )
    failed = _run_stage(
        "repository",
        install_dir=tmp_path / "failed-install",
        hermes_home=failed_home,
        env=failed_env,
    )

    assert failed.returncode != 0
    assert "Failed to clone repository" in failed.stdout
    assert "Cloned via fallback mirror" not in failed.stdout
    assert not (tmp_path / "failed-install").exists()


def test_mirror_checkout_refuses_different_history_pre_effect(tmp_path: Path) -> None:
    """Negative regression for P1: mirror returns valid Git graph but different history."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = _executable(fake_bin / "git", _FAKE_GIT)
    mirror = "mirror://hermes-agent"
    expected_commit = "1111111111111111111111111111111111111111"

    home = tmp_path / "home"
    log = tmp_path / "git.log"
    install_dir = tmp_path / "install"
    env = _base_env(tmp_path, fake_bin)
    env.update(
        FAKE_GIT_LOG=str(log),
        FAKE_GIT_MIRROR=mirror,
        GIT_FALLBACK_REPO_URL=mirror,
        FAKE_GIT_CAT_FILE_FAIL="1",
        FAKE_GIT_HEAD_SHA="2222222222222222222222222222222222222222",
    )
    result = _run_stage(
        "repository",
        install_dir=install_dir,
        hermes_home=home,
        env=env,
        extra_args=["--commit", expected_commit],
    )

    assert result.returncode != 0
    assert "does not contain expected commit" in result.stdout or "does not match expected upstream commit" in result.stdout
    assert "Cloned via fallback mirror" not in result.stdout
    assert not install_dir.exists(), "Tree must be refused and cleaned up pre-effect"


def test_existing_checkout_fork_pins_from_configured_origin(tmp_path: Path) -> None:
    """Regression for P1: missing commit pin on a fork must fetch from origin (the fork), not Nous."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = _executable(fake_bin / "git", _FAKE_GIT)
    fork_url = "https://github.com/contributor-fork/hermes-agent.git"
    fork_commit = "3333333333333333333333333333333333333333"

    home = tmp_path / "home"
    log = tmp_path / "git.log"
    install_dir = tmp_path / "install"
    (install_dir / ".git").mkdir(parents=True)

    env = _base_env(tmp_path, fake_bin)
    env.update(
        FAKE_GIT_LOG=str(log),
        FAKE_GIT_ORIGIN_URL=fork_url,
        FAKE_GIT_CAT_FILE_FAIL="1",
    )
    result = _run_stage(
        "repository",
        install_dir=install_dir,
        hermes_home=home,
        env=env,
        extra_args=["--commit", fork_commit],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    pin_fetches = [line for line in calls if f"fetch origin {fork_commit}" in line]
    assert pin_fetches, f"Expected fetch from fork origin, got calls: {calls}"
    assert not any("NousResearch/hermes-agent" in line and fork_commit in line for line in calls)


def test_existing_checkout_fork_fetch_failure_fails_closed_without_switching_origin(tmp_path: Path) -> None:
    """Regression for P1: fetch failure on a fork fails closed and does not clobber refs with Nous mirror."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = _executable(fake_bin / "git", _FAKE_GIT)
    fork_url = "https://github.com/contributor-fork/hermes-agent.git"
    mirror = "mirror://hermes-agent"

    home = tmp_path / "home"
    log = tmp_path / "git.log"
    install_dir = tmp_path / "install"
    (install_dir / ".git").mkdir(parents=True)

    env = _base_env(tmp_path, fake_bin)
    env.update(
        FAKE_GIT_LOG=str(log),
        FAKE_GIT_ORIGIN_URL=fork_url,
        FAKE_GIT_FETCH_ORIGIN_FAIL="1",
        GIT_FALLBACK_REPO_URL=mirror,
        FAKE_GIT_MIRROR=mirror,
    )
    result = _run_stage(
        "repository",
        install_dir=install_dir,
        hermes_home=home,
        env=env,
    )

    assert result.returncode != 0
    assert "Refusing to overwrite fork tracking ref with upstream fallback mirror" in result.stdout
    calls = log.read_text(encoding="utf-8").splitlines()
    assert not any("refs/remotes/origin" in line and mirror in line for line in calls)


def test_uv_index_fallback_is_failure_triggered_and_respects_user_override(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(fake_bin / "dpkg", _FAKE_DPKG)
    fallback = "mirror://pypi-fallback"

    def run_uv_case(name: str, configured_index: str | None) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        install_dir = tmp_path / f"{name}-install"
        hermes_home = tmp_path / f"{name}-home"
        uv_path = hermes_home / "bin" / "uv"
        uv_path.parent.mkdir(parents=True)
        _executable(uv_path, _FAKE_UV)
        (install_dir / "venv" / "bin").mkdir(parents=True)
        (install_dir / "venv" / "bin" / "python").symlink_to(Path(sys.executable))
        (install_dir / "uv.lock").write_text("# transport double\n", encoding="utf-8")
        (install_dir / "pyproject.toml").write_text("[project]\nname='transport-double'\n", encoding="utf-8")
        log = tmp_path / f"{name}-uv.log"
        env = _base_env(tmp_path, fake_bin)
        env.update(
            FAKE_UV_LOG=str(log),
            REAL_PYTHON=sys.executable,
            UV_FALLBACK_INDEX=fallback,
        )
        if configured_index is not None:
            env["UV_DEFAULT_INDEX"] = configured_index
        else:
            env.pop("UV_DEFAULT_INDEX", None)
        completed = _run_stage("python-deps", install_dir=install_dir, hermes_home=hermes_home, env=env)
        attempts = log.read_text(encoding="utf-8").splitlines()
        return completed, attempts

    fallback_result, fallback_attempts = run_uv_case("fallback", None)
    assert fallback_result.returncode == 0, fallback_result.stdout + fallback_result.stderr
    assert fallback_attempts == ["<unset>", fallback]
    assert "fallback index" in fallback_result.stdout

    user_index = "mirror://user-selected"
    user_result, user_attempts = run_uv_case("user", user_index)
    assert user_result.returncode == 0, user_result.stdout + user_result.stderr
    assert user_attempts == [user_index]
    assert "fallback index" not in user_result.stdout