"""Behavior tests for the installer's managed-checkout update path."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)


def _run(*args: str | Path, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run("git", *args, cwd=repo)


def _make_remote(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    _run("git", "init", "--bare", remote)
    _run("git", "init", seed)
    _git(seed, "config", "user.email", "installer-test@example.invalid")
    _git(seed, "config", "user.name", "Installer Test")
    (seed / "base.txt").write_text("base\n", encoding="utf-8")
    _git(seed, "add", "base.txt")
    _git(seed, "commit", "-m", "initial")
    _git(seed, "branch", "-M", "main")
    _git(seed, "remote", "add", "origin", remote.as_uri())
    _git(seed, "push", "-u", "origin", "main")
    return remote, seed


def _push_remote_update(seed: Path) -> str:
    (seed / "remote.txt").write_text("remote update\n", encoding="utf-8")
    _git(seed, "add", "remote.txt")
    _git(seed, "commit", "-m", "remote update")
    _git(seed, "push", "origin", "main")
    return _git(seed, "rev-parse", "HEAD").stdout.strip()


def _run_repository_stage(tmp_path: Path, install_dir: Path) -> list[list[str]]:
    real_git = shutil.which("git")
    assert real_git is not None

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git_log = tmp_path / "git.log"
    wrapper = bin_dir / "git"
    wrapper.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$HERMES_TEST_GIT_LOG"\n'
        f"exec {shlex.quote(real_git)} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "HERMES_HOME": str(tmp_path / "hermes-home"),
            "HERMES_TEST_GIT_LOG": str(git_log),
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
        }
    )
    subprocess.run(
        [
            "bash",
            str(INSTALL_SH),
            "--stage",
            "repository",
            "--dir",
            str(install_dir),
            "--branch",
            "main",
            "--non-interactive",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.split() for line in git_log.read_text(encoding="utf-8").splitlines()]


def _network_update_commands(commands: list[list[str]]) -> list[list[str]]:
    return [args for args in commands if args and args[0] in {"fetch", "pull"}]


def test_shallow_managed_checkout_uses_one_bounded_fetch(tmp_path: Path) -> None:
    remote, seed = _make_remote(tmp_path)
    install_dir = tmp_path / "install"
    _run(
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        "main",
        remote.as_uri(),
        install_dir,
    )
    expected_head = _push_remote_update(seed)

    commands = _run_repository_stage(tmp_path, install_dir)

    network_commands = _network_update_commands(commands)
    assert len(network_commands) == 1, network_commands
    assert network_commands[0][0] == "fetch"
    if "--depth" in network_commands[0]:
        depth_index = network_commands[0].index("--depth")
        assert network_commands[0][depth_index + 1] == "1"
    else:
        assert "--depth=1" in network_commands[0]
    assert _git(install_dir, "rev-parse", "HEAD").stdout.strip() == expected_head
    assert _git(install_dir, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"


def test_shallow_ahead_only_checkout_resets_to_remote_tip(tmp_path: Path) -> None:
    remote, seed = _make_remote(tmp_path)
    install_dir = tmp_path / "install"
    _run(
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        "main",
        remote.as_uri(),
        install_dir,
    )
    expected_head = _git(seed, "rev-parse", "HEAD").stdout.strip()
    _git(install_dir, "config", "user.email", "installer-test@example.invalid")
    _git(install_dir, "config", "user.name", "Installer Test")
    (install_dir / "local-only.txt").write_text("local commit\n", encoding="utf-8")
    _git(install_dir, "add", "local-only.txt")
    _git(install_dir, "commit", "-m", "ahead-only local commit")

    commands = _run_repository_stage(tmp_path, install_dir)

    network_commands = _network_update_commands(commands)
    assert len(network_commands) == 1, network_commands
    assert network_commands[0][0] == "fetch"
    assert "--depth" in network_commands[0] or "--depth=1" in network_commands[0]
    assert _git(install_dir, "rev-parse", "HEAD").stdout.strip() == expected_head
    assert _git(install_dir, "rev-parse", "origin/main").stdout.strip() == expected_head
    assert not (install_dir / "local-only.txt").exists()
    assert _git(install_dir, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"


def test_full_diverged_checkout_recovers_without_shallowing_or_losing_changes(
    tmp_path: Path,
) -> None:
    remote, seed = _make_remote(tmp_path)
    install_dir = tmp_path / "install"
    _run("git", "clone", "--branch", "main", remote.as_uri(), install_dir)
    _git(install_dir, "config", "user.email", "installer-test@example.invalid")
    _git(install_dir, "config", "user.name", "Installer Test")

    (install_dir / "local-only.txt").write_text("local commit\n", encoding="utf-8")
    _git(install_dir, "add", "local-only.txt")
    _git(install_dir, "commit", "-m", "local-only commit")
    (install_dir / "base.txt").write_text("preserve this dirty change\n", encoding="utf-8")
    expected_head = _push_remote_update(seed)

    commands = _run_repository_stage(tmp_path, install_dir)

    network_commands = _network_update_commands(commands)
    assert len(network_commands) == 1, network_commands
    assert network_commands[0][0] == "fetch"
    assert "--depth" not in network_commands[0]
    assert "--depth=1" not in network_commands[0]
    assert _git(install_dir, "rev-parse", "HEAD").stdout.strip() == expected_head
    assert _git(install_dir, "rev-parse", "--is-shallow-repository").stdout.strip() == "false"
    assert (install_dir / "base.txt").read_text(encoding="utf-8") == (
        "preserve this dirty change\n"
    )
