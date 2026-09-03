"""Client-only update for runtime-free remote Desktop installs.

Regression: a missing ``venv/bin/hermes`` used to abort as a broken local
install even when Desktop was a remote-only client. These tests drive the
stdlib updater against a real git pair and a temp ``HERMES_HOME``.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from hermes_cli.client_only_update import (
    UpdateSurface,
    classify_update_kind,
    remote_mode_from_connection_docs,
    run_client_only_update,
)


def _git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Hermes Test",
        "GIT_AUTHOR_EMAIL": "hermes@example.invalid",
        "GIT_COMMITTER_NAME": "Hermes Test",
        "GIT_COMMITTER_EMAIL": "hermes@example.invalid",
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return (result.stdout or "").strip()


def _init_repo(path: Path, *, message: str = "initial") -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-b", "main")
    _git(path, "config", "commit.gpgSign", "false")
    _git(path, "config", "user.name", "Hermes Test")
    _git(path, "config", "user.email", "hermes@example.invalid")
    (path / "README.md").write_text(message + "\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", message)
    return _git(path, "rev-parse", "HEAD")


def _clone_behind(origin: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git(origin.parent, "clone", "--quiet", str(origin), str(dest))
    _git(dest, "config", "commit.gpgSign", "false")
    return _git(dest, "rev-parse", "HEAD")


def _write_exe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


# ── classification ────────────────────────────────────────────────────────


def test_classify_runtime_free_remote_is_client_only():
    assert (
        classify_update_kind(
            UpdateSurface(has_venv_hermes=False, has_venv_python=False, remote_mode=True)
        )
        == "client_only"
    )


def test_classify_present_venv_stays_full_install_even_in_remote_mode():
    assert (
        classify_update_kind(
            UpdateSurface(has_venv_hermes=True, has_venv_python=True, remote_mode=True)
        )
        == "full_install"
    )


def test_classify_missing_venv_in_local_mode_is_broken_local():
    assert (
        classify_update_kind(
            UpdateSurface(
                has_venv_hermes=False,
                has_venv_python=False,
                remote_mode=False,
                has_bootstrap_marker=True,
            )
        )
        == "broken_local"
    )


def test_classify_partial_venv_is_broken_local():
    assert (
        classify_update_kind(
            UpdateSurface(has_venv_hermes=True, has_venv_python=False, remote_mode=True)
        )
        == "broken_local"
    )


def test_registry_primary_remote_wins_over_legacy_local_mode():
    assert remote_mode_from_connection_docs(
        {"mode": "local"},
        {
            "launchMode": "primary",
            "primary": "vps",
            "lastUsed": "local",
            "connections": [
                {"id": "local", "kind": "local"},
                {"id": "vps", "kind": "ssh"},
            ],
        },
    )


def test_last_used_local_with_last_used_launch_mode_is_not_remote():
    assert not remote_mode_from_connection_docs(
        {"mode": "local"},
        {
            "launchMode": "last-used",
            "primary": "vps",
            "lastUsed": "local",
            "connections": [
                {"id": "local", "kind": "local"},
                {"id": "vps", "kind": "ssh"},
            ],
        },
    )


# ── apply ─────────────────────────────────────────────────────────────────


def test_missing_runtime_success_and_commit_readback(tmp_path: Path):
    origin = tmp_path / "origin"
    first = _init_repo(origin)
    (origin / "README.md").write_text("second\n", encoding="utf-8")
    _git(origin, "add", "README.md")
    _git(origin, "commit", "-m", "second")
    second = _git(origin, "rev-parse", "HEAD")

    install = tmp_path / "home" / "hermes-agent"
    _clone_behind(origin, install)
    _git(install, "reset", "--hard", first)
    connections = tmp_path / "home" / "connections.json"
    connections.write_text(
        json.dumps(
            {
                "launchMode": "primary",
                "primary": "vps",
                "connections": [{"id": "vps", "kind": "ssh", "label": "vps"}],
            }
        ),
        encoding="utf-8",
    )
    before_connections = connections.read_text(encoding="utf-8")

    result = run_client_only_update(
        install,
        hermes_home=tmp_path / "home",
        remote_mode=True,
        force_client_only=True,
        skip_desktop_build=True,
    )

    assert result.ok
    assert result.kind == "client_only"
    assert result.installed_commit == second
    assert result.fleet_restarted is False
    assert result.connections_rewritten is False
    assert _git(install, "rev-parse", "HEAD") == second
    assert connections.read_text(encoding="utf-8") == before_connections
    receipt = json.loads((tmp_path / "home" / "logs" / "update_receipts" / "latest.json").read_text())
    assert receipt["kind"] == "client_only"
    assert receipt["post_update"]["sha"] == second
    assert any(
        step.get("name") == "fleet_restart" and step.get("skipped") for step in receipt["steps"]
    )


def test_broken_local_missing_runtime_is_refused(tmp_path: Path):
    install = tmp_path / "hermes-agent"
    _init_repo(install)
    result = run_client_only_update(install, remote_mode=False, skip_desktop_build=True)
    assert not result.ok
    assert result.exit_code == 3
    assert result.kind == "broken_local"
    assert "needs repair" in result.message


def test_full_install_is_refused_by_client_only_path(tmp_path: Path):
    install = tmp_path / "hermes-agent"
    _init_repo(install)
    _write_exe(install / "venv" / "bin" / "hermes")
    _write_exe(install / "venv" / "bin" / "python3")
    result = run_client_only_update(
        install, remote_mode=True, force_client_only=True, skip_desktop_build=True
    )
    assert not result.ok
    assert result.kind == "full_install"
    assert result.exit_code == 64


def test_git_failure_rolls_back(tmp_path: Path):
    origin = tmp_path / "origin"
    first = _init_repo(origin)
    install = tmp_path / "home" / "hermes-agent"
    _clone_behind(origin, install)
    # Break fetch by pointing origin at a missing remote.
    _git(install, "remote", "set-url", "origin", str(tmp_path / "missing.git"))

    result = run_client_only_update(
        install,
        hermes_home=tmp_path / "home",
        force_client_only=True,
        skip_desktop_build=True,
    )

    assert not result.ok
    assert result.rolled_back
    assert _git(install, "rev-parse", "HEAD") == first


def test_build_failure_rolls_back_and_does_not_claim_new_commit(tmp_path: Path):
    origin = tmp_path / "origin"
    first = _init_repo(origin)
    (origin / "README.md").write_text("second\n", encoding="utf-8")
    _git(origin, "add", "README.md")
    _git(origin, "commit", "-m", "second")
    second = _git(origin, "rev-parse", "HEAD")

    install = tmp_path / "home" / "hermes-agent"
    _clone_behind(origin, install)
    _git(install, "reset", "--hard", first)
    desktop = install / "apps" / "desktop"
    desktop.mkdir(parents=True)
    (desktop / "package.json").write_text("{}", encoding="utf-8")

    result = run_client_only_update(
        install,
        hermes_home=tmp_path / "home",
        force_client_only=True,
        build_command=["/bin/sh", "-c", "echo build-failed >&2; exit 1"],
    )

    assert not result.ok
    assert result.exit_code == 6
    assert result.rolled_back
    assert result.installed_commit == first
    assert _git(install, "rev-parse", "HEAD") == first
    assert _git(install, "rev-parse", "HEAD") != second


def test_dependency_command_failure_rolls_back(tmp_path: Path):
    origin = tmp_path / "origin"
    first = _init_repo(origin)
    (origin / "README.md").write_text("second\n", encoding="utf-8")
    _git(origin, "add", "README.md")
    _git(origin, "commit", "-m", "second")

    install = tmp_path / "home" / "hermes-agent"
    _clone_behind(origin, install)
    _git(install, "reset", "--hard", first)
    desktop = install / "apps" / "desktop"
    desktop.mkdir(parents=True)
    (desktop / "package.json").write_text("{}", encoding="utf-8")

    result = run_client_only_update(
        install,
        hermes_home=tmp_path / "home",
        force_client_only=True,
        build_command=["/bin/sh", "-c", "echo npm ERR! missing peer >&2; exit 2"],
    )

    assert not result.ok
    assert result.rolled_back
    assert _git(install, "rev-parse", "HEAD") == first


@pytest.mark.skipif(not shutil.which("git"), reason="git required")
def test_client_only_never_invokes_fleet_or_gateway_helpers(tmp_path: Path, monkeypatch):
    origin = tmp_path / "origin"
    _init_repo(origin)
    install = tmp_path / "home" / "hermes-agent"
    _clone_behind(origin, install)

    calls: list[list[str]] = []

    def recording_run(args, *, cwd, env=None):
        calls.append(list(args))
        return subprocess.run(
            list(args),
            cwd=str(cwd),
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    result = run_client_only_update(
        install,
        hermes_home=tmp_path / "home",
        force_client_only=True,
        skip_desktop_build=True,
        run=recording_run,
    )

    assert result.ok
    joined = [" ".join(c) for c in calls]
    assert all("fleet" not in " ".join(c) for c in calls)
    assert all("gateway" not in c for c in joined)
    assert all(c[0] == "git" for c in calls)
