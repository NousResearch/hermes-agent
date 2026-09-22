"""Credential-free pinned-update integration fixtures.

These tests use only disposable local Git repositories. They do not open SSH,
install software, contact a network host, or persist credentials.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


def git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr}")
    return result


def fixture(tmp_path: Path) -> dict[str, Path | str]:
    source = tmp_path / "source"
    bare = tmp_path / "origin.git"
    install = tmp_path / "install"
    git(tmp_path, "init", "--bare", str(bare))
    git(tmp_path, "init", "-b", "main", str(source))
    git(source, "config", "user.name", "fixture")
    git(source, "config", "user.email", "fixture@example.test")
    (source / "hermes_cli").mkdir()
    (source / "hermes_cli" / "update_rollout_protocol.json").write_text(
        json.dumps({"protocol": 1}) + "\n", encoding="utf-8"
    )
    (source / "payload.txt").write_text("A\n", encoding="utf-8")
    git(source, "add", ".")
    git(source, "commit", "-m", "A")
    commit_a = git(source, "rev-parse", "HEAD").stdout.strip()
    git(source, "remote", "add", "origin", str(bare))
    git(source, "push", "-u", "origin", "main")
    git(tmp_path, "clone", "-b", "main", str(bare), str(install))
    git(install, "config", "user.name", "fixture")
    git(install, "config", "user.email", "fixture@example.test")
    return {"source": source, "bare": bare, "install": install, "a": commit_a}


def commit_source(state: dict[str, Path | str], text: str, message: str) -> str:
    source = state["source"]
    assert isinstance(source, Path)
    (source / "payload.txt").write_text(text, encoding="utf-8")
    git(source, "add", "payload.txt")
    git(source, "commit", "-m", message)
    return git(source, "rev-parse", "HEAD").stdout.strip()


def request(state: dict[str, Path | str], target: str, current: str):
    from hermes_cli.update_target import SourceBinding, TargetRequest

    install = state["install"]
    assert isinstance(install, Path)
    origin = git(install, "remote", "get-url", "origin").stdout.strip()
    return TargetRequest(
        target,
        "1" * 32,
        current,
        SourceBinding(
            str(install.resolve()),
            origin,
            "refs/remotes/origin/main",
            target,
            "fixture",
            "d" * 64,
            1,
        ),
    )


def configure_install_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_cli.update_target as update_target

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "install_id").write_text("1" * 32 + "\n", encoding="utf-8")
    monkeypatch.setattr(update_target, "get_default_hermes_root", lambda: hermes_home)


def test_disposable_origin_applies_reviewed_b_after_branch_moves_to_c(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli.update_target import apply_pinned_target

    configure_install_id(tmp_path, monkeypatch)
    state = fixture(tmp_path)
    source = state["source"]
    install = state["install"]
    assert isinstance(source, Path) and isinstance(install, Path)
    current = str(state["a"])
    reviewed_b = commit_source(state, "B\n", "B")
    git(source, "push", "origin", "main")
    commit_c = commit_source(state, "C\n", "C")
    git(source, "push", "origin", "main")

    result = apply_pinned_target(install, request(state, reviewed_b, current))

    assert result.target_sha == reviewed_b
    assert reviewed_b != commit_c
    assert git(install, "rev-parse", "HEAD").stdout.strip() == reviewed_b
    assert (install / "payload.txt").read_text(encoding="utf-8") == "B\n"
    assert git(install, "remote").stdout.strip() == "origin"


def test_disposable_origin_refuses_reviewed_commit_removed_from_authorized_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli.update_target import PinnedTargetRefused, apply_pinned_target

    configure_install_id(tmp_path, monkeypatch)
    state = fixture(tmp_path)
    source = state["source"]
    bare = state["bare"]
    install = state["install"]
    assert isinstance(source, Path) and isinstance(bare, Path) and isinstance(install, Path)
    current = str(state["a"])
    reviewed_b = commit_source(state, "B\n", "B")
    git(source, "push", "origin", "main")
    git(install, "fetch", "origin", "main")
    git(bare, "update-ref", "refs/heads/main", current)

    with pytest.raises(PinnedTargetRefused, match="target-not-reachable"):
        apply_pinned_target(install, request(state, reviewed_b, current))

    assert git(install, "rev-parse", "HEAD").stdout.strip() == current
    assert (install / "payload.txt").read_text(encoding="utf-8") == "A\n"
