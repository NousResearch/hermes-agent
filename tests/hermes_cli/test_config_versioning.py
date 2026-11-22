"""Tests for git-native config versioning (issue #12238)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from hermes_cli.config_versioning import (
    NotARepo,
    _tracked_candidates,
    init_repo,
    is_repo,
    snapshot,
    status,
)


def test_snapshot_commits_config_and_ignores_runtime() -> None:
    with tempfile.TemporaryDirectory() as td:
        home = Path(td)
        (home / "config.yaml").write_text("model: test\n", encoding="utf-8")
        (home / "mcp.json").write_text("{}", encoding="utf-8")
        (home / "state.db").write_text("binary-ish", encoding="utf-8")
        (home / ".env").write_text("SECRET=1\n", encoding="utf-8")

        init_repo(home)
        assert is_repo(home) is True

        first = snapshot(home, "first")
        assert first != "nothing-to-commit"

        # Modify config, create runtime noise: only config should show as changed.
        (home / "config.yaml").write_text("model: changed\n", encoding="utf-8")
        (home / "state.db").write_text("more noise", encoding="utf-8")
        s = status(home)
        assert "config.yaml" in s
        assert "state.db" not in s
        assert ".env" not in s

        second = snapshot(home, "second")
        assert second != "nothing-to-commit"
        assert second != first


def test_snapshot_refuses_without_repo() -> None:
    with tempfile.TemporaryDirectory() as td:
        home = Path(td)
        (home / "config.yaml").write_text("x", encoding="utf-8")
        try:
            snapshot(home)
        except NotARepo:
            pass
        else:
            raise AssertionError("snapshot must refuse before init_repo")


def test_init_repo_is_idempotent_and_candidates_are_config_only() -> None:
    with tempfile.TemporaryDirectory() as td:
        home = Path(td)
        (home / "config.yaml").write_text("x", encoding="utf-8")
        (home / "SOUL.md").write_text("y", encoding="utf-8")
        (home / "cache").mkdir()
        init_repo(home)
        ignore = (home / ".gitignore").read_text(encoding="utf-8")
        init_repo(home)  # second call must not duplicate or fail
        assert ignore == (home / ".gitignore").read_text(encoding="utf-8")

        cands = _tracked_candidates(home)
        assert "config.yaml" in cands
        assert "SOUL.md" in cands
        assert "cache" not in cands

        assert snapshot(home) != "nothing-to-commit"
        assert snapshot(home) == "nothing-to-commit"  # clean tree
