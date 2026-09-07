"""Regression tests for conversation-worktree Python environment setup."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agent.conversation_worktree import ConversationWorktreeManager
from agent.conversation_worktree_policy import ConversationWorktreePolicy
from hermes_state import SessionDB


def _git(args: list[str], cwd: Path) -> None:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_new_conversation_worktree_links_source_python_environment(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    _git(["config", "user.name", "Test"], repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["add", "README.md"], repo)
    _git(["commit", "-q", "-m", "seed"], repo)

    source_python = repo / ".venv" / "bin" / "python"
    source_python.parent.mkdir(parents=True)
    source_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    source_python.chmod(0o755)

    db = SessionDB(tmp_path / "state.db")
    try:
        manager = ConversationWorktreeManager(
            ConversationWorktreePolicy(
                enabled=True,
                source_worktree=repo,
                worktree_root=tmp_path / "conversation-worktrees",
                branch_prefix="hermes/session",
                bootstrap=False,
                bootstrap_command=(),
                bootstrap_timeout=1.0,
                create_timeout=3.0,
                retain_until_explicit_cleanup=True,
            ),
            db,
        )
        binding = manager.bind_new_root_session(
            "new-python-environment", conversation_kind="interactive"
        )
    finally:
        db.close()

    assert binding is not None
    linked = binding.path / ".venv"
    assert linked.is_symlink()
    assert linked.resolve() == (repo / ".venv").resolve()
    assert (linked / "bin" / "python").is_file()
