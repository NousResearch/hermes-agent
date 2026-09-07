"""Tests for the git-verified completion guardrail (post-mortem 2026-09-07).

Covers the two dispatch/completion levers that stop a repo-editing card from
reaching `done` with uncommitted work or from sharing a checkout:

  1. Worktree isolation (`_maybe_upgrade_repo_scratch`): a card dispatched as
     `scratch` whose body references a git repo is auto-upgraded to an isolated
     per-task worktree, so two concurrent cards never share one working tree.
  2. Mandatory skill wiring (`_default_spawn`): every default-spawned worker
     force-loads `kanban-git-verified-completion`, so it commits + pushes before
     `kanban_complete` (avoiding the kernel-level gate rejection).

The dispatch machinery moved to ``hermes_cli.kanban_db_dispatch`` in the
Sep 2026 decomposition, so this file imports it as ``kbd`` (and the connect
layer as ``kbc``) rather than reaching them through ``kanban_db``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _make_git_repo(base: Path) -> str:
    repo = base / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    def git(*args, **kw):
        return subprocess.run(
            ["git"] + list(args), cwd=repo, capture_output=True, text=True, **kw
        )

    git("init", "-q")
    git("config", "user.email", "t@t.t")
    git("config", "user.name", "t")
    return str(repo)


def _mk(conn, *, title="card", body=None, workspace_kind="scratch", **kw):
    return kb.create_task(
        conn, title=title, body=body,
        workspace_kind=workspace_kind, assignee="conan", **kw,
    )


def test_repo_referencing_scratch_card_upgraded_to_worktree(kanban_home, tmp_path):
    """Acceptance: a card that edits a repo must never be dispatched as a shared
    `scratch` checkout. `_maybe_upgrade_repo_scratch` auto-upgrades it to an
    isolated per-task worktree."""
    repo = _make_git_repo(tmp_path)
    conn = kbc.connect()
    try:
        tid = _mk(conn, body=f"implement the fix in {repo}/src/main.py")
        task = kb.get_task(conn, tid)
        assert task.workspace_kind == "scratch"

        upgraded = kbd._maybe_upgrade_repo_scratch(conn, task, board=None)

        assert upgraded.workspace_kind == "worktree"
        # The worktree must anchor on the actual repo the card edits.
        assert Path(upgraded.workspace_path).resolve() == Path(repo).resolve()
        assert upgraded.branch_name == f"wt/{tid}"

        # The upgrade is observable in the event log.
        evs = kb.list_events(conn, tid)
        kinds = [e.kind for e in evs]
        assert "workspace_upgraded_to_worktree" in kinds
    finally:
        conn.close()


def test_non_repo_scratch_card_left_alone_with_warning(kanban_home, tmp_path):
    """A scratch card that references no git repo stays scratch (we cannot
    isolate on a repo that doesn't exist), but the hazard is surfaced as an
    event so an operator can pin it to a project if it edits a repo."""
    conn = kbc.connect()
    try:
        tid = _mk(conn, body="update the SSOT rules per the incident")
        task = kb.get_task(conn, tid)

        result = kbd._maybe_upgrade_repo_scratch(conn, task, board=None)

        assert result.workspace_kind == "scratch"
        evs = kb.list_events(conn, tid)
        kinds = [e.kind for e in evs]
        assert "scratch_repo_reference_warning" in kinds
    finally:
        conn.close()


def test_default_spawn_force_loads_git_verify_skill(kanban_home, monkeypatch):
    """Acceptance: the mandatory git-verified-completion skill is wired into the
    default dispatch path — even a worker with no task.skills gets it."""
    repo = _make_git_repo(kanban_home.parent)
    conn = kbc.connect()
    try:
        tid = kb.create_task(
            conn, title="spawn", assignee="conan",
            body=f"do it in {repo}", workspace_kind="scratch",
        )
        task = kb.get_task(conn, tid)
    finally:
        conn.close()

    # Task was upgraded by dispatch policy? For the spawn test we just need a
    # task object; the skill wiring is independent of workspace_kind.
    assert task.id == tid

    captured = {}

    class _FakeProc:
        pid = 4242

    real_popen = subprocess.Popen

    def fake_popen(args, **kw):
        captured["args"] = list(args)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    # Resolve the assignee's profile without touching a real one: these are
    # imported inside `_default_spawn` from hermes_cli.profiles.
    monkeypatch.setattr(
        "hermes_cli.profiles.normalize_profile_name", lambda name: name
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.resolve_profile_env", lambda name: str(kanban_home)
    )

    try:
        pid = kbd._default_spawn(task, workspace=repo, board=None)
    finally:
        subprocess.Popen = real_popen

    assert pid == 4242
    args = captured["args"]
    assert "--skills" in args
    i = args.index("--skills")
    assert i + 1 < len(args)
    assert args[i + 1] == kbd.DEFAULT_GIT_VERIFY_SKILL
