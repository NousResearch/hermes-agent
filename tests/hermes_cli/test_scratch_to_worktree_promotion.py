"""Tests for the scratch→worktree auto-promotion helper.

When ``import_github_issues.py`` (or any other importer) creates a Kanban
task without ``--project``, the task lands with ``workspace_kind='scratch'``.
The dispatcher-side helper ``_maybe_promote_scratch_to_worktree`` catches
that case at resolve time: if the task body references source code
patterns AND a registered project matches, the workspace is promoted to
a linked worktree on the matching project's primary_path.

These tests pin both the matching logic and the no-match path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@dataclass
class FakeTask:
    """Minimal stand-in for ``hermes_cli.kanban_db.Task``.

    The helper only reads ``title``, ``body`` and ``workspace_kind``,
    so a slim dataclass is enough — no need to drag in the full Task
    constructor or its DB row decoder.
    """

    title: str = ""
    body: str = ""
    workspace_kind: str = "scratch"
    workspace_path: str | None = None
    branch_name: str | None = None
    id: str = "t_fake"


@pytest.fixture
def chiron_project(tmp_path, monkeypatch):
    """Register a fake 'Chiron' project pointing at a tmp git repo.

    The helper looks up projects by their primary_path via projects_db;
    we override that lookup to a controlled table instead of touching
    the user's real projects.db.
    """
    fake_repo = tmp_path / "Chiron"
    fake_repo.mkdir()
    # .git dir so _git_toplevel returns something — not strictly needed
    # for the helper, but keeps the test path realistic.
    (fake_repo / ".git").mkdir()

    # Patch projects_db.connect_closing to return our fake row.
    from hermes_cli import projects_db as _pdb

    @dataclass
    class FakeRow:
        slug: str
        name: str
        primary_path: str

        def __getitem__(self, key: str):
            return getattr(self, key)

    rows = [FakeRow(slug="chiron", name="Chiron", primary_path=str(fake_repo))]

    class FakeConn:
        def execute(self, *_args, **_kwargs):
            class _Cur:
                def fetchall(_self):
                    return rows

            return _Cur()

        def close(self):
            pass

    @dataclass
    class FakeCM:
        rows: list

        def __enter__(self):
            return FakeConn()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(_pdb, "connect_closing", lambda: FakeCM(rows=rows))
    return fake_repo


class TestScratchToWorktreePromotion:
    def test_promotes_when_body_re_ferences_src_agents(self, chiron_project):
        """Issue body with src/agents/ + project basename → promote."""
        task = FakeTask(
            title="#339 — Pattern 42 fix",
            body="Fix the gate logic in src/agents/architect_agent.py for Chiron",
        )
        result = kb._maybe_promote_scratch_to_worktree(task)
        assert result == chiron_project, (
            f"expected promotion to {chiron_project}, got {result}"
        )

    def test_promotes_when_body_re_ferences_tests(self, chiron_project):
        task = FakeTask(
            title="add tests",
            body="Add a test for the new behaviour in tests/test_x.py (Chiron)",
        )
        assert kb._maybe_promote_scratch_to_worktree(task) == chiron_project

    def test_promotes_when_body_re_ferences_pytest(self, chiron_project):
        task = FakeTask(
            title="run tests",
            body="pytest tests/ should pass after the fix — Chiron vårdcentral",
        )
        assert kb._maybe_promote_scratch_to_worktree(task) == chiron_project

    def test_no_promotion_for_non_code_task(self, chiron_project):
        """Tasks without source-code markers must NOT trigger promotion."""
        task = FakeTask(
            title="follow up on email",
            body="Send the spec to the team and schedule a meeting.",
        )
        assert kb._maybe_promote_scratch_to_worktree(task) is None

    def test_no_promotion_when_project_does_not_match(self, tmp_path, monkeypatch):
        """Body has code markers but no registered project basename matches."""
        unrelated_repo = tmp_path / "SomeOther"
        unrelated_repo.mkdir()

        from hermes_cli import projects_db as _pdb

        @dataclass
        class FakeRow:
            slug: str
            name: str
            primary_path: str

            def __getitem__(self, key: str):
                return getattr(self, key)

        rows = [FakeRow(slug="other", name="Other", primary_path=str(unrelated_repo))]

        class FakeConn:
            def execute(self, *_a, **_kw):
                class _Cur:
                    def fetchall(_s):
                        return rows

                return _Cur()

            def close(self):
                pass

        @dataclass
        class FakeCM:
            rows: list

            def __enter__(self):
                return FakeConn()

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(_pdb, "connect_closing", lambda: FakeCM(rows=rows))

        task = FakeTask(
            title="#339 — Pattern 42 fix",
            body="Fix src/agents/architect_agent.py — this is for Chiron",
        )
        # Body says "Chiron" but no project with that name registered →
        # helper returns None (no false positive onto Other repo).
        assert kb._maybe_promote_scratch_to_worktree(task) is None

    def test_no_promotion_when_projects_db_lookup_fails(self, monkeypatch):
        """If projects_db is unreachable, helper must fail closed (None)."""
        from hermes_cli import projects_db as _pdb

        def boom():
            raise RuntimeError("simulated DB outage")

        monkeypatch.setattr(_pdb, "connect_closing", boom)

        task = FakeTask(
            title="fix",
            body="Fix src/agents/foo.py",
        )
        assert kb._maybe_promote_scratch_to_worktree(task) is None