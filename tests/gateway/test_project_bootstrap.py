"""Tests for gateway/project_bootstrap.py and its wiring into the first turn.

These tests cover both:
  - The pure loader (is_project_root, build_project_context, ProjectContext.render)
  - The wiring in agent/turn_context.build_turn_context that injects a
    ``system_reminder`` row on the first turn of any session whose cwd is a
    project root, idempotent per session_id.

Structural reference: ``tests/gateway/test_topic_routing.py``.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.project_bootstrap import (
    ProjectContext,
    build_project_context,
    is_project_root,
)


# ── is_project_root ──────────────────────────────────────────────────────────


class TestIsProjectRoot:
    """Tests for the project-root detection rules."""

    def test_recognizes_standard_projects_with_readme(self, tmp_path: Path) -> None:
        """A directory with README.md under a projects_root is a project root."""
        project = tmp_path / "my-proj"
        project.mkdir()
        (project / "README.md").write_text("# My Project\n")
        projects_root = tmp_path
        assert is_project_root(project, projects_root) is True

    def test_recognizes_marker_files_without_parent_match(self, tmp_path: Path) -> None:
        """Loosened rule: a directory with a marker file is a project root
        regardless of its parent (covers desktop sessions whose cwd is a
        project root identified by markers alone)."""
        project = tmp_path / "deep" / "nested" / "my-proj"
        project.mkdir(parents=True)
        (project / "pyproject.toml").write_text("[project]\n")
        # Parent is NOT the projects_root — but the marker check still wins.
        assert is_project_root(project, tmp_path / "different-root") is True

    def test_rejects_empty_directory(self, tmp_path: Path) -> None:
        """An empty directory is not a project root."""
        project = tmp_path / "empty"
        project.mkdir()
        assert is_project_root(project, tmp_path) is False

    def test_rejects_directory_without_markers(self, tmp_path: Path) -> None:
        """A directory with neither marker files nor .hermes/code under a
        projects_root is not a project root."""
        project = tmp_path / "noise"
        project.mkdir()
        (project / "random.txt").write_text("x")
        assert is_project_root(project, tmp_path) is False

    def test_rejects_directory_with_hermes_subdir_but_wrong_parent(
        self, tmp_path: Path
    ) -> None:
        """Heuristic-only fallback (.hermes/, code/) requires the parent
        match — a stray .hermes/ in an unrelated directory must not trigger
        bootstrap."""
        project = tmp_path / "somewhere" / "else" / "noisy"
        project.mkdir(parents=True)
        (project / ".hermes").mkdir()
        projects_root = tmp_path / "projects"
        projects_root.mkdir()
        assert is_project_root(project, projects_root) is False

    def test_accepts_nonexistent_path(self, tmp_path: Path) -> None:
        """A non-existent path returns False (never raises)."""
        missing = tmp_path / "does-not-exist"
        assert is_project_root(missing, tmp_path) is False


# ── build_project_context ────────────────────────────────────────────────────


class TestBuildProjectContext:
    """Tests for the ProjectContext loader and renderer."""

    def test_returns_bootstrap_envelope(self, tmp_path: Path) -> None:
        """The rendered output starts with the <project_bootstrap cwd=...> tag."""
        project = tmp_path / "with-readme"
        project.mkdir()
        (project / "README.md").write_text(
            "# Test Project\n\nA short readme for the test.\n"
        )
        # Build via the module under test; pass tmp_path as projects_root
        # so the parent rule (loosened) does not block marker-file match.
        ctx = build_project_context(cwd=str(project), projects_root=str(tmp_path))
        rendered = ctx.render()
        assert rendered.startswith(f"<project_bootstrap cwd={str(project)!r}>")
        assert "</project_bootstrap>" in rendered
        assert "Test Project" in rendered  # README excerpt visible

    def test_render_is_deterministic_given_fixed_cwd(self, tmp_path: Path) -> None:
        """Re-rendering the same project returns a stable string (modulo
        git log which may shift between calls)."""
        project = tmp_path / "det"
        project.mkdir()
        (project / "README.md").write_text("# Det\n")
        ctx1 = build_project_context(cwd=str(project), projects_root=str(tmp_path))
        ctx2 = build_project_context(cwd=str(project), projects_root=str(tmp_path))
        # README and directory listing sections are deterministic.
        # We do not compare git_log because it depends on subprocess state.
        r1 = ctx1.render().split("## Recent git activity")[0]
        r2 = ctx2.render().split("## Recent git activity")[0]
        assert r1 == r2

    def test_render_caps_output_at_8kb(self, tmp_path: Path) -> None:
        """A ProjectContext with enough content to exceed 8KB must be
        truncated to the budget with a [truncated] marker."""
        # Build a ProjectContext manually so we can inflate the directory
        # listing past the 8KB budget (the file-based loader caps the
        # README excerpt at _README_MAX_LINES = 100, which alone stays
        # well under the budget).
        ctx = ProjectContext(
            cwd=str(tmp_path / "fat"),
            name="fat",
            has_git=False,
            directory_listing=[f"  {i:04d}/subdir-{i:04d}/" for i in range(2000)],
            readme_excerpt="\n".join(f"line {i}" for i in range(500)),
            readme_path="README.md",
            git_log=[f"commit {i}: stuff" for i in range(500)],
        )
        rendered = ctx.render()
        # _MAX_REMINDER_BYTES is 8192; rendered length must stay under or at it.
        assert len(rendered.encode("utf-8")) <= 8192
        assert "[truncated]" in rendered
        # The closing tag is appended after truncation so the agent can
        # still parse the envelope.
        assert "</project_bootstrap>" in rendered

    def test_render_succeeds_for_minimal_project(self, tmp_path: Path) -> None:
        """A bare project (no README, no .hermes, no .git) still renders
        without raising."""
        project = tmp_path / "bare"
        project.mkdir()
        ctx = build_project_context(cwd=str(project), projects_root=str(tmp_path))
        rendered = ctx.render()
        # Must always contain both the opening and closing tags, even with
        # no project-specific content.
        assert "<project_bootstrap" in rendered
        assert "</project_bootstrap>" in rendered
        # And the warning is NOT present for a successful read.
        assert "Bootstrap warning" not in rendered


# ── Wiring: build_turn_context injects system_reminder ───────────────────────


def _make_minimal_agent(
    session_id: str,
    cwd: str,
    session_db,
) -> MagicMock:
    """Build a minimal AIAgent-shaped MagicMock for build_turn_context.

    Only the attributes build_turn_context reads during the bootstrap
    branch are set; everything else is a no-op MagicMock.
    """
    agent = MagicMock()
    agent.session_id = session_id
    agent._session_db = session_db
    return agent


def _seed_session_row(session_db, session_id: str, cwd: str) -> None:
    """Insert a session row into the SessionDB's underlying sqlite."""
    import time as _time

    conn = session_db._conn  # type: ignore[attr-defined]
    # Use OR IGNORE so this is idempotent across test re-runs in the same
    # process (xdist workers may share a tmp dir by accident).
    conn.execute(
        "INSERT OR IGNORE INTO sessions (id, source, started_at, cwd) "
        "VALUES (?, ?, ?, ?)",
        (session_id, "test", _time.time(), cwd),
    )
    # Always UPDATE cwd — tests with the same session_id but a new cwd
    # fixture (e.g. parametrized variants) need cwd refreshed.
    conn.execute("UPDATE sessions SET cwd = ? WHERE id = ?", (cwd, session_id))
    conn.commit()


@pytest.fixture
def in_memory_session_db():
    """Build a SessionDB pointed at a temp file, sharing the schema of the
    production messages table. Used by the wiring tests below."""
    from hermes_state import SessionDB

    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"test_bootstrap_{os.getpid()}.db"
    db = SessionDB(db_path=Path(tmp))
    yield db
    try:
        db.close()
    except Exception:
        pass
    try:
        os.remove(tmp)
    except OSError:
        pass


class TestWiringInjectsBootstrap:
    """Wiring tests for the system_reminder injection in build_turn_context.

    These tests call ``_maybe_inject_project_bootstrap`` directly rather
    than going through the full ``build_turn_context`` prologue, which
    would require a fully-shaped AIAgent and many other attributes the
    bootstrap branch never reads. The bootstrap is a focused, idempotent
    side effect on the in-memory message list; that's exactly what we
    exercise here.
    """

    def test_first_turn_with_project_cwd_injects_system_reminder(
        self, tmp_path: Path, in_memory_session_db
    ) -> None:
        """First turn of a session whose cwd is a project root produces a
        system_reminder row in the message list."""
        from agent.turn_context import _maybe_inject_project_bootstrap

        project = tmp_path / "wired"
        project.mkdir()
        (project / "README.md").write_text("# Wired\n")
        session_id = "sess_wired_1"
        _seed_session_row(in_memory_session_db, session_id, str(project))

        agent = _make_minimal_agent(session_id, str(project), in_memory_session_db)
        messages: list = []

        _maybe_inject_project_bootstrap(agent, messages)

        # messages should contain [bootstrap]; bootstrap is appended at the
        # end (caller appends user_msg separately on the actual prologue path).
        assert len(messages) == 1
        bootstrap = messages[0]
        assert bootstrap["role"] == "system"
        assert bootstrap["display_kind"] == "system_reminder"
        assert bootstrap["display_metadata"]["source"] == "project_bootstrap"
        assert bootstrap["display_metadata"]["cwd"] == str(project)
        assert "<project_bootstrap" in bootstrap["content"]

    def test_second_turn_of_same_session_does_not_reinject(
        self, tmp_path: Path, in_memory_session_db
    ) -> None:
        """A second run_conversation on the same session_id skips the
        bootstrap because a system_reminder row already exists."""
        from agent.turn_context import _maybe_inject_project_bootstrap

        project = tmp_path / "wired2"
        project.mkdir()
        (project / "README.md").write_text("# Wired2\n")
        session_id = "sess_wired_2"
        _seed_session_row(in_memory_session_db, session_id, str(project))

        # Pre-seed a system_reminder row to simulate an already-bootstrapped
        # session (e.g. resume-into-bootstrapped-session).
        in_memory_session_db._conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, "
            "active, display_kind) VALUES (?, 'system', '<prior bootstrap>', "
            "1.0, 1, 'system_reminder')",
            (session_id,),
        )
        in_memory_session_db._conn.commit()

        agent = _make_minimal_agent(session_id, str(project), in_memory_session_db)
        messages: list = []

        _maybe_inject_project_bootstrap(agent, messages)

        # No bootstrap row appended.
        assert messages == []

    def test_skipped_when_cwd_is_not_project_root(
        self, tmp_path: Path, in_memory_session_db
    ) -> None:
        """A session whose cwd is /tmp/random does NOT inject a bootstrap."""
        from agent.turn_context import _maybe_inject_project_bootstrap

        session_id = "sess_random"
        # /tmp itself is not a project root (no markers, no .hermes).
        _seed_session_row(in_memory_session_db, session_id, str(tmp_path))

        agent = _make_minimal_agent(session_id, str(tmp_path), in_memory_session_db)
        messages: list = []

        _maybe_inject_project_bootstrap(agent, messages)

        assert messages == []

    def test_skipped_when_session_has_no_cwd(
        self, tmp_path: Path, in_memory_session_db
    ) -> None:
        """A session whose cwd is empty/missing does NOT inject a bootstrap."""
        from agent.turn_context import _maybe_inject_project_bootstrap

        session_id = "sess_nocwd"
        _seed_session_row(in_memory_session_db, session_id, "")  # empty cwd

        agent = _make_minimal_agent(session_id, "", in_memory_session_db)
        messages: list = []

        _maybe_inject_project_bootstrap(agent, messages)

        assert messages == []

    def test_skipped_when_session_db_missing(self, tmp_path: Path) -> None:
        """If the agent has no _session_db attribute, bootstrap is skipped
        (no DB access means no idempotency check; we choose fail-safe skip)."""
        from agent.turn_context import _maybe_inject_project_bootstrap

        project = tmp_path / "no_db"
        project.mkdir()
        (project / "README.md").write_text("# NoDB\n")

        agent = _make_minimal_agent("sess_nodb", str(project), None)
        messages: list = []

        _maybe_inject_project_bootstrap(agent, messages)

        assert messages == []

    def test_idempotent_across_two_invocations(
        self, tmp_path: Path, in_memory_session_db
    ) -> None:
        """Two back-to-back calls on the same fresh session produce exactly
        ONE bootstrap row. The second call sees the first's persisted row
        and short-circuits."""
        from agent.turn_context import _maybe_inject_project_bootstrap

        project = tmp_path / "twice"
        project.mkdir()
        (project / "README.md").write_text("# Twice\n")
        session_id = "sess_twice"
        _seed_session_row(in_memory_session_db, session_id, str(project))

        agent = _make_minimal_agent(session_id, str(project), in_memory_session_db)
        messages: list = []

        _maybe_inject_project_bootstrap(agent, messages)
        first_count = len(messages)
        assert first_count == 1

        # Simulate the second invocation: caller would normally persist the
        # first bootstrap and then call again on the same session. Persist
        # it manually and re-call.
        bootstrap = messages[0]
        in_memory_session_db._conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, "
            "active, display_kind, display_metadata) VALUES (?, ?, ?, ?, 1, "
            "?, ?)",
            (
                session_id,
                bootstrap["role"],
                bootstrap["content"],
                1.0,
                bootstrap["display_kind"],
                None,
            ),
        )
        in_memory_session_db._conn.commit()
        messages.clear()

        _maybe_inject_project_bootstrap(agent, messages)
        assert messages == []


# ── SessionDB.has_display_kind_message ───────────────────────────────────────


class TestHasDisplayKindMessage:
    """Tests for the SessionDB helper that powers the idempotency check."""

    def test_returns_false_when_no_message_exists(self, in_memory_session_db) -> None:
        assert (
            in_memory_session_db.has_display_kind_message("sess_empty", "system_reminder")
            is False
        )

    def test_returns_true_when_matching_active_row_exists(
        self, tmp_path: Path, in_memory_session_db
    ) -> None:
        sid = "sess_match"
        # Seed a parent session row so the messages FK is satisfied.
        _seed_session_row(in_memory_session_db, sid, str(tmp_path))
        in_memory_session_db._conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, "
            "active, display_kind) VALUES (?, 'system', 'x', 1.0, 1, 'system_reminder')",
            (sid,),
        )
        in_memory_session_db._conn.commit()
        assert in_memory_session_db.has_display_kind_message(sid, "system_reminder") is True

    def test_returns_false_when_only_compacted_match_exists(
        self, tmp_path: Path, in_memory_session_db
    ) -> None:
        """A compacted (active=0) row must NOT suppress a fresh bootstrap
        after an intentional reset."""
        sid = "sess_compacted"
        _seed_session_row(in_memory_session_db, sid, str(tmp_path))
        in_memory_session_db._conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, "
            "active, display_kind) VALUES (?, 'system', 'x', 1.0, 0, 'system_reminder')",
            (sid,),
        )
        in_memory_session_db._conn.commit()
        assert (
            in_memory_session_db.has_display_kind_message(sid, "system_reminder")
            is False
        )

    def test_returns_false_for_empty_args(self, in_memory_session_db) -> None:
        assert in_memory_session_db.has_display_kind_message("", "system_reminder") is False
        assert in_memory_session_db.has_display_kind_message("sess_x", "") is False
