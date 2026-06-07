"""Tests for BoardDispatchHealth, board_dispatch_health(), and format_stuck().

These cover the per-board dispatch health probe used by the gateway and
CLI daemon stuck-detectors to identify WHICH boards have spawnable work
backing up behind zero spawns.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def fake_profile(monkeypatch):
    """Make profile_exists return True for any name."""
    monkeypatch.setattr(
        "hermes_cli.kanban_db.profile_exists",
        lambda name: True,
        raising=False,
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists",
        lambda name: True,
        raising=False,
    )


def _set_running(conn, task_id: str) -> None:
    """Directly flip a task into running status for test purposes."""
    conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (task_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# BoardDispatchHealth dataclass
# ---------------------------------------------------------------------------


class TestBoardDispatchHealthDataclass:
    def test_str_format(self):
        h = kb.BoardDispatchHealth(slug="my-board", ready=3, running=1, spawnable=True)
        assert str(h) == "my-board (3 ready, 1 running)"

    def test_str_format_with_terminal_lanes(self):
        h = kb.BoardDispatchHealth(
            slug="mixed", ready=5, ready_nonspawnable=2, running=1, spawnable=True,
        )
        assert str(h) == "mixed (5 ready, 1 running) (2 terminal lanes)"

    def test_str_format_zero_counts(self):
        h = kb.BoardDispatchHealth(slug="empty-board")
        assert str(h) == "empty-board (0 ready, 0 running)"

    def test_defaults(self):
        h = kb.BoardDispatchHealth(slug="test")
        assert h.ready == 0
        assert h.ready_nonspawnable == 0
        assert h.running == 0
        assert h.spawnable is False


# ---------------------------------------------------------------------------
# format_stuck (staticmethod)
# ---------------------------------------------------------------------------


class TestFormatStuck:
    def test_empty_list(self):
        assert kb.BoardDispatchHealth.format_stuck([]) == "unknown"

    def test_all_spawnable(self):
        boards = [
            kb.BoardDispatchHealth(slug="board-a", ready=3, running=1, spawnable=True),
            kb.BoardDispatchHealth(slug="board-b", ready=1, running=0, spawnable=True),
        ]
        result = kb.BoardDispatchHealth.format_stuck(boards)
        assert result == "board-a (3 ready, 1 running), board-b (1 ready, 0 running)"

    def test_mixed_spawnable(self):
        """Only spawnable boards appear in output."""
        boards = [
            kb.BoardDispatchHealth(slug="idle-board", spawnable=False),
            kb.BoardDispatchHealth(slug="stuck-board", ready=5, running=0, spawnable=True),
        ]
        result = kb.BoardDispatchHealth.format_stuck(boards)
        assert result == "stuck-board (5 ready, 0 running)"
        assert "idle-board" not in result

    def test_none_spawnable(self):
        boards = [
            kb.BoardDispatchHealth(slug="a", spawnable=False),
            kb.BoardDispatchHealth(slug="b", spawnable=False),
        ]
        assert kb.BoardDispatchHealth.format_stuck(boards) == "unknown"

    def test_spawnable_with_terminal_lanes_annotation(self):
        """Terminal lane count is surfaced in the stuck summary."""
        boards = [
            kb.BoardDispatchHealth(
                slug="mixed", ready=5, ready_nonspawnable=2,
                running=1, spawnable=True,
            ),
        ]
        result = kb.BoardDispatchHealth.format_stuck(boards)
        assert "2 terminal lanes" in result


# ---------------------------------------------------------------------------
# board_dispatch_health (integration with DB)
# ---------------------------------------------------------------------------


class TestBoardDispatchHealthProbe:
    def test_empty_board(self, kanban_home, fake_profile):
        """Default board with no tasks returns ready=0, running=0, spawnable=False."""
        results = kb.board_dispatch_health()
        assert len(results) >= 1
        default = [r for r in results if r.slug == "default"][0]
        assert default.ready == 0
        assert default.ready_nonspawnable == 0
        assert default.running == 0
        assert default.spawnable is False

    def test_counts_ready_tasks(self, kanban_home, fake_profile):
        """Ready + assigned + unclaimed tasks are counted."""
        with kb.connect() as conn:
            # No parents → status defaults to 'ready'
            kb.create_task(conn, title="t1", assignee="worker")
            kb.create_task(conn, title="t2", assignee="worker")
        results = kb.board_dispatch_health()
        default = [r for r in results if r.slug == "default"][0]
        assert default.ready == 2

    def test_counts_running_tasks(self, kanban_home, fake_profile):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="t1", assignee="worker")
            _set_running(conn, tid)
        results = kb.board_dispatch_health()
        default = [r for r in results if r.slug == "default"][0]
        assert default.running == 1

    def test_spawnable_true_for_real_profile(self, kanban_home, fake_profile):
        with kb.connect() as conn:
            kb.create_task(conn, title="t1", assignee="worker")
        results = kb.board_dispatch_health()
        default = [r for r in results if r.slug == "default"][0]
        assert default.spawnable is True

    def test_spawnable_false_for_unassigned(self, kanban_home, fake_profile):
        """Unassigned tasks don't count as spawnable."""
        with kb.connect() as conn:
            kb.create_task(conn, title="t1")
        results = kb.board_dispatch_health()
        default = [r for r in results if r.slug == "default"][0]
        assert default.spawnable is False

    def test_nonspawnable_counted_for_terminal_lanes(self, kanban_home, monkeypatch):
        """Tasks assigned to non-existent profiles (terminal lanes) are
        counted in ready_nonspawnable but don't make spawnable=True."""
        monkeypatch.setattr(
            "hermes_cli.kanban_db.profile_exists",
            lambda name: False,
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.profile_exists",
            lambda name: False,
            raising=False,
        )
        with kb.connect() as conn:
            kb.create_task(conn, title="t1", assignee="orion-cc")
        results = kb.board_dispatch_health()
        default = [r for r in results if r.slug == "default"][0]
        assert default.ready == 1
        assert default.ready_nonspawnable == 1
        assert default.spawnable is False

    def test_nonspawnable_counts_tasks_not_assignees(self, kanban_home, monkeypatch):
        """Multiple tasks to the same terminal-lane assignee count all tasks,
        not just distinct assignees."""
        monkeypatch.setattr(
            "hermes_cli.kanban_db.profile_exists",
            lambda name: False,
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.profile_exists",
            lambda name: False,
            raising=False,
        )
        with kb.connect() as conn:
            kb.create_task(conn, title="t1", assignee="orion-cc")
            kb.create_task(conn, title="t2", assignee="orion-cc")
            kb.create_task(conn, title="t3", assignee="orion-cc")
        results = kb.board_dispatch_health()
        default = [r for r in results if r.slug == "default"][0]
        assert default.ready == 3
        assert default.ready_nonspawnable == 3  # 3 tasks, not 1 assignee
        assert default.spawnable is False

    def test_mixed_spawnable_and_nonspawnable(self, kanban_home, monkeypatch):
        """Both spawnable and non-spawnable tasks on the same board."""
        call_count = {"n": 0}
        def _exists(name):
            call_count["n"] += 1
            return name == "worker"
        monkeypatch.setattr(
            "hermes_cli.kanban_db.profile_exists",
            _exists,
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.profile_exists",
            _exists,
            raising=False,
        )
        with kb.connect() as conn:
            kb.create_task(conn, title="t1", assignee="worker")
            kb.create_task(conn, title="t2", assignee="orion-cc")
            kb.create_task(conn, title="t3", assignee="orion-research")
        results = kb.board_dispatch_health()
        default = [r for r in results if r.slug == "default"][0]
        assert default.ready == 3
        assert default.ready_nonspawnable == 2
        assert default.spawnable is True

    def test_multiple_boards(self, kanban_home, fake_profile):
        """Health is reported per-board across all active boards."""
        kb.create_board("project-x")
        # Add a task to the default board
        with kb.connect() as conn:
            kb.create_task(conn, title="default-task", assignee="worker")
        # Add a running task to project-x
        with kb.connect(board="project-x") as conn:
            tid = kb.create_task(conn, title="x-task", assignee="worker")
            _set_running(conn, tid)
        results = kb.board_dispatch_health()
        slugs = {r.slug for r in results}
        assert "default" in slugs
        assert "project-x" in slugs
        # Verify per-board counts
        default = [r for r in results if r.slug == "default"][0]
        project_x = [r for r in results if r.slug == "project-x"][0]
        assert default.ready == 1
        assert project_x.running == 1

    def test_graceful_on_corrupt_board(self, kanban_home, fake_profile):
        """A board with a corrupt/missing DB should not crash the probe."""
        kb.create_board("broken")
        # Delete the DB to simulate corruption
        db_path = kb.kanban_db_path("broken")
        if db_path.exists():
            db_path.unlink()
        # Should still return an entry for the broken board (with defaults)
        results = kb.board_dispatch_health()
        broken = [r for r in results if r.slug == "broken"]
        assert len(broken) == 1
        assert broken[0].ready == 0
        assert broken[0].ready_nonspawnable == 0
        assert broken[0].running == 0
        assert broken[0].spawnable is False

    def test_list_boards_failure_falls_back_to_default(self, kanban_home, fake_profile):
        """If list_boards raises, falls back to default board only."""
        with patch.object(kb, "list_boards", side_effect=RuntimeError("boom")):
            results = kb.board_dispatch_health()
        assert len(results) == 1
        assert results[0].slug == "default"

    def test_empty_string_assignee_does_not_crash(self, kanban_home, fake_profile):
        """Tasks with assignee='' should not crash the probe via
        profile_exists('') ValueError."""
        with kb.connect() as conn:
            # Create a task, then manually set assignee to empty string
            tid = kb.create_task(conn, title="t1", assignee="worker")
            conn.execute("UPDATE tasks SET assignee = '' WHERE id = ?", (tid,))
            conn.commit()
        # Should not raise — empty-string assignees are excluded from counts
        results = kb.board_dispatch_health()
        default = [r for r in results if r.slug == "default"][0]
        assert default.ready == 0  # excluded by assignee != ''
        assert default.ready_nonspawnable == 0

    def test_empty_assignee_does_not_silence_real_spawnable(self, kanban_home, fake_profile):
        """A board with one real task AND one empty-assignee task should
        still report the real task's health — not silently drop everything
        because has_spawnable_ready crashes on profile_exists('')."""
        with kb.connect() as conn:
            kb.create_task(conn, title="good", assignee="worker")
            tid2 = kb.create_task(conn, title="bad", assignee="worker")
            conn.execute("UPDATE tasks SET assignee = '' WHERE id = ?", (tid2,))
            conn.commit()
        results = kb.board_dispatch_health()
        default = [r for r in results if r.slug == "default"][0]
        assert default.ready == 1  # the good task counted
        assert default.spawnable is True

    def test_claim_locked_tasks_excluded_from_ready(self, kanban_home, fake_profile):
        """Tasks with a claim_lock set should not be counted as ready."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="locked", assignee="worker")
            conn.execute(
                "UPDATE tasks SET claim_lock = ? WHERE id = ?",
                ("session-abc", tid),
            )
            conn.commit()
        results = kb.board_dispatch_health()
        default = [r for r in results if r.slug == "default"][0]
        assert default.ready == 0
        assert default.running == 0

    def test_profile_import_failure_skips_nonspawnable(self, kanban_home, monkeypatch):
        """When hermes_cli.profiles can't be imported, ready_nonspawnable
        stays 0 — the probe can't distinguish real profiles from terminal
        lanes, so it reports conservatively."""
        # Seed a task assigned to a non-profile lane.
        with kb.connect() as conn:
            kb.create_task(conn, title="t1", assignee="orion-cc")
        # Make importing hermes_cli.profiles.profile_exists raise ImportError
        # so board_dispatch_health's `from ... import` fails and
        # _profile_exists stays None.
        monkeypatch.setattr(
            "hermes_cli.profiles.profile_exists",
            None,
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.kanban_db.profile_exists",
            None,
            raising=False,
        )
        results = kb.board_dispatch_health()
        default = [r for r in results if r.slug == "default"][0]
        assert default.ready == 1
        assert default.ready_nonspawnable == 0  # can't introspect → 0
