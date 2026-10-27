"""Tests for the kanban size gate and decomposer size contract.

Verifies:
1. The deterministic estimator correctly measures card complexity
2. The size gate routes oversized cards to triage at creation time
3. The decomposer refuses oversized children with a typed error
4. The config key kanban.size_gate.threshold is respected
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_size_gate import (
    DEFAULT_SIZE_GATE_THRESHOLD,
    OversizedChildError,
    SizeEstimate,
    estimate_card_size,
    load_size_gate_threshold,
    should_route_to_triage,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


class TestSizeEstimator:
    """Test the deterministic size estimator."""

    def test_empty_card(self):
        """An empty card has zero size."""
        estimate = estimate_card_size("", "")
        assert estimate.score == 0
        assert estimate.acceptance_criteria_count == 0
        assert estimate.file_count == 0
        assert estimate.deliverable_count == 0

    def test_minimal_card(self):
        """A minimal card with just a title has zero size."""
        estimate = estimate_card_size("Fix the bug", None)
        assert estimate.score == 0

    def test_card_with_acceptance_criteria(self):
        """Count acceptance criteria items in a numbered list."""
        body = """
## Acceptance Criteria
1. First criterion
2. Second criterion
3. Third criterion
4. Fourth criterion
5. Fifth criterion
"""
        estimate = estimate_card_size("Task", body)
        assert estimate.acceptance_criteria_count == 5
        # The "## Acceptance Criteria" header is also counted as a deliverable
        # Score = 5 * 3 (AC weight) + 1 * 1 (deliverable weight) = 16
        assert estimate.score == 16

    def test_card_with_files(self):
        """Count distinct files named in the body."""
        body = """
Files to modify:
- src/auth/oauth2.py
- src/auth/verification.py
- src/users/profile.py
- src/admin/dashboard.py
"""
        estimate = estimate_card_size("Task", body)
        assert estimate.file_count == 4
        # Score = 4 * 2 (weight) = 8
        assert estimate.score == 8

    def test_card_with_deliverables(self):
        """Count distinct deliverables (## headers)."""
        body = """
## Authentication
Implement OAuth2 flow

## User Management
Build profile system

## Admin Dashboard
Create admin interface
"""
        estimate = estimate_card_size("Task", body)
        assert estimate.deliverable_count == 3
        # Score = 3 * 1 (weight) = 3
        assert estimate.score == 3

    def test_complex_card(self):
        """A complex card with AC, files, and deliverables."""
        body = """
## Acceptance Criteria
1. First criterion
2. Second criterion
3. Third criterion

## Files to Create
- src/auth/oauth2.py
- src/auth/verification.py
- src/users/profile.py

## Documentation
Write API docs
"""
        estimate = estimate_card_size("Task", body)
        # 3 AC * 3 = 9
        # 3 files * 2 = 6
        # 3 deliverables * 1 = 3
        # Total = 18
        assert estimate.acceptance_criteria_count == 3
        assert estimate.file_count == 3
        assert estimate.deliverable_count == 3
        assert estimate.score == 18

    def test_is_oversized(self):
        """Test the is_oversized method."""
        estimate = SizeEstimate(score=35, acceptance_criteria_count=10, file_count=2, deliverable_count=1)
        assert estimate.is_oversized(30) is True
        assert estimate.is_oversized(40) is False
        assert estimate.is_oversized(35) is False  # Equal is not oversized


class TestSizeGateConfig:
    """Test the size gate configuration."""

    def test_default_threshold(self):
        """Default threshold is 30."""
        assert DEFAULT_SIZE_GATE_THRESHOLD == 30

    def test_load_default_threshold(self, kanban_home):
        """Load the default threshold when no config is set."""
        threshold = load_size_gate_threshold()
        assert threshold == 30

    def test_load_custom_threshold(self, kanban_home, monkeypatch):
        """Load a custom threshold from config."""
        # Create a config.yaml with custom threshold
        config_path = kanban_home / "config.yaml"
        config_path.write_text("""
_config_version: 100
kanban:
  size_gate:
    threshold: 50
""")
        # The loader will read the new config file
        threshold = load_size_gate_threshold()
        assert threshold == 50


class TestSizeGateRouting:
    """Test the size gate routing logic."""

    def test_small_card_does_not_route_to_triage(self, kanban_home):
        """A small card does not route to triage."""
        should_route, estimate = should_route_to_triage("Small task", "Just fix the bug")
        assert should_route is False
        assert estimate.score < DEFAULT_SIZE_GATE_THRESHOLD

    def test_oversized_card_routes_to_triage(self, kanban_home):
        """An oversized card routes to triage."""
        body = """
## Acceptance Criteria
1. First
2. Second
3. Third
4. Fourth
5. Fifth
6. Sixth
7. Seventh
8. Eighth
9. Ninth
10. Tenth
11. Eleventh
12. Twelfth

## Files
- src/a.py
- src/b.py
- src/c.py
- src/d.py
- src/e.py
- src/f.py
- src/g.py
- src/h.py
"""
        should_route, estimate = should_route_to_triage("Large task", body)
        assert should_route is True
        assert estimate.score > DEFAULT_SIZE_GATE_THRESHOLD


class TestSizeGateInCreateTask:
    """Test the size gate integration with create_task."""

    def test_small_card_goes_to_ready(self, kanban_home):
        """A small card with explicit assignee goes to ready."""
        with kbc.connect() as conn:
            task_id = kb.create_task(
                conn,
                title="Small task",
                body="Just fix the bug",
                assignee="junior-dev",
                created_by="test",
                triage=False,
            )
            task = kb.get_task(conn, task_id)

        assert task.status == "ready"

    def test_oversized_card_routes_to_triage(self, kanban_home):
        """An oversized card with explicit assignee routes to triage."""
        body = """
## Acceptance Criteria
1. First
2. Second
3. Third
4. Fourth
5. Fifth
6. Sixth
7. Seventh
8. Eighth
9. Ninth
10. Tenth
11. Eleventh
12. Twelfth

## Files
- src/a.py
- src/b.py
- src/c.py
- src/d.py
- src/e.py
- src/f.py
- src/g.py
- src/h.py
"""
        with kbc.connect() as conn:
            task_id = kb.create_task(
                conn,
                title="Large task",
                body=body,
                assignee="junior-dev",
                created_by="test",
                triage=False,
            )
            task = kb.get_task(conn, task_id)

            # Check that size_estimate event was recorded
            events = conn.execute(
                "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY created_at",
                (task_id,),
            ).fetchall()
            event_kinds = [e["kind"] for e in events]

        assert task.status == "triage"
        assert "size_estimate" in event_kinds

    def test_explicit_triage_bypasses_size_gate(self, kanban_home):
        """A card with explicit triage=True goes to triage regardless of size."""
        with kbc.connect() as conn:
            task_id = kb.create_task(
                conn,
                title="Small task",
                body="Just fix the bug",
                assignee="junior-dev",
                created_by="test",
                triage=True,
            )
            task = kb.get_task(conn, task_id)

        assert task.status == "triage"


class TestDecomposerSizeContract:
    """Test the decomposer's size contract for children."""

    def test_decomposer_refuses_oversized_child(self, kanban_home):
        """The decomposer refuses to emit an oversized child."""
        from hermes_cli.kanban_db_graph import decompose_triage_task

        # Create a triage task
        with kbc.connect() as conn:
            task_id = kb.create_task(
                conn,
                title="Parent task",
                body="Some work",
                assignee="orchestrator",
                created_by="test",
                triage=True,
            )

        # Try to decompose with an oversized child
        oversized_child_body = """
## Acceptance Criteria
1. First
2. Second
3. Third
4. Fourth
5. Fifth
6. Sixth
7. Seventh
8. Eighth
9. Ninth
10. Tenth
11. Eleventh
12. Twelfth

## Files
- src/a.py
- src/b.py
- src/c.py
- src/d.py
- src/e.py
- src/f.py
- src/g.py
- src/h.py
"""
        children = [
            {"title": "Oversized child", "body": oversized_child_body, "assignee": "junior-dev"},
        ]

        with kbc.connect() as conn:
            with pytest.raises(OversizedChildError) as exc_info:
                decompose_triage_task(
                    conn,
                    task_id,
                    root_assignee="orchestrator",
                    children=children,
                    author="decomposer",
                )

        assert exc_info.value.child_index == 0
        assert exc_info.value.child_title == "Oversized child"
        assert exc_info.value.estimate.score > DEFAULT_SIZE_GATE_THRESHOLD
        assert exc_info.value.threshold == DEFAULT_SIZE_GATE_THRESHOLD

    def test_decomposer_accepts_small_children(self, kanban_home):
        """The decomposer accepts small children."""
        from hermes_cli.kanban_db_graph import decompose_triage_task

        # Create a triage task
        with kbc.connect() as conn:
            task_id = kb.create_task(
                conn,
                title="Parent task",
                body="Some work",
                assignee="orchestrator",
                created_by="test",
                triage=True,
            )

        children = [
            {"title": "Small child 1", "body": "Fix the bug", "assignee": "junior-dev"},
            {"title": "Small child 2", "body": "Add tests", "assignee": "junior-dev"},
        ]

        with kbc.connect() as conn:
            child_ids = decompose_triage_task(
                conn,
                task_id,
                root_assignee="orchestrator",
                children=children,
                author="decomposer",
            )

        assert child_ids is not None
        assert len(child_ids) == 2

    def test_decomposer_refuses_mixed_sizes(self, kanban_home):
        """The decomposer refuses if any child is oversized."""
        from hermes_cli.kanban_db_graph import decompose_triage_task

        # Create a triage task
        with kbc.connect() as conn:
            task_id = kb.create_task(
                conn,
                title="Parent task",
                body="Some work",
                assignee="orchestrator",
                created_by="test",
                triage=True,
            )

        oversized_child_body = """
## Acceptance Criteria
1. First
2. Second
3. Third
4. Fourth
5. Fifth
6. Sixth
7. Seventh
8. Eighth
9. Ninth
10. Tenth
11. Eleventh
12. Twelfth

## Files
- src/a.py
- src/b.py
- src/c.py
- src/d.py
- src/e.py
- src/f.py
- src/g.py
- src/h.py
"""
        children = [
            {"title": "Small child", "body": "Fix the bug", "assignee": "junior-dev"},
            {"title": "Oversized child", "body": oversized_child_body, "assignee": "junior-dev"},
        ]

        with kbc.connect() as conn:
            with pytest.raises(OversizedChildError) as exc_info:
                decompose_triage_task(
                    conn,
                    task_id,
                    root_assignee="orchestrator",
                    children=children,
                    author="decomposer",
                )

        # The oversized child is at index 1
        assert exc_info.value.child_index == 1
        assert exc_info.value.child_title == "Oversized child"
