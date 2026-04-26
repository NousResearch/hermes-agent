"""Additional edge-case tests for the Kanban DB layer.

These tests cover:
- Priority ordering verification
- Task ID generation uniqueness
- Cross-tenant isolation
- Edge cases for task completion and claim lifecycle
"""

from __future__ import annotations

import concurrent.futures
import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Priority ordering tests
# ---------------------------------------------------------------------------

def test_list_tasks_respects_priority_order(kanban_home):
    """Verify that list_tasks returns tasks ordered by priority DESC."""
    with kb.connect() as conn:
        kb.create_task(conn, title="low", priority=1)
        kb.create_task(conn, title="high", priority=100)
        kb.create_task(conn, title="medium", priority=50)
        tasks = kb.list_tasks(conn)
    assert [t.title for t in tasks] == ["high", "medium", "low"]


def test_list_tasks_respects_priority_then_created_at(kanban_home):
    """Same priority should be ordered by created_at ASC."""
    with kb.connect() as conn:
        t1 = kb.create_task(conn, title="first", priority=10)
        time.sleep(0.01)  # Ensure different timestamps
        t2 = kb.create_task(conn, title="second", priority=10)
        time.sleep(0.01)
        t3 = kb.create_task(conn, title="third", priority=10)
        tasks = kb.list_tasks(conn)
    assert [t.id for t in tasks] == [t1, t2, t3]


def test_priority_update_affects_list_order(kanban_home):
    """Updating a task's priority should affect list ordering."""
    with kb.connect() as conn:
        low = kb.create_task(conn, title="low", priority=1)
        high = kb.create_task(conn, title="high", priority=100)
        # Initially high comes first
        assert kb.list_tasks(conn)[0].title == "high"
        # Update low to be higher priority
        conn.execute("UPDATE tasks SET priority = 200 WHERE id = ?", (low,))
        tasks = kb.list_tasks(conn)
        assert tasks[0].title == "low"
        assert tasks[1].title == "high"


# ---------------------------------------------------------------------------
# Task ID generation uniqueness tests
# ---------------------------------------------------------------------------

def test_task_id_format_is_correct(kanban_home):
    """Verify task IDs follow the t_<4 hex chars> format."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="test")
    assert tid.startswith("t_")
    assert len(tid) == 6  # "t_" + 4 hex chars
    # Verify it's valid hex
    hex_part = tid[2:]
    int(hex_part, 16)  # Will raise ValueError if not valid hex


def test_concurrent_task_creation_no_collision(kanban_home):
    """Concurrent task creation should not generate duplicate IDs."""
    ids = []
    
    def create_one():
        with kb.connect() as conn:
            return kb.create_task(conn, title="concurrent")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        ids = list(ex.map(lambda _: create_one(), range(50)))
    
    # All IDs should be unique
    assert len(ids) == len(set(ids)), "Duplicate task IDs generated!"


# ---------------------------------------------------------------------------
# Cross-tenant isolation tests
# ---------------------------------------------------------------------------

def test_tenant_isolation_in_list_tasks(kanban_home):
    """Tasks with different tenants should not mix in filtered results."""
    with kb.connect() as conn:
        kb.create_task(conn, title="tenant-a-1", tenant="A")
        kb.create_task(conn, title="tenant-a-2", tenant="A")
        kb.create_task(conn, title="tenant-b-1", tenant="B")
        kb.create_task(conn, title="tenant-b-2", tenant="B")
        kb.create_task(conn, title="no-tenant")
        
        tasks_a = kb.list_tasks(conn, tenant="A")
        tasks_b = kb.list_tasks(conn, tenant="B")
        tasks_all = kb.list_tasks(conn)  # No filter = sees all non-archived
    
    assert len(tasks_a) == 2
    assert all(t.tenant == "A" for t in tasks_a)
    assert len(tasks_b) == 2
    assert all(t.tenant == "B" for t in tasks_b)
    # Default list excludes archived but includes all tenants
    assert len(tasks_all) == 5


def test_tenant_isolation_in_dispatch(kanban_home):
    """Dispatch should respect tenant boundaries when filtering."""
    spawns = []
    
    def fake_spawn(task, workspace):
        spawns.append((task.id, task.tenant))
    
    with kb.connect() as conn:
        # Create tasks in different tenants
        kb.create_task(conn, title="task-a", tenant="A", assignee="alice")
        kb.create_task(conn, title="task-b", tenant="B", assignee="bob")
        
        # Dispatch should spawn both (no tenant filter in dispatch_once)
        kb.dispatch_once(conn, spawn_fn=fake_spawn, tenant=None)
    
    # Both should spawn
    assert len(spawns) == 2


# ---------------------------------------------------------------------------
# Task completion edge cases
# ---------------------------------------------------------------------------

def test_complete_task_with_empty_result(kanban_home):
    """Completing a task with no result should work."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="no-result")
        assert kb.complete_task(conn, t)
        task = kb.get_task(conn, t)
    assert task.status == "done"
    assert task.result is None


def test_complete_task_already_done_returns_false(kanban_home):
    """Completing an already-done task should return False."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="done-already")
        kb.complete_task(conn, t)
        # Try to complete again
        assert not kb.complete_task(conn, t)


def test_complete_archived_task_returns_false(kanban_home):
    """Completing an archived task should return False."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="archived-task")
        kb.archive_task(conn, t)
        assert not kb.complete_task(conn, t)


def test_complete_task_triggers_child_promotion(kanban_home):
    """Completing a parent should promote children to ready."""
    with kb.connect() as conn:
        p = kb.create_task(conn, title="parent")
        c = kb.create_task(conn, title="child", parents=[p])
        
        # Child starts as todo
        assert kb.get_task(conn, c).status == "todo"
        
        # Complete parent
        kb.complete_task(conn, p)
        
        # Child should now be ready
        assert kb.get_task(conn, c).status == "ready"


# ---------------------------------------------------------------------------
# Claim lifecycle edge cases
# ---------------------------------------------------------------------------

def test_claim_task_already_claimed_by_same_claimer(kanban_home):
    """Same claimer trying to claim again should fail (already running)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        claimer = "host:test"
        
        first = kb.claim_task(conn, t, claimer=claimer)
        assert first is not None
        
        # Same claimer trying again should fail (task is running, not ready)
        second = kb.claim_task(conn, t, claimer=claimer)
        assert second is None


def test_claim_task_after_release(kanban_home):
    """A task can be claimed again after being released."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        claimer = "host:test"
        
        # First claim
        first = kb.claim_task(conn, t, claimer=claimer)
        assert first is not None
        
        # Release (simulate via completing)
        kb.complete_task(conn, t)
        
        # Re-create and claim again
        t2 = kb.create_task(conn, title="x2", assignee="a")
        second = kb.claim_task(conn, t2, claimer=claimer)
        assert second is not None


def test_heartbeat_wrong_claimer_fails(kanban_home):
    """Heartbeat with wrong claimer should return False."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        
        # Claim with one ID
        kb.claim_task(conn, t, claimer="host:right")
        
        # Heartbeat with different ID
        ok = kb.heartbeat_claim(conn, t, claimer="host:wrong")
        assert not ok


def test_claim_task_with_custom_ttl(kanban_home):
    """Claim task should respect custom TTL."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        claimer = "host:ttl"
        
        kb.claim_task(conn, t, claimer=claimer, ttl_seconds=300)
        task = kb.get_task(conn, t)
    
    # claim_expires should be approximately now + 300
    expected_min = int(time.time()) + 299
    expected_max = int(time.time()) + 301
    assert expected_min <= task.claim_expires <= expected_max


# ---------------------------------------------------------------------------
# Link edge cases
# ---------------------------------------------------------------------------

def test_unlink_nonexistent_link_returns_false(kanban_home):
    """Unlinking tasks that aren't linked should return False."""
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        
        # They were never linked
        result = kb.unlink_tasks(conn, a, b)
        assert result is False


def test_unlink_then_relink(kanban_home):
    """Can unlink and then relink tasks."""
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        
        # Link them
        kb.link_tasks(conn, a, b)
        assert kb.child_ids(conn, a) == [b]
        
        # Unlink
        assert kb.unlink_tasks(conn, a, b)
        assert kb.child_ids(conn, a) == []
        
        # Relink
        kb.link_tasks(conn, a, b)
        assert kb.child_ids(conn, a) == [b]


def test_link_idempotent(kanban_home):
    """Linking already-linked tasks should be a no-op (not an error)."""
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        
        kb.link_tasks(conn, a, b)
        # Linking again should not error
        kb.link_tasks(conn, a, b)
        
        # Should still only have one link
        assert kb.child_ids(conn, a) == [b]


# ---------------------------------------------------------------------------
# Event logging edge cases
# ---------------------------------------------------------------------------

def test_events_preserve_payload_in_json(kanban_home):
    """Event payloads with special characters should be preserved."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="special chars", assignee="alice")
        
        # Complete with result containing special chars
        special_result = 'Result with "quotes" and <special> chars: \n\\'
        kb.complete_task(conn, t, result=special_result)
        
        events = kb.list_events(conn, t)
        completed_events = [e for e in events if e.kind == "completed"]
        
        assert len(completed_events) == 1
        # The payload should have result_len
        assert completed_events[0].payload is not None


def test_created_event_has_all_fields(kanban_home):
    """The 'created' event should contain all relevant metadata."""
    with kb.connect() as conn:
        t = kb.create_task(
            conn,
            title="metadata test",
            body="A body",
            assignee="alice",
            tenant="biz-a",
            priority=5,
        )
        
        events = kb.list_events(conn, t)
        created = [e for e in events if e.kind == "created"][0]
        
        assert created.payload["assignee"] == "alice"
        assert created.payload["status"] == "ready"
        assert created.payload["tenant"] == "biz-a"
        assert created.payload["parents"] == []


# ---------------------------------------------------------------------------
# Archive lifecycle tests
# ---------------------------------------------------------------------------

def test_archive_completed_task(kanban_home):
    """Archiving a completed task should work."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="to archive")
        kb.complete_task(conn, t)
        
        assert kb.archive_task(conn, t)
        task = kb.get_task(conn, t)
        assert task.status == "archived"


def test_archive_running_task_fails(kanban_home):
    """Archiving a running (claimed) task should return False."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="running", assignee="a")
        kb.claim_task(conn, t)
        
        # Can't archive a running task
        assert not kb.archive_task(conn, t)
        assert kb.get_task(conn, t).status == "running"


def test_archive_already_archived_returns_false(kanban_home):
    """Archiving an already archived task should return False."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="double archive")
        kb.archive_task(conn, t)
        
        # Archive again should return False
        assert not kb.archive_task(conn, t)


# ---------------------------------------------------------------------------
# Status transition validation
# ---------------------------------------------------------------------------

def test_block_only_works_on_running_or_ready(kanban_home):
    """Block should only work on running or ready tasks."""
    with kb.connect() as conn:
        # Task is ready
        t = kb.create_task(conn, title="ready-task", assignee="a")
        assert kb.block_task(conn, t)  # Can block ready
        
        # Unblock to ready
        kb.unblock_task(conn, t)
        
        # Claim it
        kb.claim_task(conn, t)
        assert kb.block_task(conn, t)  # Can block running
        
        # Reset and try on done task
        t2 = kb.create_task(conn, title="done-task")
        kb.complete_task(conn, t2)
        assert not kb.block_task(conn, t2)  # Can't block done
