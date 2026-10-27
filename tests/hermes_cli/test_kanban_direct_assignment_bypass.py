"""Test that proves the direct-assignment bypass bug.

When a card is created with an explicit --assignee and an obviously oversized
brief (10+ acceptance criteria, 15+ named files), it should be routed through
triage for decomposition. However, the current code bypasses triage entirely
and sends it straight to ready/dispatch.

This test FAILS on current code (proving the bypass exists) and should PASS
after the fix is applied.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _oversized_brief():
    """Generate an obviously oversized brief that should trigger decomposition."""
    return """
This is a massive task that clearly needs decomposition before dispatch.

## Acceptance Criteria (12 distinct items)
1. Implement user authentication with OAuth2 flow
2. Add email verification with token expiration
3. Build password reset flow with secure tokens
4. Create user profile management with avatar upload
5. Implement role-based access control (RBAC)
6. Add audit logging for all sensitive operations
7. Build admin dashboard with user management
8. Implement two-factor authentication (TOTP)
9. Add session management with device tracking
10. Create API rate limiting per user/role
11. Build webhook system for external integrations
12. Implement data export in multiple formats (JSON, CSV, PDF)

## Files to Create/Modify (16 files)
- src/auth/oauth2.py (new)
- src/auth/verification.py (new)
- src/auth/password_reset.py (new)
- src/users/profile.py (new)
- src/users/rbac.py (new)
- src/audit/logger.py (new)
- src/admin/dashboard.py (new)
- src/admin/user_management.py (new)
- src/auth/totp.py (new)
- src/auth/sessions.py (new)
- src/api/rate_limit.py (new)
- src/webhooks/dispatcher.py (new)
- src/webhooks/registry.py (new)
- src/export/json_exporter.py (new)
- src/export/csv_exporter.py (new)
- src/export/pdf_exporter.py (new)

## Additional Requirements
- All endpoints must have comprehensive test coverage (>90%)
- Documentation must be written for every public API
- Security review required before merge
- Performance benchmarks for all new endpoints
- Integration tests with external OAuth providers
- Database migration scripts for new tables
- Error handling with user-friendly messages
- Logging at appropriate levels (DEBUG/INFO/WARN/ERROR)
"""


def test_oversized_brief_with_explicit_assignee_must_be_decomposed(kanban_home):
    """An oversized brief with explicit assignee must go through decomposition.
    
    EXPECTED BEHAVIOR (after fix):
    - A card with 12+ acceptance criteria and 15+ named files should be detected as oversized
    - It should be routed to triage OR automatically decomposed
    - A 'decomposed' event should exist in task_events
    - Child tasks should be created
    
    CURRENT BEHAVIOR (bug):
    - The card goes straight to 'ready' with no decomposition
    - No 'decomposed' event is recorded
    - The oversized brief is dispatched as-is to a single worker
    
    This test FAILS on current code (proving the bypass exists) and should PASS
    after the fix is applied.
    """
    oversized_body = _oversized_brief()
    
    # Create a card with explicit assignee and oversized brief
    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Implement comprehensive user management system",
            body=oversized_body,
            assignee="junior-dev",  # Explicit assignee
            created_by="test",
            triage=False,  # Default: bypass triage (this is the bug)
        )
        
        # Check the task status
        task = kb.get_task(conn, task_id)
        
        # Check for decomposed events
        events = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY created_at",
            (task_id,)
        ).fetchall()
        event_kinds = [e["kind"] for e in events]
        
        # Check for child tasks
        child_tasks = conn.execute(
            "SELECT id FROM tasks WHERE created_by = ? AND id != ?",
            ("decomposer", task_id)
        ).fetchall()
    
    # EXPECTED: Task should either be in triage OR have been decomposed
    # This assertion FAILS on current code (task.status == 'ready')
    assert task.status in ["triage", "todo"], (
        f"Oversized brief with explicit assignee should be in triage or todo, "
        f"but got '{task.status}'. This proves the bypass exists."
    )
    
    # EXPECTED: Should have a 'decomposed' event OR be in triage awaiting decomposition
    # This assertion FAILS on current code (no decomposition)
    has_decomposition = (
        "decomposed" in event_kinds or 
        task.status == "triage" or
        len(child_tasks) > 0
    )
    assert has_decomposition, (
        f"Oversized brief should have been decomposed or routed to triage. "
        f"Events: {event_kinds}, status: {task.status}, children: {len(child_tasks)}. "
        f"This proves the bypass exists."
    )
    
    print(f"✓ Task {task_id} was properly handled: status={task.status}, events={event_kinds}")


def test_triaged_card_gets_decomposed(kanban_home):
    """Control test: a card created with triage=True goes through decomposition.
    
    This test should PASS on both current and fixed code, showing that the
    triage path works correctly when explicitly used.
    """
    from hermes_cli.kanban_db_graph import decompose_triage_task
    
    oversized_body = _oversized_brief()
    
    # Create a card WITH triage
    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="Implement comprehensive user management system",
            body=oversized_body,
            assignee="junior-dev",
            created_by="test",
            triage=True,  # Explicit triage
        )
        
        task = kb.get_task(conn, task_id)
    
    # Should be in triage status
    assert task.status == "triage", f"Expected 'triage', got '{task.status}'"
    
    # Simulate decomposition (in real flow, this is done by the decomposer)
    children = [
        {"title": "Auth system", "body": "OAuth2, verification, password reset", "assignee": "junior-dev"},
        {"title": "User management", "body": "Profile, RBAC, admin dashboard", "assignee": "junior-dev"},
        {"title": "Security features", "body": "2FA, sessions, audit logging", "assignee": "junior-dev"},
    ]
    
    with kbc.connect() as conn:
        child_ids = decompose_triage_task(
            conn,
            task_id,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
        
        # Check events
        events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY created_at",
            (task_id,)
        ).fetchall()
        event_kinds = [e["kind"] for e in events]
    
    assert child_ids is not None
    assert len(child_ids) == 3
    assert "decomposed" in event_kinds, f"Expected 'decomposed' event, got: {event_kinds}"
    
    print(f"✓ Control test passed: Task {task_id} was properly decomposed into {len(child_ids)} children")
