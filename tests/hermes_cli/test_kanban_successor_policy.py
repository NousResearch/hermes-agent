"""TDD acceptance tests for idempotent Kanban successor card creation policy.

Covers:
  1. Duplicate card suppression within the same tick/retry/timeout
  2. blocked successor non-reuse (terminal history preserved)
  3. Budget exceeded / non-actionable status → card 0
  4. review-required creates independent successor (no parent dependency)
  5. CLI/gateway/dashboard/daemon paths use the same policy helper
  6. Regression: existing review-successor / deadlock / quota / lifecycle tests
  7. Task body: no complete until independent Bright review passes
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_successor_policy as sp


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def conn(tmp_path, monkeypatch):
    """Fresh in-memory-backed kanban DB per test."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    (tmp_path / ".hermes").mkdir(parents=True, exist_ok=True)
    kb.init_db()
    c = kb.connect()
    yield c
    c.close()


def make_req(
    source_id: str = "src-001",
    reason_class: str = "review-required",
    reason_detail: str = "code needs review",
    assignee: str = "bright",
    title: str = "Review successor",
    body: str = "Please review.",
    target_ref: str = "",
) -> sp.SuccessorRequest:
    return sp.SuccessorRequest(
        source_task_id=source_id,
        reason_class=reason_class,
        reason_detail=reason_detail,
        assignee=assignee,
        title=title,
        body=body,
        target_ref=target_ref,
    )


# ---------------------------------------------------------------------------
# 1. Duplicate suppression — same tick / retry / timeout
# ---------------------------------------------------------------------------


def test_duplicate_card_suppressed_on_second_evaluate(conn):
    """evaluate_successor returns 'reuse' for an identical active request."""
    req = make_req()
    state = sp.SuccessorPolicyState()

    # First evaluate: allowed create.
    d1 = sp.evaluate_successor(conn, req, state=state)
    assert d1.allowed
    assert d1.action == "create"

    # Apply (creates card).
    card_id = sp.apply_successor(conn, req, d1)
    assert card_id is not None

    # Second evaluate with same request: should reuse the active successor.
    d2 = sp.evaluate_successor(conn, req, state=state)
    assert d2.action == "reuse"
    assert d2.existing_id == card_id
    assert not d2.allowed or d2.action == "reuse"  # no new card needed


def test_same_idempotency_key_returns_existing_card(conn):
    """Two evaluate+apply cycles with the same key yield the same card id."""
    req = make_req(target_ref="commit-abc123")
    state = sp.SuccessorPolicyState()

    d1 = sp.evaluate_successor(conn, req, state=state)
    id1 = sp.apply_successor(conn, req, d1)

    state2 = sp.SuccessorPolicyState()  # fresh per-tick state, same DB
    d2 = sp.evaluate_successor(conn, req, state=state2)
    id2 = sp.apply_successor(conn, req, d2) if d2.action == "create" else d2.existing_id

    assert id1 == id2, "Same idempotency key must return same card"


# ---------------------------------------------------------------------------
# 2. Blocked successor non-reuse; terminal history preserved
# ---------------------------------------------------------------------------


def test_blocked_successor_not_reused(conn):
    """A successor in 'blocked' status is NOT reused — a new card is allowed."""
    req = make_req()
    state = sp.SuccessorPolicyState()

    d1 = sp.evaluate_successor(conn, req, state=state)
    card_id = sp.apply_successor(conn, req, d1)
    assert card_id is not None

    # Move the successor to blocked.
    kb.block_task(conn, card_id, reason="manual-block", kind="needs_input")

    state2 = sp.SuccessorPolicyState()
    d2 = sp.evaluate_successor(conn, req, state=state2)
    # Policy must NOT reuse blocked successor.
    assert d2.action != "reuse", "blocked successor must not be reused"


def test_terminal_successor_not_reused(conn):
    """Done/cancelled/archived successors are NOT reused."""
    for terminal_action in ("done", "cancelled", "archived"):
        req = make_req(source_id=f"src-{terminal_action}", target_ref=terminal_action)
        state = sp.SuccessorPolicyState()

        d1 = sp.evaluate_successor(conn, req, state=state)
        card_id = sp.apply_successor(conn, req, d1)
        assert card_id is not None

        if terminal_action == "done":
            kb.complete_task(conn, card_id, summary="done")
        elif terminal_action == "cancelled":
            conn.execute("UPDATE tasks SET status='cancelled' WHERE id=?", (card_id,))
            conn.commit()
        else:
            kb.archive_task(conn, card_id)

        state2 = sp.SuccessorPolicyState()
        d2 = sp.evaluate_successor(conn, req, state=state2)
        assert d2.action in ("create", "defer", "reject"), (
            f"{terminal_action} successor must not be reused, got {d2.action}"
        )


# ---------------------------------------------------------------------------
# 3. Budget exceeded / non-actionable → 0 new cards
# ---------------------------------------------------------------------------


def test_per_source_budget_exceeded_returns_defer(conn):
    """After max_per_source_total successors, evaluate returns 'defer'."""
    budget = 3
    policy = sp.SuccessorPolicy(max_per_source_total=budget)
    source_id = "src-budget-test"

    for i in range(budget):
        req = make_req(source_id=source_id, target_ref=f"ref-{i}")
        state = sp.SuccessorPolicyState()
        d = sp.evaluate_successor(conn, req, state=state, policy=policy)
        if d.action == "create":
            sp.apply_successor(conn, req, d)

    # Next request for the same source must be deferred.
    req_extra = make_req(source_id=source_id, target_ref="ref-overflow")
    state_new = sp.SuccessorPolicyState()
    d_extra = sp.evaluate_successor(conn, req_extra, state=state_new, policy=policy)
    assert d_extra.action == "defer", f"Expected defer after budget, got {d_extra.action}"


def test_global_per_tick_budget_limits_new_cards(conn):
    """max_per_tick_global caps total new successors across all sources in one tick."""
    tick_cap = 2
    policy = sp.SuccessorPolicy(max_per_tick_global=tick_cap)
    state = sp.SuccessorPolicyState()
    created = 0

    for i in range(tick_cap + 3):
        req = make_req(source_id=f"src-{i}", target_ref=f"ref-{i}")
        d = sp.evaluate_successor(conn, req, state=state, policy=policy)
        if d.action == "create":
            sp.apply_successor(conn, req, d, state=state)
            created += 1

    assert created <= tick_cap, f"Global tick cap {tick_cap} exceeded: created {created}"


def test_non_actionable_reason_rejected(conn):
    """needs_input, credential_missing, observation reasons → reject, no card."""
    non_actionable = [
        "needs_input",
        "credential_missing",
        "observation",
        "duplicate_diagnosis",
        "dirty_workspace",
    ]
    for reason in non_actionable:
        req = make_req(
            source_id=f"src-{reason}",
            reason_class=reason,
            target_ref=reason,
        )
        state = sp.SuccessorPolicyState()
        d = sp.evaluate_successor(conn, req, state=state)
        assert d.action == "reject", (
            f"reason_class={reason!r} should be rejected, got {d.action!r}"
        )


def test_cooldown_blocks_too_fast_recreation(conn, monkeypatch):
    """Same idempotency key within cooldown window after a terminal is deferred."""
    cooldown = 300
    policy = sp.SuccessorPolicy(cooldown_seconds=cooldown)

    req = make_req(target_ref="cooldown-ref")
    state = sp.SuccessorPolicyState()

    # Create and immediately complete (terminal).
    d1 = sp.evaluate_successor(conn, req, state=state, policy=policy)
    card_id = sp.apply_successor(conn, req, d1)
    kb.complete_task(conn, card_id, summary="done quickly")

    # Simulate "now" is still within cooldown (last_created_at = now).
    # evaluate_successor should defer because cooldown hasn't elapsed.
    state2 = sp.SuccessorPolicyState()
    d2 = sp.evaluate_successor(conn, req, state=state2, policy=policy)
    assert d2.action == "defer", (
        f"Expected defer within cooldown, got {d2.action}"
    )


# ---------------------------------------------------------------------------
# 4. review-required creates independent successor (no parent)
# ---------------------------------------------------------------------------


def test_review_required_successor_has_no_parent(conn):
    """review-required successor card must NOT be added as child of source."""
    req = make_req(reason_class="review-required")
    state = sp.SuccessorPolicyState()

    d = sp.evaluate_successor(conn, req, state=state)
    assert d.action == "create"

    card_id = sp.apply_successor(conn, req, d)
    assert card_id is not None

    # Check no parent link exists.
    parents = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ?",
        (card_id,),
    ).fetchall()
    assert parents == [], (
        "review-required successor must have no parents (deadlock prevention)"
    )


def test_remediation_successor_has_no_parent(conn):
    """remediation successor also gets no auto-parent."""
    req = make_req(reason_class="remediation")
    state = sp.SuccessorPolicyState()
    d = sp.evaluate_successor(conn, req, state=state)
    card_id = sp.apply_successor(conn, req, d)
    parents = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ?", (card_id,)
    ).fetchall()
    assert parents == []


# ---------------------------------------------------------------------------
# 5. Unified entry point — CLI / gateway / daemon use same policy
# ---------------------------------------------------------------------------


def test_policy_module_exposes_canonical_entry_point():
    """kanban_successor_policy must expose evaluate_successor and apply_successor."""
    assert hasattr(sp, "evaluate_successor"), "evaluate_successor missing"
    assert hasattr(sp, "apply_successor"), "apply_successor missing"
    assert hasattr(sp, "SuccessorRequest"), "SuccessorRequest missing"
    assert hasattr(sp, "SuccessorDecision"), "SuccessorDecision missing"
    assert hasattr(sp, "SuccessorPolicyState"), "SuccessorPolicyState missing"
    assert hasattr(sp, "SuccessorPolicy"), "SuccessorPolicy missing"


def test_kanban_db_imports_successor_policy():
    """kanban_db must import or reference kanban_successor_policy."""
    import importlib, inspect
    src = inspect.getsource(kb)
    assert "kanban_successor_policy" in src or "successor_policy" in src, (
        "kanban_db.py must reference successor_policy so all creation paths use it"
    )


# ---------------------------------------------------------------------------
# 6. Audit events recorded on defer / budget_exceeded
# ---------------------------------------------------------------------------


def test_audit_event_recorded_on_defer(conn):
    """When evaluate returns defer, apply_successor records a structured event."""
    policy = sp.SuccessorPolicy(max_per_tick_global=0)  # cap=0 → always defer
    req = make_req(source_id="src-audit")
    state = sp.SuccessorPolicyState()

    d = sp.evaluate_successor(conn, req, state=state, policy=policy)
    assert d.action == "defer"

    # apply_successor with defer should record an audit event.
    sp.apply_successor(conn, req, d)

    events = conn.execute(
        "SELECT kind, payload FROM task_events WHERE kind = 'successor_deferred' "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchall()
    assert events, "Expected 'successor_deferred' audit event in task_events"


def test_audit_event_recorded_on_create(conn):
    """When a new card is created, an audit event is recorded."""
    req = make_req(source_id="src-create-audit")
    state = sp.SuccessorPolicyState()
    d = sp.evaluate_successor(conn, req, state=state)
    sp.apply_successor(conn, req, d)

    events = conn.execute(
        "SELECT kind FROM task_events WHERE kind = 'successor_created' "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchall()
    assert events, "Expected 'successor_created' audit event"


def test_audit_event_recorded_on_reuse(conn):
    """When an existing card is reused, an audit event is recorded."""
    req = make_req(source_id="src-reuse-audit")
    state = sp.SuccessorPolicyState()

    d1 = sp.evaluate_successor(conn, req, state=state)
    sp.apply_successor(conn, req, d1)

    state2 = sp.SuccessorPolicyState()
    d2 = sp.evaluate_successor(conn, req, state=state2)
    assert d2.action == "reuse"
    sp.apply_successor(conn, req, d2)

    events = conn.execute(
        "SELECT kind FROM task_events WHERE kind = 'successor_reused' "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchall()
    assert events, "Expected 'successor_reused' audit event"


# ---------------------------------------------------------------------------
# 6b. Regression: idempotency key in kanban_db unchanged
# ---------------------------------------------------------------------------


def test_kanban_db_idempotency_key_unchanged(conn):
    """Direct create_task idempotency still works after policy module addition."""
    a = kb.create_task(conn, title="first", idempotency_key="reg-key-001")
    b = kb.create_task(conn, title="second", idempotency_key="reg-key-001")
    assert a == b, "kanban_db idempotency_key regression"


def test_kanban_db_archived_idempotency_key_allows_new(conn):
    """Archived task doesn't block fresh create with same idempotency key."""
    a = kb.create_task(conn, title="first", idempotency_key="arch-key")
    kb.archive_task(conn, a)
    b = kb.create_task(conn, title="second", idempotency_key="arch-key")
    assert a != b


# ---------------------------------------------------------------------------
# 6c. Regression: policy rejects unknown reason_class (fail-closed)
# ---------------------------------------------------------------------------


def test_unknown_reason_class_is_rejected(conn):
    """An unrecognized reason_class is rejected (fail-closed)."""
    req = make_req(reason_class="totally_unknown_reason")
    state = sp.SuccessorPolicyState()
    d = sp.evaluate_successor(conn, req, state=state)
    assert d.action == "reject", f"Unknown reason class should be rejected, got {d.action}"


# ---------------------------------------------------------------------------
# 6d. per-reason / per-tick de-dup
# ---------------------------------------------------------------------------


def test_per_reason_per_tick_dedup(conn):
    """Same reason_class from different sources within one tick → only first allowed."""
    policy = sp.SuccessorPolicy(max_per_reason_per_tick=1)
    state = sp.SuccessorPolicyState()

    req1 = make_req(source_id="src-A", reason_class="review-required", target_ref="A")
    req2 = make_req(source_id="src-B", reason_class="review-required", target_ref="B")

    d1 = sp.evaluate_successor(conn, req1, state=state, policy=policy)
    assert d1.action == "create"
    sp.apply_successor(conn, req1, d1, state=state)

    d2 = sp.evaluate_successor(conn, req2, state=state, policy=policy)
    assert d2.action == "defer", (
        f"Same reason_class in same tick should be deferred after max=1, got {d2.action}"
    )
