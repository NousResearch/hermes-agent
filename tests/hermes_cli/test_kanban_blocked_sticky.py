"""Regression tests for #28712 — kanban dispatcher must not auto-promote
worker-initiated ``kanban_block`` (sticky blocks), but must keep
auto-recovering circuit-breaker blocks.

The bug: when a worker called ``kanban_block(reason="review-required:
...")`` to hand off to a human, the dispatcher's ``recompute_ready``
would promote the task back to ``ready`` on the next tick.  The fresh
worker found nothing to do (work already applied), exited cleanly, and
got recorded as a ``protocol_violation`` → ``gave_up`` → promote → loop
until manual intervention.

These tests pin down:

* Worker / operator-initiated blocks are sticky and survive
  ``recompute_ready``.
* Circuit-breaker blocks (``gave_up`` event, status flipped via
  ``_record_task_failure``) still auto-recover — the original intent
  of #40c1decb3 is preserved.
* An explicit ``kanban_unblock`` clears the sticky state.
* The full block → promote → crash → ``gave_up`` loop is broken after
  this fix: subsequent ticks leave the task blocked.

The tangentially related schema-init ordering bug originally reported
in #28712 (``init_db`` crashing on legacy DBs that pre-dated the
``session_id`` migration) is covered separately by
``test_kanban_db.py::test_connect_migrates_legacy_db_before_optional_column_indexes``,
landed via #28754 / #28781 ahead of this fix.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Initial-status human gates require an affirmative authorization record
# ---------------------------------------------------------------------------


def _initial_gate(conn, *, with_parent=False):
    parents = []
    parent = None
    if with_parent:
        parent = kb.create_task(conn, title="parent")
        parents = [parent]
    gate = kb.create_task(
        conn,
        title="human gate",
        assignee="worker",
        parents=parents,
        initial_status="blocked",
    )
    return gate, parent


def _event_kinds(conn, task_id):
    return [event.kind for event in kb.list_events(conn, task_id)]


def _append_raw_event(conn, task_id, kind, payload):
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?, ?, ?, ?)",
        (task_id, kind, payload, int(time.time())),
    )
    conn.commit()


def test_initial_gate_preserves_blocked_lifecycle_event(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        events = kb.list_events(conn, gate)
        assert [event.kind for event in events] == [
            "created",
            "human_gate_created",
            "blocked",
        ]
        assert events[-1].payload == {
            "reason": "initial-status: created-blocked",
            "source": "create_task",
        }


def test_initial_gate_idempotency_refuses_existing_runnable_task(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        existing = kb.create_task(
            conn,
            title="ordinary task",
            assignee="worker",
            idempotency_key="shared-operation",
        )

        with pytest.raises(ValueError, match="not a human gate"):
            kb.create_task(
                conn,
                title="approval required",
                assignee="worker",
                initial_status="blocked",
                idempotency_key="shared-operation",
            )

        task = kb.get_task(conn, existing)
        assert task is not None and task.status == "ready"
        assert _event_kinds(conn, existing) == ["created"]


def test_initial_gate_idempotent_retry_accepts_existing_gate(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        first = kb.create_task(
            conn,
            title="approval required",
            assignee="worker",
            initial_status="blocked",
            idempotency_key="same-gate",
        )
        second = kb.create_task(
            conn,
            title="retry",
            assignee="worker",
            initial_status="blocked",
            idempotency_key="same-gate",
        )

        assert second == first
        assert _event_kinds(conn, first) == [
            "created",
            "human_gate_created",
            "blocked",
        ]


def test_initial_gate_idempotency_rejects_ambiguous_legacy_duplicates(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        ordinary = kb.create_task(
            conn,
            title="ordinary task",
            assignee="worker",
            idempotency_key="shared-operation",
        )
        gate = kb.create_task(
            conn,
            title="approval required",
            assignee="worker",
            initial_status="blocked",
            idempotency_key="gate-operation",
        )
        conn.execute(
            "UPDATE tasks SET idempotency_key=?, created_at=created_at+1 "
            "WHERE id=?",
            ("shared-operation", gate),
        )
        conn.commit()

        with pytest.raises(ValueError, match="not a human gate"):
            kb.create_task(
                conn,
                title="approval retry",
                assignee="worker",
                initial_status="blocked",
                idempotency_key="shared-operation",
            )

        ordinary_task = kb.get_task(conn, ordinary)
        gate_task = kb.get_task(conn, gate)
        assert ordinary_task is not None and ordinary_task.status == "ready"
        assert gate_task is not None and gate_task.status == "blocked"


def test_idempotent_gate_and_runnable_creators_serialize(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_lookup = getattr(kb, "_lookup_idempotent_task")
    first_lookup_barrier = threading.Barrier(2)
    local = threading.local()

    def synchronized_lookup(conn, idempotency_key, *, require_human_gate):
        result = original_lookup(
            conn,
            idempotency_key,
            require_human_gate=require_human_gate,
        )
        if not getattr(local, "passed_fast_path", False):
            local.passed_fast_path = True
            first_lookup_barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(kb, "_lookup_idempotent_task", synchronized_lookup)
    outcomes: list[tuple[str, str]] = []
    outcomes_lock = threading.Lock()

    def create(initial_status: str | None) -> None:
        try:
            with kb.connect() as conn:
                if initial_status is None:
                    task_id = kb.create_task(
                        conn,
                        title="same logical operation",
                        assignee="worker",
                        idempotency_key="racing-operation",
                    )
                else:
                    task_id = kb.create_task(
                        conn,
                        title="same logical operation",
                        assignee="worker",
                        initial_status=initial_status,
                        idempotency_key="racing-operation",
                    )
            outcome = ("task", task_id)
        except ValueError as exc:
            outcome = ("error", str(exc))
        except Exception as exc:  # pragma: no cover - asserted below
            outcome = ("unexpected", repr(exc))
        with outcomes_lock:
            outcomes.append(outcome)

    threads = [
        threading.Thread(target=create, args=(None,)),
        threading.Thread(target=create, args=("blocked",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    with kb.connect() as conn:
        rows = conn.execute(
            "SELECT id, status FROM tasks WHERE idempotency_key=?",
            ("racing-operation",),
        ).fetchall()
    assert len(rows) == 1
    assert len(outcomes) == 2
    assert all(kind != "unexpected" for kind, _value in outcomes)
    if rows[0]["status"] == "blocked":
        assert outcomes[0][1] == outcomes[1][1]
    else:
        assert sorted(kind for kind, _value in outcomes) == ["error", "task"]


def test_authorization_is_bound_to_task_execution_content(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        gate = kb.create_task(
            conn,
            title="approved operation",
            body="print safe report",
            assignee="worker",
            initial_status="blocked",
        )
        assert kb.unblock_task(
            conn,
            gate,
            actor="chief",
            reason="approved safe report",
        )
        authorization = next(
            event
            for event in kb.list_events(conn, gate)
            if event.kind == "human_gate_authorized"
        )
        assert authorization.payload is not None
        assert isinstance(authorization.payload["task_fingerprint"], str)
        assert len(authorization.payload["task_fingerprint"]) == 64

        conn.execute(
            "UPDATE tasks SET body=? WHERE id=?",
            ("perform different operation", gate),
        )
        conn.commit()

        assert kb.has_spawnable_ready(conn) is False
        assert kb.claim_task(conn, gate, claimer="worker") is None
        task = kb.get_task(conn, gate)
        assert task is not None and task.status == "blocked"
        rejected = kb.list_events(conn, gate)[-1]
        assert rejected.kind == "claim_rejected"
        assert rejected.payload == {"reason": "human_gate_not_authorized"}

        assert kb.unblock_task(
            conn,
            gate,
            actor="chief",
            reason="approved changed operation",
        )
        claimed = kb.claim_task(conn, gate, claimer="worker")
        assert claimed is not None
        assert claimed.body == "perform different operation"


def test_initial_blocked_gate_survives_parent_completion(kanban_home: Path) -> None:
    with kb.connect() as conn:
        gate, parent = _initial_gate(conn, with_parent=True)
        assert kb.get_task(conn, gate).status == "blocked"
        assert kb.complete_task(conn, parent, result="done")
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, gate).status == "blocked"
        assert "promoted" not in _event_kinds(conn, gate)


def test_legacy_initial_gate_is_detected_without_new_marker(kanban_home: Path) -> None:
    """A pre-fix DB has only created(status=blocked), not the new marker."""
    with kb.connect() as conn:
        gate, parent = _initial_gate(conn, with_parent=True)
        conn.execute(
            "DELETE FROM task_events WHERE task_id = ? "
            "AND kind IN ('human_gate_created', 'blocked')",
            (gate,),
        )
        conn.commit()
        assert _event_kinds(conn, gate) == ["created"]

        assert kb.complete_task(conn, parent, result="done")
        assert kb.get_task(conn, gate).status == "blocked"
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (gate,))
        conn.commit()
        assert kb.claim_task(conn, gate, claimer="dispatcher") is None
        assert kb.get_task(conn, gate).status == "blocked"


@pytest.mark.parametrize(
    "created_payload",
    [
        None,
        "{not-json",
        "null",
        "[]",
        "{}",
        json.dumps({"status": "BLOCKED"}),
        json.dumps({"status": "BLOCKED", "from_decompose_of": "t_parent"}),
    ],
)
def test_ambiguous_legacy_created_payload_fails_closed(
    kanban_home: Path, created_payload
) -> None:
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        conn.execute(
            "DELETE FROM task_events WHERE task_id = ? AND kind = 'human_gate_created'",
            (gate,),
        )
        conn.execute(
            "UPDATE task_events SET payload = ? "
            "WHERE task_id = ? AND kind = 'created'",
            (created_payload, gate),
        )
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (gate,))
        conn.commit()
        assert kb.claim_task(conn, gate, claimer="dispatcher") is None
        assert kb.get_task(conn, gate).status == "blocked"


@pytest.mark.parametrize(
    ("actor", "reason"),
    [
        (None, "approved"),
        ("chief", None),
        ("", "approved"),
        ("chief", ""),
        ("   ", "approved"),
        ("chief", "   "),
        (123, "approved"),
        ("chief", 123),
    ],
)
def test_initial_gate_unblock_requires_typed_nonblank_actor_and_reason(
    kanban_home: Path, actor, reason
) -> None:
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        before = _event_kinds(conn, gate)
        assert kb.unblock_task(conn, gate, actor=actor, reason=reason) is False
        assert kb.get_task(conn, gate).status == "blocked"
        assert _event_kinds(conn, gate) == before


def test_human_gate_authorization_event_precedes_unblocked_with_trimmed_payload(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        assert kb.unblock_task(
            conn,
            gate,
            actor="  chief  ",
            reason="  User authorized exact SHA abc123 in session s_1  ",
        )
        assert kb.get_task(conn, gate).status == "ready"
        events = kb.list_events(conn, gate)
        assert [event.kind for event in events] == [
            "created",
            "human_gate_created",
            "blocked",
            "human_gate_authorized",
            "unblocked",
        ]
        authorization, unblocked = events[-2:]
        assert authorization.id < unblocked.id
        assert authorization.payload is not None
        assert authorization.payload["actor"] == "chief"
        assert (
            authorization.payload["reason"]
            == "User authorized exact SHA abc123 in session s_1"
        )
        assert isinstance(authorization.payload["task_fingerprint"], str)
        assert len(authorization.payload["task_fingerprint"]) == 64
        assert unblocked.payload is None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "{not-json",
        "null",
        "[]",
        '"approved"',
        "{}",
        json.dumps({"actor": "chief", "reason": "   "}),
        json.dumps({"actor": 123, "reason": "approved"}),
    ],
)
def test_malformed_authorization_events_fail_closed_without_crashing(
    kanban_home: Path, payload
) -> None:
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        _append_raw_event(conn, gate, "human_gate_authorized", payload)
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (gate,))
        conn.commit()
        assert kb.claim_task(conn, gate, claimer="dispatcher") is None
        assert kb.get_task(conn, gate).status == "blocked"


def test_later_invalid_authorization_relocks_gate(kanban_home: Path) -> None:
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        _append_raw_event(
            conn,
            gate,
            "human_gate_authorized",
            json.dumps({"actor": "chief", "reason": "approved"}),
        )
        _append_raw_event(conn, gate, "human_gate_authorized", "[]")
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (gate,))
        conn.commit()
        assert kb.claim_task(conn, gate, claimer="dispatcher") is None
        assert kb.get_task(conn, gate).status == "blocked"


def test_claim_fails_closed_for_gate_forced_to_ready(kanban_home: Path) -> None:
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (gate,))
        conn.commit()
        assert kb.claim_task(conn, gate, claimer="dispatcher") is None
        assert kb.get_task(conn, gate).status == "blocked"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (gate,)
        ).fetchone()[0] == 0
        assert "claimed" not in _event_kinds(conn, gate)
        rejected = [e for e in kb.list_events(conn, gate) if e.kind == "claim_rejected"]
        assert rejected[-1].payload == {"reason": "human_gate_not_authorized"}


def test_dispatch_ready_fails_closed_for_unresolved_gate(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    spawned = []

    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (gate,))
        conn.commit()

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append((task.id, workspace)),
        )

        assert result.spawned == []
        assert spawned == []
        task = kb.get_task(conn, gate)
        assert task is not None and task.status == "blocked"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (gate,)
        ).fetchone()[0] == 0
        rejected = [e for e in kb.list_events(conn, gate) if e.kind == "claim_rejected"]
        assert rejected[-1].payload == {"reason": "human_gate_not_authorized"}


def test_claim_review_fails_closed_for_gate_forced_to_review(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (gate,))
        conn.commit()

        assert kb.claim_review_task(conn, gate, claimer="reviewer") is None
        task = kb.get_task(conn, gate)
        assert task is not None and task.status == "blocked"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (gate,)
        ).fetchone()[0] == 0
        assert "claimed" not in _event_kinds(conn, gate)
        rejected = [e for e in kb.list_events(conn, gate) if e.kind == "claim_rejected"]
        assert rejected[-1].payload == {"reason": "human_gate_not_authorized"}


def test_dispatch_review_fails_closed_for_unresolved_gate(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    spawned = []

    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (gate,))
        conn.commit()

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append((task.id, workspace)),
        )

        assert result.spawned == []
        assert spawned == []
        task = kb.get_task(conn, gate)
        assert task is not None and task.status == "blocked"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (gate,)
        ).fetchone()[0] == 0
        rejected = [e for e in kb.list_events(conn, gate) if e.kind == "claim_rejected"]
        assert rejected[-1].payload == {"reason": "human_gate_not_authorized"}


def test_dispatch_ready_dry_run_does_not_report_unresolved_gate_spawnable(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (gate,))
        conn.commit()
        before = _event_kinds(conn, gate)

        result = kb.dispatch_once(conn, dry_run=True)

        assert result.spawned == []
        task = kb.get_task(conn, gate)
        assert task is not None and task.status == "ready"
        assert _event_kinds(conn, gate) == before
        assert conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (gate,)
        ).fetchone()[0] == 0


def test_dispatch_review_dry_run_does_not_report_unresolved_gate_spawnable(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (gate,))
        conn.commit()
        before = _event_kinds(conn, gate)

        result = kb.dispatch_once(conn, dry_run=True)

        assert result.spawned == []
        task = kb.get_task(conn, gate)
        assert task is not None and task.status == "review"
        assert _event_kinds(conn, gate) == before
        assert conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (gate,)
        ).fetchone()[0] == 0


def test_unresolved_ready_gate_is_not_reported_spawnable(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (gate,))
        conn.commit()

        assert kb.has_spawnable_ready(conn) is False


def test_unresolved_review_gate_is_not_reported_spawnable(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (gate,))
        conn.commit()

        assert kb.has_spawnable_review(conn) is False


@pytest.mark.parametrize("dry_run", [False, True])
def test_manual_promote_cannot_bypass_human_gate_even_with_force(
    kanban_home: Path, dry_run: bool
) -> None:
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        ok, error = kb.promote_task(
            conn,
            gate,
            actor="operator",
            reason="recovery",
            force=True,
            dry_run=dry_run,
        )
        assert ok is False
        assert "not authorized" in error
        assert kb.get_task(conn, gate).status == "blocked"
        assert "promoted_manual" not in _event_kinds(conn, gate)


# ---------------------------------------------------------------------------
# Worker-initiated kanban_block must be sticky
# ---------------------------------------------------------------------------


def test_worker_block_is_not_auto_promoted_by_recompute_ready(kanban_home: Path) -> None:
    """A standalone task that a worker explicitly blocks for review
    must stay blocked across an arbitrary number of dispatcher ticks.
    Before #28712's fix, ``recompute_ready`` would silently flip it
    back to ``ready`` on the very next tick."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="needs human review")
        kb.claim_task(conn, tid)
        assert kb.block_task(
            conn, tid,
            reason="review-required: please verify ACL change",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "blocked"

        # Hammer the promotion code — exactly the dispatcher loop's
        # behaviour, just compressed in time.
        for _ in range(5):
            promoted = kb.recompute_ready(conn)
            assert promoted == 0, "worker-blocked task must not auto-promote"
            assert kb.get_task(conn, tid).status == "blocked"


def test_worker_block_on_child_with_done_parents_is_still_sticky(kanban_home: Path) -> None:
    """The parent-completion path is the one ``recompute_ready`` was
    designed for, so it's the most dangerous false-positive: even when
    every parent is done, a worker-initiated block on the child must
    stay blocked."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="parent ok")

        kb.claim_task(conn, child)
        kb.block_task(
            conn, child,
            reason="review-required: child needs sign-off",
            expected_run_id=kb.get_task(conn, child).current_run_id,
        )
        assert kb.get_task(conn, child).status == "blocked"

        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        assert kb.get_task(conn, child).status == "blocked"


# ---------------------------------------------------------------------------
# Circuit-breaker blocks still auto-recover (preserve #40c1decb3 intent)
# ---------------------------------------------------------------------------


def test_circuit_breaker_block_still_auto_promotes(kanban_home: Path) -> None:
    """A child that was put into ``blocked`` *without* a worker-issued
    ``kanban_block`` (e.g. a transient crash, manual DB triage) and whose
    ``consecutive_failures`` is still *below* the circuit-breaker limit
    must get auto-promoted when its parents complete — preserves the
    pre-#28712 recovery semantics for genuinely transient failures.

    The complementary case — a block whose failure count has *reached*
    the limit must stay blocked — is covered by
    ``test_kanban_db.py::test_recompute_ready_skips_tasks_at_failure_limit``
    (#35072).  Together they pin the contract: ``recompute_ready`` defers
    the give-up decision to the same effective limit the breaker uses, so
    the two never disagree.
    """
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="ok")

        # Simulate a transient circuit-breaker / direct triage that flips
        # status without emitting a ``blocked`` event — exactly what
        # ``_record_task_failure`` does below the limit.  One failure is
        # under the default limit (2), so recovery is still correct.
        conn.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=1, "
            "last_failure_error='transient error' WHERE id=?",
            (child,),
        )
        conn.commit()

        promoted = kb.recompute_ready(conn)
        assert promoted == 1
        task = kb.get_task(conn, child)
        assert task.status == "ready"
        # Counter is preserved across recovery (not reset) so the breaker
        # can still accumulate if the task keeps failing (#35072).
        assert task.consecutive_failures == 1


def test_gave_up_event_alone_does_not_make_block_sticky(kanban_home: Path) -> None:
    """The circuit-breaker emits ``gave_up`` (not ``blocked``).  Make
    sure ``_has_sticky_block`` doesn't accidentally treat ``gave_up``
    as sticky — otherwise we'd regress the safety net for genuinely
    transient crashes."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="ok")

        # Status + event match what _record_task_failure writes when
        # the breaker trips.
        conn.execute(
            "UPDATE tasks SET status='blocked' WHERE id=?", (child,),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'gave_up', NULL, ?)",
            (child, int(time.time())),
        )
        conn.commit()

        promoted = kb.recompute_ready(conn)
        assert promoted == 1
        assert kb.get_task(conn, child).status == "ready"


# ---------------------------------------------------------------------------
# unblock_task clears the sticky state
# ---------------------------------------------------------------------------


def test_unblock_clears_sticky_state_and_lets_block_recover(kanban_home: Path) -> None:
    """``hermes kanban unblock`` (or the ``kanban_unblock`` tool) is
    the only legitimate way out of a worker-initiated block.  After
    unblock, a *subsequent* circuit-breaker block on the same task
    must again be eligible for auto-recovery."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid,
            reason="review-required: ...",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.unblock_task(conn, tid)
        # After unblock the task is no longer blocked at all.
        assert kb.get_task(conn, tid).status == "ready"

        # Now simulate a *later* circuit-breaker block (no new
        # ``blocked`` event, just status flip).  The most recent
        # block/unblock event is ``unblocked`` → guard does not fire
        # → recompute can recover.
        conn.execute(
            "UPDATE tasks SET status='blocked' WHERE id=?", (tid,),
        )
        conn.commit()

        promoted = kb.recompute_ready(conn)
        assert promoted == 1
        assert kb.get_task(conn, tid).status == "ready"


# ---------------------------------------------------------------------------
# Full bug-shaped loop: block → promote → crash → gave_up → next tick
# ---------------------------------------------------------------------------


def test_protocol_violation_loop_is_broken(kanban_home: Path) -> None:
    """Reproduces the exact #28712 loop and asserts the dispatcher
    leaves the task blocked instead of cycling.

    Loop shape from the issue:

    1. Worker calls ``kanban_block`` → status='blocked',
       ``task_runs.outcome='blocked'``, ``blocked`` event.
    2. (Bug) Dispatcher promotes back to ``ready``.
    3. Fresh worker exits cleanly without terminal tool call →
       ``protocol_violation`` event.
    4. ``_record_task_failure(failure_limit=1)`` → ``gave_up`` event,
       status='blocked' again.
    5. (Bug) Dispatcher promotes again → infinite loop.

    With the fix in place, step 2 never happens — the test simulates
    one would-be loop cycle by faking the crash-then-gave_up entries
    that *would* have been written and asserts the *next* tick still
    leaves the task blocked.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="loop reproducer")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid,
            reason="review-required: human eyes please",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "blocked"

        # First dispatcher tick — must NOT promote.
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"

        # Simulate the (hypothetical) protocol_violation + gave_up
        # entries that the dispatcher would have written if the bug
        # were still present.  Even with those event rows in place,
        # the worker-initiated ``blocked`` event is the most recent
        # of the ``{blocked, unblocked}`` pair, so the sticky guard
        # still fires.
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'protocol_violation', NULL, ?)",
            (tid, now),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'gave_up', NULL, ?)",
            (tid, now + 1),
        )
        conn.commit()

        # Subsequent ticks must still leave it blocked.
        for _ in range(3):
            promoted = kb.recompute_ready(conn)
            assert promoted == 0
            assert kb.get_task(conn, tid).status == "blocked"


# ---------------------------------------------------------------------------
# Schema-init recovery on legacy DBs is covered by
# tests/hermes_cli/test_kanban_db.py::test_connect_migrates_legacy_db_before_optional_column_indexes
# (landed via #28754 / #28781).  The original PR shipped a duplicate test
# here; dropped during salvage to avoid two assertions of the same contract.
# ---------------------------------------------------------------------------
