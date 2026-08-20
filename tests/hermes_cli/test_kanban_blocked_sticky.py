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


@pytest.mark.parametrize("kind", ["needs_input", "capability"])
def test_human_block_kinds_remain_sticky(
    kanban_home: Path,
    kind: str,
) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title=f"human block: {kind}")
        kb.claim_task(conn, tid)
        assert kb.block_task(
            conn,
            tid,
            reason="operator action required",
            kind=kind,
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"


@pytest.mark.parametrize("completed_parent", [False, True])
def test_initially_blocked_task_is_sticky(
    kanban_home: Path,
    completed_parent: bool,
) -> None:
    """Creation-time human-ops blocks are deliberate, not dependency waits."""
    with kb.connect() as conn:
        parents: list[str] = []
        if completed_parent:
            parent = kb.create_task(conn, title="completed parent")
            conn.execute(
                "UPDATE tasks SET status = 'done' WHERE id = ?",
                (parent,),
            )
            parents.append(parent)

        tid = kb.create_task(
            conn,
            title="awaiting human ops",
            parents=parents,
            initial_status="blocked",
        )

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"


@pytest.mark.parametrize("completed_parent", [False, True])
def test_dispatch_does_not_claim_or_spawn_initially_blocked_task(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed_parent: bool,
) -> None:
    """The full dispatch tick must preserve a creation-time blocked card."""
    with kb.connect() as conn:
        parents: list[str] = []
        if completed_parent:
            parent = kb.create_task(conn, title="completed parent")
            conn.execute(
                "UPDATE tasks SET status = 'done' WHERE id = ?",
                (parent,),
            )
            parents.append(parent)

        tid = kb.create_task(
            conn,
            title="awaiting human ops",
            assignee="worker",
            parents=parents,
            initial_status="blocked",
        )
        claim_calls: list[str] = []
        spawn_calls: list[str] = []
        real_claim_task = kb.claim_task

        def spy_claim_task(connection, task_id, **kwargs):
            claim_calls.append(task_id)
            return real_claim_task(connection, task_id, **kwargs)

        def fake_spawn(task, _workspace):
            spawn_calls.append(task.id)
            return 12345

        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
        monkeypatch.setattr(kb, "claim_task", spy_claim_task)

        result = kb.dispatch_once(conn, spawn_fn=fake_spawn)

        assert result.promoted == 0
        assert result.spawned == []
        assert claim_calls == []
        assert spawn_calls == []
        assert kb.get_task(conn, tid).status == "blocked"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "",
        "{}",
        '{"assignee":"worker"}',
        '{"status":"unknown"}',
        '{"status":null}',
        '{"status":[]}',
        '{"status":{}}',
        "{not-json",
        "[]",
        '"blocked"',
        "1",
        "true",
    ],
)
def test_unreadable_creation_provenance_fails_closed(
    kanban_home: Path,
    payload: str | None,
) -> None:
    """A corrupt legacy event cannot crash dispatch or release blocked work."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="blocked with corrupt provenance",
            initial_status="blocked",
        )
        conn.execute(
            "UPDATE task_events SET payload = ? "
            "WHERE task_id = ? AND kind = 'created'",
            (payload, tid),
        )

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"


def test_deep_creation_provenance_does_not_abort_recompute_ready(
    kanban_home: Path,
) -> None:
    """Recursive JSON corruption must preserve the blocked card."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="blocked with deeply nested provenance",
            initial_status="blocked",
        )
        conn.execute(
            "UPDATE task_events SET payload = ? "
            "WHERE task_id = ? AND kind = 'created'",
            ("[" * 2000 + "]" * 2000, tid),
        )

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"


def test_dispatch_does_not_claim_or_spawn_deep_creation_provenance(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One corrupt created event must not abort or release dispatch work."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="blocked with deeply nested provenance",
            assignee="worker",
            initial_status="blocked",
        )
        conn.execute(
            "UPDATE task_events SET payload = ? "
            "WHERE task_id = ? AND kind = 'created'",
            ("[" * 2000 + "]" * 2000, tid),
        )
        claim_calls: list[str] = []
        spawn_calls: list[str] = []
        real_claim_task = kb.claim_task

        def spy_claim_task(connection, task_id, **kwargs):
            claim_calls.append(task_id)
            return real_claim_task(connection, task_id, **kwargs)

        def fake_spawn(task, _workspace):
            spawn_calls.append(task.id)
            return 12345

        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
        monkeypatch.setattr(kb, "claim_task", spy_claim_task)

        result = kb.dispatch_once(conn, spawn_fn=fake_spawn)

        assert result.promoted == 0
        assert result.spawned == []
        assert claim_calls == []
        assert spawn_calls == []
        assert kb.get_task(conn, tid).status == "blocked"




# ---------------------------------------------------------------------------
# Circuit-breaker blocks still auto-recover (preserve #40c1decb3 intent)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# unblock_task clears the sticky state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("clear_with", ["unblock", "promote"])
@pytest.mark.parametrize(
    ("failures", "expected_promoted"),
    [(0, 1), (2, 0)],
)
def test_explicit_clear_restores_circuit_breaker_retry_policy(
    kanban_home: Path,
    clear_with: str,
    failures: int,
    expected_promoted: int,
) -> None:
    """Old creation provenance must not make a later ``gave_up`` sticky."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="operator parked",
            initial_status="blocked",
        )

        if clear_with == "unblock":
            assert kb.unblock_task(conn, tid)
            expected_event = "unblocked"
        else:
            ok, error = kb.promote_task(conn, tid, actor="operator")
            assert ok and error is None
            expected_event = "promoted_manual"

        events = kb.list_events(conn, tid)
        assert events[-1].kind == expected_event

        # A later circuit-breaker block has no explicit ``blocked`` event and
        # keeps the existing retry-policy semantics: below the limit it may
        # recover; at the limit it remains blocked.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'blocked', consecutive_failures = ? "
                "WHERE id = ?",
                (failures, tid),
            )
            kb._append_event(conn, tid, "gave_up", {"failures": failures})

        assert kb.recompute_ready(conn, failure_limit=2) == expected_promoted
        expected_status = "ready" if expected_promoted else "blocked"
        assert kb.get_task(conn, tid).status == expected_status


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
