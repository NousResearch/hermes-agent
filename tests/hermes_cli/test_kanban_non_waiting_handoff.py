"""Non-waiting Kanban handoff contracts."""

from __future__ import annotations

import concurrent.futures
import json
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
    kb.init_db(board="tech-coe")
    return home


def test_concurrent_idempotent_create_has_one_winner(kanban_home):
    def create(_: int) -> str:
        with kbc.connect_closing(board="tech-coe") as conn:
            return kb.create_task(
                conn,
                title="same intent",
                assignee="software-eng",
                idempotency_key="daily:2026-10-05:v1",
                board="tech-coe",
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        ids = list(pool.map(create, range(20)))

    assert len(set(ids)) == 1
    with kbc.connect_closing(board="tech-coe") as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ? AND status != 'archived'",
            ("daily:2026-10-05:v1",),
        ).fetchone()[0]
    assert count == 1


def test_idempotent_replay_with_different_intent_fails_closed(kanban_home):
    with kbc.connect_closing(board="tech-coe") as conn:
        kb.create_task(
            conn, title="original", assignee="software-eng",
            idempotency_key="phase:key", board="tech-coe",
        )
        with pytest.raises(ValueError, match="immutable intent"):
            kb.create_task(
                conn, title="changed", assignee="software-eng",
                idempotency_key="phase:key", board="tech-coe",
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE idempotency_key = 'phase:key'"
        ).fetchone()[0] == 1


def test_completion_commits_evidence_and_promotes_child_immediately(kanban_home):
    with kbc.connect_closing(board="tech-coe") as conn:
        parent = kb.create_task(conn, title="parent", assignee="software-eng", board="tech-coe")
        child = kb.create_task(
            conn, title="child", assignee="qa-security", parents=[parent], board="tech-coe",
        )
        assert kb.get_task(conn, child).status == "todo"
        assert kb.complete_task(
            conn,
            parent,
            summary="implementation complete",
            metadata={"evidence": ["tests:pass"], "commit": "abc123"},
        )
        assert kb.get_task(conn, child).status == "ready"
        run = kb.latest_run(conn, parent)
        assert run.metadata["evidence"] == ["tests:pass"]
        kinds = [
            row["kind"] for row in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (child,)
            )
        ]
        assert "promoted" in kinds


def test_escalation_is_deduplicated_privacy_safe_and_kill_switchable(kanban_home):
    secret = "LINE-ID-raw-secret"
    with kbc.connect_closing(board="tech-coe") as conn:
        source = kb.create_task(
            conn, title="customer phone 0812345678", body=secret,
            assignee="software-eng", board="tech-coe",
        )
        assert kb.block_task(conn, source, reason=secret, kind="needs_input")

        assert kbd.process_handoff_escalations(
            conn, board="tech-coe", enabled=False,
        ) == []
        first = kbd.process_handoff_escalations(
            conn, board="tech-coe", enabled=True,
        )
        assert len(first) == 1
        assert kbd.process_handoff_escalations(
            conn, board="tech-coe", enabled=True,
        ) == []

        escalation = kb.get_task(conn, first[0])
        assert escalation.assignee == "tech-cto"
        serialized = json.dumps(
            {
                "title": escalation.title,
                "body": escalation.body,
                "events": [
                    json.loads(row["payload"] or "{}")
                    for row in conn.execute(
                        "SELECT payload FROM task_events WHERE task_id = ? ORDER BY id", (source,)
                    )
                ],
            },
            sort_keys=True,
        )
        assert secret not in escalation.body
        assert "0812345678" not in escalation.body
        handoff_rows = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? "
            "AND kind = 'handoff_escalated' ORDER BY id",
            (source,),
        ).fetchall()
        assert len(handoff_rows) == 1
        handoff_payload = json.loads(handoff_rows[0]["payload"])
        assert set(handoff_payload) == {
            "source_event_id", "board", "task_ref", "state", "failure_class",
            "attempt_count", "owner_profile", "timestamp", "run_id",
        }
        assert serialized.count("Kanban escalation") == 1


def test_escalation_suppresses_when_source_state_supersedes_failure(kanban_home):
    """Repro of t_e9a27ba7 / t_f6e84af7 / t_06e69fcf / t_b4511d0e:

    source emitted ``blocked``, then completed before the escalator was
    enabled. The escalator must NOT page a new escalation card; it must
    write a sanitized ``handoff_suppressed`` audit event and the
    consumed-outbox row must record ``suppressed_terminal``.
    """
    secret = "sk_liv...data"
    with kbc.connect_closing(board="tech-coe") as conn:
        source = kb.create_task(
            conn, title="customer profile 0812345678", body=secret,
            assignee="software-eng", board="tech-coe",
        )
        assert kb.block_task(conn, source, reason=secret, kind="needs_input")
        # Successful completion supersedes the blocked failure epoch.
        assert kb.complete_task(
            conn, source, summary="recovered",
            metadata={"evidence": ["tests:pass"]},
        )
        assert kb.get_task(conn, source).status == "done"

        # First-time enable: bootstrap cursor lands "now" so the BLOCKED
        # source event is reachable via the >= predicate; the current-state
        # check must downgrade it to a sanitized suppression audit.
        assert kbd.process_handoff_escalations(
            conn, board="tech-coe", enabled=True,
        ) == []

        decisions = [
            dict(r) for r in conn.execute(
                "SELECT source_event_id, source_task_id, decision, "
                "       escalation_task_id, source_kind "
                "  FROM escalation_consumed_events "
                " WHERE board = 'tech-coe' "
                " ORDER BY source_event_id",
            )
        ]
        blocked_decision = next(
            d for d in decisions if d["source_kind"] == "blocked"
        )
        assert blocked_decision["decision"] == "suppressed_terminal"
        assert blocked_decision["escalation_task_id"] is None

        suppressed = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? "
            "AND kind = 'handoff_suppressed' ORDER BY id",
            (source,),
        ).fetchall()
        assert len(suppressed) == 1
        payload = json.loads(suppressed[0]["payload"])
        assert payload["decision"] == "suppressed"
        assert payload["state"] == "done"
        assert payload["failure_class"] == "needs_input"
        for forbidden in (secret, "0812345678", "customer profile"):
            assert forbidden not in json.dumps(payload)
        assert set(payload) == {
            "source_event_id", "board", "decision", "reason", "state",
            "failure_class", "owner_profile", "timestamp",
        }

        # Rescan is a no-op even after the source is archived/deleted.
        assert kbd.process_handoff_escalations(
            conn, board="tech-coe", enabled=True,
        ) == []


def test_escalation_dedup_survives_close_and_resume(kanban_home):
    """A second dispatcher tick after the first one committed must not
    create a duplicate escalation card. The persistent outbox row is the
    durable cursor across process restarts.
    """
    with kbc.connect_closing(board="tech-coe") as conn:
        source = kb.create_task(
            conn, title="incident", body="some secret",
            assignee="software-eng", board="tech-coe",
        )
        assert kb.block_task(conn, source, reason="x", kind="capability")
        first = kbd.process_handoff_escalations(
            conn, board="tech-coe", enabled=True,
        )
        assert len(first) == 1
        # The outbox cursor row must exist for the source event.
        outbox = conn.execute(
            "SELECT * FROM escalation_consumed_events WHERE board = 'tech-coe'"
        ).fetchall()
        assert len(outbox) == 1
        assert outbox[0]["decision"] == "escalated"

        # Even if the id cursor were reset (manual replay) the outbox
        # primary key still suppresses the duplicate.
        conn.execute("DELETE FROM kanban_escalation_cursors")
        again = kbd.process_handoff_escalations(
            conn, board="tech-coe", enabled=True,
        )
        assert again == []

        # Exactly one escalation task and one handoff_escalated event.
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE idempotency_key LIKE 'escalate:%'"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind = 'handoff_escalated'"
        ).fetchone()[0] == 1


def test_escalation_cursor_bootstrap_skips_pre_enable_backlog(kanban_home):
    """Bounded bootstrap: a 100-day-old ``gave_up`` event that landed
    before escalation was enabled must NOT page as a current incident.
    A new failure landing after enable must still be processed.
    """
    import time
    with kbc.connect_closing(board="tech-coe") as conn:
        old_source = kb.create_task(
            conn, title="ancient failure", assignee="software-eng", board="tech-coe",
        )
        # Backdate the synthetic failure event by directly mutating
        # created_at — equivalent to a 100-day-old historical event.
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, 'gave_up', ?, ?)",
            (
                old_source,
                json.dumps({"reason": "stale", "kind": "gave_up", "source_status": "ready"}),
                int(time.time()) - 100 * 24 * 60 * 60,
            ),
        )
        # A brand-new failure after enable must STILL be processed.
        new_source = kb.create_task(
            conn, title="current failure", body="LINE-ID-raw-secret",
            assignee="software-eng", board="tech-coe",
        )
        assert kb.block_task(conn, new_source, reason="x", kind="needs_input")

        # First enabled tick.
        escalated = kbd.process_handoff_escalations(
            conn, board="tech-coe", enabled=True,
        )
        # Only the new failure is escalated; the 100-day-old row is
        # bounded out of scope by the lookback floor and is never
        # written into the consumed outbox.
        assert len(escalated) == 1
        # No escalation card was created for the old source.
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks "
            "WHERE idempotency_key LIKE 'escalate:%' "
            "  AND body LIKE ?",
            (f"%{old_source}%",),
        ).fetchone()[0] == 0
        # The 100-day-old row is bounded out by the recency filter, so
        # no consumed-outbox row exists for it. The new failure has one.
        assert conn.execute(
            "SELECT COUNT(*) FROM escalation_consumed_events "
            "WHERE board = 'tech-coe'"
        ).fetchone()[0] == 1
        # Rescan is a no-op for both windows.
        assert kbd.process_handoff_escalations(
            conn, board="tech-coe", enabled=True,
        ) == []


def test_escalation_failure_epoch_keying_keeps_independent_cards(kanban_home):
    """A retried-after-failure task that fails a second time must produce
    a SECOND escalation card, not collapse onto the first. Keyed by
    ``escalate:<board>:<task_id>:<failure_class>:<failure_epoch>``.
    """
    with kbc.connect_closing(board="tech-coe") as conn:
        source = kb.create_task(
            conn, title="flaky", assignee="software-eng", board="tech-coe",
        )
        # First capability failure epoch.
        kb.block_task(conn, source, reason="retry 1", kind="capability")
        first = kbd.process_handoff_escalations(
            conn, board="tech-coe", enabled=True,
        )
        assert len(first) == 1
        # Manually transition source to ready to allow a re-block.
        conn.execute(
            "UPDATE tasks SET status = 'ready', block_kind = NULL, "
            "block_recurrences = 0, claim_lock = NULL, claim_expires = NULL "
            "WHERE id = ?",
            (source,),
        )
        # Retry — re-block with a new failure epoch.
        kb.block_task(conn, source, reason="retry 2", kind="capability")
        second = kbd.process_handoff_escalations(
            conn, board="tech-coe", enabled=True,
        )
        assert len(second) == 1
        # Distinct escalation tasks (failure epoch is part of the key).
        assert first[0] != second[0]
        # Each card carries its own source_event_id.
        body_first = json.loads(kb.get_task(conn, first[0]).body)
        body_second = json.loads(kb.get_task(conn, second[0]).body)
        assert body_first["source_event_id"] != body_second["source_event_id"]
        assert body_first["source_event_id"] < body_second["source_event_id"]
