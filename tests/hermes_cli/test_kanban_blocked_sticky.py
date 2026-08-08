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
from hermes_cli.dashboard_auth import Session
from hermes_cli.dashboard_auth.base import _attest_verified_session


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolate all board state from the operator's active Kanban database."""
    for key in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_ATTACHMENTS_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    assert kb.kanban_db_path().is_relative_to(home)
    return home


def _raw_human_session() -> Session:
    return Session(
        user_id="user-123",
        email="operator@example.com",
        display_name="Test Operator",
        provider="test",
        org_id="org-1",
        access_token="verified-by-auth-middleware",
        expires_at=9_999_999_999,
        refresh_token="refresh-token",
    )


def _human_session() -> Session:
    return _attest_verified_session(_raw_human_session())


def test_unverified_session_object_cannot_authorize_human_gate(
    kanban_home: Path,
) -> None:
    """Constructing the public Session DTO is not middleware verification."""
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)

        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="caller supplied a plausible session DTO",
            session=_raw_human_session(),
        ) is False
        task = kb.get_task(conn, gate)
        assert task is not None
        assert task.status == "blocked"
        assert "human_gate_authorized" not in _event_kinds(conn, gate)


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


def test_normal_tasks_skip_execution_fingerprint_work(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="ordinary task")

        def unexpected_fingerprint(*_args, **_kwargs):
            raise AssertionError("ordinary tasks must not build gate fingerprints")

        monkeypatch.setattr(kb, "_human_gate_task_fingerprint", unexpected_fingerprint)
        assert kb._human_gate_state(conn, task_id) == (False, False)


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
        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="approved safe report",
            session=_human_session(),
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

        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="approved changed operation",
            session=_human_session(),
        )
        claimed = kb.claim_task(conn, gate, claimer="worker")
        assert claimed is not None
        assert claimed.body == "perform different operation"


def test_authorization_fingerprint_includes_provider_override(kanban_home: Path) -> None:
    with kb.connect() as conn:
        gate = kb.create_task(
            conn,
            title="provider-bound operation",
            assignee="worker",
            initial_status="blocked",
            model_override="model-a",
            provider_override="provider-a",
        )
        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="approved provider-a/model-a",
            session=_human_session(),
        )

        assert kb.set_model_override(
            conn,
            gate,
            model="model-a",
            provider="provider-b",
        )

        assert kb.claim_task(conn, gate, claimer="dispatcher") is None
        assert kb.get_task(conn, gate).status == "blocked"


@pytest.mark.parametrize("mutation", ["comment", "attachment", "dependency"])
def test_authorization_fingerprint_includes_task_specific_context(
    kanban_home: Path,
    mutation: str,
) -> None:
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="approved exact task context",
            session=_human_session(),
        )

        if mutation == "comment":
            kb.add_comment(conn, gate, "operator", "Use the privileged fallback")
        elif mutation == "attachment":
            kb.add_attachment(
                conn,
                gate,
                filename="instructions.txt",
                stored_path=str(kanban_home / "instructions.txt"),
                content_type="text/plain",
                size=24,
                uploaded_by="operator",
            )
        else:
            parent = kb.create_task(conn, title="new prerequisite")
            claimed = kb.claim_task(conn, parent)
            assert claimed is not None
            assert kb.complete_task(
                conn,
                parent,
                result="dependency result",
                expected_run_id=claimed.current_run_id,
            )
            kb.link_tasks(conn, parent, gate)

        assert kb.claim_task(conn, gate, claimer="dispatcher") is None
        assert kb.get_task(conn, gate).status == "blocked"


def test_authorization_fingerprint_binds_attachment_bytes(
    kanban_home: Path,
) -> None:
    attachment = kanban_home / "approved-instructions.txt"
    attachment.write_bytes(b"approved bytes")
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        kb.add_attachment(
            conn,
            gate,
            filename=attachment.name,
            stored_path=str(attachment),
            content_type="text/plain",
            size=attachment.stat().st_size,
            uploaded_by="operator",
        )
        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="approved attached instructions",
            session=_human_session(),
        )

        sealed = kb.list_attachments(conn, gate)[0]
        assert Path(sealed.stored_path) != attachment
        assert Path(sealed.stored_path).read_bytes() == b"approved bytes"

        # Only the digest-addressed snapshot is worker-visible. Mutating the
        # caller's original path cannot change the authorized input.
        attachment.write_bytes(b"tampered original bytes")

        claimed = kb.claim_task(conn, gate, claimer="dispatcher")
        assert claimed is not None


def test_authorization_fingerprint_relocks_if_sealed_attachment_is_tampered(
    kanban_home: Path,
) -> None:
    attachment = kanban_home / "approved-instructions.txt"
    attachment.write_bytes(b"approved bytes")
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        kb.add_attachment(
            conn,
            gate,
            filename=attachment.name,
            stored_path=str(attachment),
            content_type="text/plain",
            size=attachment.stat().st_size,
            uploaded_by="operator",
        )
        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="approved attached instructions",
            session=_human_session(),
        )

        sealed_path = Path(kb.list_attachments(conn, gate)[0].stored_path)
        sealed_path.chmod(0o644)
        sealed_path.write_bytes(b"tampered sealed bytes")

        assert kb.claim_task(conn, gate, claimer="dispatcher") is None
        task = kb.get_task(conn, gate)
        assert task is not None and task.status == "blocked"


def test_dispatch_revalidates_authorization_after_claim_before_spawn(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    spawned: list[str] = []

    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="approved pre-claim definition",
            session=_human_session(),
        )
        original_materialize = kb._materialize_claim_execution

        def mutate_after_claim(*args, **kwargs):
            materialized = original_materialize(*args, **kwargs)
            if materialized:
                with kb.connect() as attacker_conn:
                    kb.add_comment(
                        attacker_conn,
                        gate,
                        "operator",
                        "Changed after claim but before spawn",
                    )
            return materialized

        monkeypatch.setattr(kb, "_materialize_claim_execution", mutate_after_claim)
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append(task.id),
        )

        assert result.spawned == []
        assert spawned == []
        task = kb.get_task(conn, gate)
        assert task is not None and task.status == "blocked"
        run = conn.execute(
            "SELECT status, outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (gate,),
        ).fetchone()
        assert run is not None
        assert (run["status"], run["outcome"]) == ("blocked", "blocked")


def test_stale_authorization_relock_reaches_notification_cursor(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        kb.add_notify_sub(
            conn,
            task_id=gate,
            platform="telegram",
            chat_id="operator-chat",
        )
        _, initial = kb.unseen_events_for_sub(
            conn,
            task_id=gate,
            platform="telegram",
            chat_id="operator-chat",
            kinds=["blocked"],
        )
        assert [event.kind for event in initial] == ["blocked"]
        kb.advance_notify_cursor(
            conn,
            task_id=gate,
            platform="telegram",
            chat_id="operator-chat",
            new_cursor=initial[-1].id,
        )

        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="approved exact current instructions",
            session=_human_session(),
        )
        kb.add_comment(conn, gate, "operator", "changed after authorization")

        assert kb.claim_task(conn, gate, claimer="dispatcher") is None
        _, relock = kb.unseen_events_for_sub(
            conn,
            task_id=gate,
            platform="telegram",
            chat_id="operator-chat",
            kinds=["blocked"],
        )
        assert [event.kind for event in relock] == ["blocked"]
        assert relock[0].payload == {
            "reason": "human_gate_not_authorized",
            "source": "claim",
        }


def test_dispatch_runtime_materialization_preserves_valid_authorization(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    spawned: list[tuple[str, str]] = []

    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="approved scratch execution",
            session=_human_session(),
        )

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append((task.id, workspace)),
        )

        assert len(result.spawned) == 1
        assert spawned and spawned[0][0] == gate
        assert kb.get_task(conn, gate).status == "running"
        assert "human_gate_materialized" in _event_kinds(conn, gate)


def test_initial_blocked_gate_survives_parent_completion(kanban_home: Path) -> None:
    with kb.connect() as conn:
        gate, parent = _initial_gate(conn, with_parent=True)
        assert kb.get_task(conn, gate).status == "blocked"
        assert kb.complete_task(conn, parent, result="done")
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, gate).status == "blocked"
        assert "promoted" not in _event_kinds(conn, gate)


def test_gate_relocks_and_can_be_reauthorized_after_parent_handoff_changes(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        gate, parent = _initial_gate(conn, with_parent=True)
        assert parent is not None
        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="approved pending parent handoff",
            session=_human_session(),
        )
        pending = kb.get_task(conn, gate)
        assert pending is not None and pending.status == "todo"

        assert kb.complete_task(conn, parent, result="new executable handoff")
        assert kb.recompute_ready(conn) == 0
        relocked = kb.get_task(conn, gate)
        assert relocked is not None and relocked.status == "blocked"

        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="approved final parent handoff",
            session=_human_session(),
        )
        claimed = kb.claim_task(conn, gate, claimer="dispatcher")
        assert claimed is not None


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
def test_raw_actor_labels_cannot_authorize_initial_gate(
    kanban_home: Path, actor, reason
) -> None:
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        before = _event_kinds(conn, gate)
        assert kb.unblock_task(conn, gate, actor=actor, reason=reason) is False
        assert kb.get_task(conn, gate).status == "blocked"
        assert _event_kinds(conn, gate) == before


def test_human_gate_authorization_event_precedes_unblocked_with_verified_principal(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="  User authorized exact SHA abc123 in session s_1  ",
            session=_human_session(),
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
        assert authorization.payload["actor"] == "test:user-123"
        assert authorization.payload["source"] == "dashboard_session"
        assert authorization.payload["provider"] == "test"
        assert authorization.payload["user_id"] == "user-123"
        assert authorization.payload["session_expires_at"] == 9_999_999_999
        assert (
            authorization.payload["reason"]
            == "User authorized exact SHA abc123 in session s_1"
        )
        assert isinstance(authorization.payload["task_fingerprint"], str)
        assert len(authorization.payload["task_fingerprint"]) == 64
        assert unblocked.payload is None


def test_expired_dashboard_session_cannot_authorize_human_gate(
    kanban_home: Path,
) -> None:
    expired = _attest_verified_session(Session(
        user_id="user-123",
        email="operator@example.com",
        display_name="Test Operator",
        provider="test",
        org_id="org-1",
        access_token="expired-token",
        expires_at=1,
        refresh_token="refresh-token",
    ))
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="approval from expired session",
            session=expired,
        ) is False
        task = kb.get_task(conn, gate)
        assert task is not None
        assert task.status == "blocked"
        assert "human_gate_authorized" not in _event_kinds(conn, gate)


def test_authorized_gate_allows_review_claim_when_definition_is_unchanged(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="approved exact reviewable operation",
            session=_human_session(),
        )
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (gate,))
        conn.commit()

        claimed = kb.claim_review_task(conn, gate, claimer="reviewer")
        assert claimed is not None
        assert claimed.status == "running"


def test_authorized_gate_allows_sticky_block_recovery_for_same_definition(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        gate, _ = _initial_gate(conn)
        assert kb.authorize_human_gate(
            conn,
            gate,
            reason="approved exact retryable operation",
            session=_human_session(),
        )
        claimed = kb.claim_task(conn, gate, claimer="worker")
        assert claimed is not None
        assert kb.block_task(
            conn,
            gate,
            reason="transient dependency",
            expected_run_id=claimed.current_run_id,
        )

        assert kb.unblock_task(conn, gate, reason="dependency recovered")
        retried = kb.claim_task(conn, gate, claimer="worker")
        assert retried is not None
        assert retried.status == "running"


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




# ---------------------------------------------------------------------------
# Circuit-breaker blocks still auto-recover (preserve #40c1decb3 intent)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# unblock_task clears the sticky state
# ---------------------------------------------------------------------------


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
