"""Real SQLite contract tests for opaque Telegram action records."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_actions import (
    ActionAlreadyClaimed,
    ActionAuthorizationError,
    ActionExpired,
    ActionIdempotencyConflict,
    ActionLeaseLost,
    ActionStaleSource,
    ActionUnknown,
    claim_action,
    get_authorized_action,
    issue_action,
    record_action_outcome,
)


ROUTE = {
    "board_identity": "default",
    "profile": "alpha",
    "telegram_principal": 42,
    "origin_chat_id": "-100123",
    "origin_thread_id": "topic-7",
    "origin_message_id": "message-9",
}


def _action_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    db_path = home / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb._INITIALIZED_PATHS.clear()
    kb.init_db(db_path=db_path)
    return db_path


def _task(db_path: Path) -> tuple[str, int, str]:
    with kbc.connect(db_path) as conn:
        task_id = kb.create_task(conn, title="action task")
        task = kb.get_task(conn, task_id)
        assert task is not None
        revision = conn.execute(
            "SELECT MAX(id) AS n FROM task_events WHERE task_id = ?", (task_id,)
        ).fetchone()["n"]
        return task_id, int(revision), task.status


def _issue(db_path: Path, *, idem: str = "choice-1", conflict: str | None = "blocker"):
    task_id, revision, status = _task(db_path)
    with kbc.connect(db_path) as conn:
        record = issue_action(
            conn,
            task_id=task_id,
            expected_revision=revision,
            expected_task_status=status,
            action_kind="ack_blocker",
            action_payload={"choice": "needs_input", "note": "inert"},
            expires_at=500,
            idempotency_key=idem,
            conflict_key=conflict,
            now=100,
            **ROUTE,
        )
    return task_id, revision, status, record


def _claim_kwargs(*, owner: str, now: int = 101) -> dict:
    return {
        **ROUTE,
        "claim_owner": owner,
        "lease_seconds": 20,
        "now": now,
    }


def test_issue_is_opaque_idempotent_and_payload_is_inert(tmp_path, monkeypatch):
    db_path = _action_home(tmp_path, monkeypatch)
    task_id, revision, status, first = _issue(db_path)
    assert len(first.token) >= 20
    assert first.token not in {task_id, "ack_blocker", "choice-1"}
    with kbc.connect(db_path) as conn:
        repeat = issue_action(
            conn,
            task_id=task_id,
            expected_revision=revision,
            expected_task_status=status,
            action_kind="ack_blocker",
            action_payload={"choice": "needs_input", "note": "inert"},
            expires_at=500,
            idempotency_key="choice-1",
            conflict_key="blocker",
            now=100,
            **ROUTE,
        )
        assert repeat.token == first.token
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM kanban_action_records"
        ).fetchone()["n"] == 1
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert [event.kind for event in kb.list_events(conn, task_id)] == ["created"]


def test_idempotency_conflict_and_wrong_actor_never_reveal_result(tmp_path, monkeypatch):
    db_path = _action_home(tmp_path, monkeypatch)
    task_id, revision, status, action = _issue(db_path)
    with kbc.connect(db_path) as conn:
        with pytest.raises(ActionIdempotencyConflict):
            issue_action(
                conn,
                task_id=task_id,
                expected_revision=revision,
                expected_task_status=status,
                action_kind="different_action",
                action_payload={"secret": "do-not-cache-to-caller"},
                expires_at=500,
                idempotency_key="choice-1",
                conflict_key="blocker",
                now=100,
                **ROUTE,
            )
        claim = claim_action(conn, action.token, **_claim_kwargs(owner="handler"))
        assert claim.attempt_id is not None
        record = record_action_outcome(
            conn,
            action.token,
            claim_owner="handler",
            claim_epoch=claim.record.claim_epoch,
            attempt_id=claim.attempt_id,
            outcome_state="completed",
            result={"ok": True, "sensitive": "server-result"},
            now=102,
            **ROUTE,
        )
        assert record.state == "completed"
        wrong = _claim_kwargs(owner="wrong")
        wrong["telegram_principal"] = 99
        with pytest.raises(ActionAuthorizationError) as excinfo:
            claim_action(conn, action.token, **wrong)
        assert "server-result" not in str(excinfo.value)
        repeated = claim_action(conn, action.token, **_claim_kwargs(owner="handler", now=103))
        assert repeated.claimed is False
        assert repeated.record.result == {"ok": True, "sensitive": "server-result"}
        authorized = get_authorized_action(conn, action.token, **ROUTE)
        assert authorized.result == record.result


def test_identity_route_expiry_and_source_revision_fail_closed(tmp_path, monkeypatch):
    db_path = _action_home(tmp_path, monkeypatch)
    task_id, revision, status, action = _issue(db_path)
    with kbc.connect(db_path) as conn:
        with pytest.raises(ActionAuthorizationError):
            copied = _claim_kwargs(owner="copied")
            copied["origin_message_id"] = "other-message"
            claim_action(
                conn, action.token, **copied,
            )
        with pytest.raises(ActionExpired):
            claim_action(conn, action.token, **_claim_kwargs(owner="late", now=500))
        expired = get_authorized_action(conn, action.token, **ROUTE)
        assert expired.state == "expired"

        # A second action bound to the original event becomes stale after a
        # real authoritative event is appended; no callback payload can make
        # it current again.
        fresh = issue_action(
            conn,
            task_id=task_id,
            expected_revision=revision,
            expected_task_status=status,
            action_kind="another_choice",
            action_payload={"choice": "keep_open"},
            expires_at=500,
            idempotency_key="choice-2",
            conflict_key=None,
            now=100,
            **ROUTE,
        )
        kb.add_comment(conn, task_id, "operator", "source changed")
        with pytest.raises(ActionStaleSource):
            claim_action(conn, fresh.token, **_claim_kwargs(owner="stale", now=103))
        stale = get_authorized_action(conn, fresh.token, **ROUTE)
        assert stale.state == "pending"


def test_claimed_expiry_becomes_unknown_and_cannot_be_retried_or_recorded(tmp_path, monkeypatch):
    db_path = _action_home(tmp_path, monkeypatch)
    _, _, _, action = _issue(db_path)
    with kbc.connect(db_path) as conn:
        short_lease = _claim_kwargs(owner="handler", now=101)
        short_lease["lease_seconds"] = 2
        claim = claim_action(conn, action.token, **short_lease)
        assert claim.attempt_id is not None
        with pytest.raises(ActionUnknown):
            claim_action(conn, action.token, **_claim_kwargs(owner="retry", now=104))
        stored = get_authorized_action(conn, action.token, **ROUTE)
        assert stored.state == "unknown"
        with pytest.raises(ActionLeaseLost):
            record_action_outcome(
                conn,
                action.token,
                claim_owner="handler",
                claim_epoch=claim.record.claim_epoch,
                attempt_id=claim.attempt_id,
                outcome_state="completed",
                result={"ok": True},
                now=105,
                **ROUTE,
            )


def test_competing_choices_cannot_both_claim(tmp_path, monkeypatch):
    db_path = _action_home(tmp_path, monkeypatch)
    task_id, revision, status, first = _issue(db_path, idem="choice-a", conflict="same-choice")
    with kbc.connect(db_path) as conn:
        second = issue_action(
            conn,
            task_id=task_id,
            expected_revision=revision,
            expected_task_status=status,
            action_kind="ack_blocker",
            action_payload={"choice": "continue"},
            expires_at=500,
            idempotency_key="choice-b",
            conflict_key="same-choice",
            now=100,
            **ROUTE,
        )
        winner = claim_action(conn, first.token, **_claim_kwargs(owner="winner"))
        loser = claim_action(conn, second.token, **_claim_kwargs(owner="loser", now=102))
        assert winner.claimed is True
        assert loser.claimed is False
        assert loser.record.state == "rejected"
        assert loser.record.result == {"ok": False, "reason": "competing_action_won"}


def test_independent_connections_compete_with_one_action_cas_winner(tmp_path, monkeypatch):
    db_path = _action_home(tmp_path, monkeypatch)
    _, _, _, action = _issue(db_path, conflict=None)

    def worker(owner: str):
        conn = kbc.connect(db_path)
        try:
            result = claim_action(conn, action.token, **_claim_kwargs(owner=owner))
            return result.claimed
        except ActionAlreadyClaimed:
            return False
        finally:
            conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, ("a", "b")))
    assert sum(results) == 1


def test_claim_and_outcome_join_caller_transaction_without_task_transition(tmp_path, monkeypatch):
    db_path = _action_home(tmp_path, monkeypatch)
    task_id, _, _, action = _issue(db_path, conflict=None)
    with kbc.connect(db_path) as conn:
        with pytest.raises(RuntimeError):
            with kb.write_txn(conn):
                claim = claim_action(conn, action.token, **_claim_kwargs(owner="handler"))
                assert claim.attempt_id is not None
                record_action_outcome(
                    conn,
                    action.token,
                    claim_owner="handler",
                    claim_epoch=claim.record.claim_epoch,
                    attempt_id=claim.attempt_id,
                    outcome_state="completed",
                    result={"ok": True},
                    now=102,
                    **ROUTE,
                )
                task = kb.get_task(conn, task_id)
                assert task is not None and task.status == "ready"
                raise RuntimeError("canonical handler rejected its outer transaction")
        stored = get_authorized_action(conn, action.token, **ROUTE)
        assert stored.state == "pending"
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
