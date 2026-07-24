from __future__ import annotations

import argparse
from pathlib import Path

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_liveness


def _isolated_home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_liveness_subcommand_is_registered() -> None:
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)

    args = parser.parse_args([
        "kanban",
        "liveness",
        "--tenant",
        "molly-v2-working-system",
        "--ready-sla-minutes",
        "15",
        "--json",
    ])

    assert args.command == "kanban"
    assert args.kanban_action == "liveness"
    assert args.tenant == "molly-v2-working-system"
    assert args.ready_sla_minutes == 15
    assert args.json is True


def test_liveness_scanner_reports_blocked_tenant_task(tmp_path: Path, monkeypatch) -> None:
    _isolated_home(tmp_path, monkeypatch)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="blocked lane",
            assignee="stark",
            tenant="molly-v2-working-system",
            initial_status="blocked",
        )
        payload = kanban_liveness.scan_liveness(
            conn,
            tenant="molly-v2-working-system",
            ready_sla_minutes=240,
            now=2_000_000,
        )

    assert payload["schema"] == "hermes.kanban_liveness.v1"
    assert payload["summary"]["health"] == "warning"
    assert payload["summary"]["finding_count"] == 1
    assert payload["findings"][0]["task_id"] == task_id
    assert payload["findings"][0]["kind"] == "blocked_task"


def test_consume_review_verdicts_approved_unblocks_source_once(tmp_path: Path, monkeypatch) -> None:
    _isolated_home(tmp_path, monkeypatch)
    with kb.connect_closing() as conn:
        source = kb.create_task(
            conn,
            title="source needing review",
            assignee="stark",
            tenant="molly-v2-working-system",
        )
        assert kb.block_task(conn, source, reason="review-required: needs independent review")
        review = kb.create_task(
            conn,
            title="review source",
            assignee="brennan",
            tenant="molly-v2-working-system",
        )
        assert kb.complete_task(
            conn,
            review,
            summary="APPROVED: looks good",
            metadata={
                "schema": "hermes.review_verdict.v1",
                "review_of": source,
                "verdict": "APPROVED",
                "approved": True,
            },
        )

        result = kb.consume_review_verdicts(conn)
        assert result.consumed_review_verdicts == [source]
        assert result.created_review_verdict_followups == []
        source_task = kb.get_task(conn, source)
        assert source_task is not None
        assert source_task.status == "ready"

        second = kb.consume_review_verdicts(conn)
        assert second.consumed_review_verdicts == []
        events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? AND kind = 'review_verdict_consumed'",
            (review,),
        ).fetchall()
        assert len(events) == 1


def test_consume_review_verdicts_approved_ignores_negated_red_human_prose(tmp_path: Path, monkeypatch) -> None:
    _isolated_home(tmp_path, monkeypatch)
    with kb.connect_closing() as conn:
        source = kb.create_task(
            conn,
            title="source with negated red prose",
            assignee="stark",
            tenant="molly-v2-working-system",
        )
        assert kb.block_task(
            conn,
            source,
            reason="review-required: no red gate remains; no human approval required",
        )
        with kb.write_txn(conn):
            kb._append_event(
                conn,
                source,
                "blocked",
                {"reason": "review-required", "gate": "yellow", "owner_hint": "stark"},
            )
        review = kb.create_task(conn, title="review source", assignee="brennan")
        assert kb.complete_task(
            conn,
            review,
            summary="APPROVED: no typed hold",
            metadata={"review_of": source, "verdict": "APPROVED"},
        )

        result = kb.consume_review_verdicts(conn)

        assert result.consumed_review_verdicts == [source]
        assert result.created_review_verdict_followups == []
        source_task = kb.get_task(conn, source)
        assert source_task is not None
        assert source_task.status == "ready"


def test_consume_review_verdicts_approved_preserves_typed_red_human_hold(tmp_path: Path, monkeypatch) -> None:
    _isolated_home(tmp_path, monkeypatch)
    with kb.connect_closing() as conn:
        source = kb.create_task(
            conn,
            title="source with typed red hold",
            assignee="stark",
            tenant="molly-v2-working-system",
        )
        assert kb.block_task(conn, source, reason="review-required: needs review")
        with kb.write_txn(conn):
            kb._append_event(
                conn,
                source,
                "blocked",
                {"reason": "review-required", "gate": "red", "human_approval_required": True},
            )
        review = kb.create_task(conn, title="review source", assignee="brennan")
        assert kb.complete_task(
            conn,
            review,
            summary="APPROVED: code is fine but red hold remains",
            metadata={"review_of": source, "verdict": "APPROVED"},
        )

        result = kb.consume_review_verdicts(conn)

        assert result.consumed_review_verdicts == [source]
        assert len(result.created_review_verdict_followups) == 1
        source_task = kb.get_task(conn, source)
        assert source_task is not None
        assert source_task.status == "blocked"
        packet = kb.get_task(conn, result.created_review_verdict_followups[0])
        assert packet is not None
        assert packet.status == "blocked"
        assert packet.assignee == "default"
        assert packet.idempotency_key == f"review-verdict:{source}:red-human-hold"


def test_consume_review_verdicts_composite_changes_requested_stop_routes_red_human(tmp_path: Path, monkeypatch) -> None:
    _isolated_home(tmp_path, monkeypatch)
    with kb.connect_closing() as conn:
        source = kb.create_task(
            conn,
            title="source with composite stop review",
            assignee="stark",
            tenant="molly-v2-working-system",
        )
        assert kb.block_task(conn, source, reason="review-required: needs independent review")
        review = kb.create_task(conn, title="review source", assignee="brennan")
        assert kb.complete_task(
            conn,
            review,
            summary="CHANGES REQUESTED / STOP: human approval required before unblock.",
            metadata={
                "schema": "hermes.review_verdict.v1",
                "review_of": source,
                "verdict": "CHANGES_REQUESTED",
            },
        )

        result = kb.consume_review_verdicts(conn)

        assert result.consumed_review_verdicts == [source]
        assert len(result.created_review_verdict_followups) == 1
        packet = kb.get_task(conn, result.created_review_verdict_followups[0])
        assert packet is not None
        assert packet.assignee == "default"
        assert packet.status == "blocked"
        assert packet.idempotency_key == f"review-verdict:{source}:red-human-hold"
        assert "verdict=STOP_RED_HUMAN" in (packet.body or "")
        source_task = kb.get_task(conn, source)
        assert source_task is not None
        assert source_task.status == "blocked"


def test_consume_review_verdicts_changes_requested_creates_one_followup(tmp_path: Path, monkeypatch) -> None:
    _isolated_home(tmp_path, monkeypatch)
    with kb.connect_closing() as conn:
        source = kb.create_task(
            conn,
            title="source needing changes",
            assignee="stark",
            tenant="molly-v2-working-system",
        )
        assert kb.block_task(conn, source, reason="review-required: needs changes review")
        review = kb.create_task(
            conn,
            title="review source",
            assignee="brennan",
            tenant="molly-v2-working-system",
        )
        assert kb.complete_task(
            conn,
            review,
            summary="CHANGES_REQUESTED: add tests",
            metadata={
                "schema": "hermes.review_verdict.v1",
                "subject_task_id": source,
                "verdict": "CHANGES_REQUESTED",
                "summary": "add tests",
            },
        )

        result = kb.consume_review_verdicts(conn)
        assert result.consumed_review_verdicts == [source]
        assert len(result.created_review_verdict_followups) == 1
        followup = kb.get_task(conn, result.created_review_verdict_followups[0])
        assert followup is not None
        assert followup.status == "blocked"
        assert followup.assignee == "stark"
        assert "source_task_id=" + source in (followup.body or "")
        source_task = kb.get_task(conn, source)
        assert source_task is not None
        assert source_task.status == "blocked"

        second = kb.consume_review_verdicts(conn)
        assert second.created_review_verdict_followups == []
        review_run = conn.execute(
            "SELECT id FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (review,),
        ).fetchone()
        assert review_run is not None
        rows = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ?",
            (f"review-verdict:{source}:{review_run['id']}:changes_requested",),
        ).fetchall()
        assert len(rows) == 1


def test_create_task_rechecks_idempotency_inside_write_transaction(tmp_path: Path, monkeypatch) -> None:
    _isolated_home(tmp_path, monkeypatch)
    key = "review-verdict:t_source:2:changes_requested"
    original_new_task_id = kb._new_task_id
    injected = {"done": False}

    def racing_new_task_id() -> str:
        if not injected["done"]:
            injected["done"] = True
            monkeypatch.setattr(kb, "_new_task_id", original_new_task_id)
            with kb.connect_closing() as contender:
                kb.create_task(
                    contender,
                    title="concurrent winner",
                    assignee="stark",
                    idempotency_key=key,
                )
            monkeypatch.setattr(kb, "_new_task_id", racing_new_task_id)
        return original_new_task_id()

    monkeypatch.setattr(kb, "_new_task_id", racing_new_task_id)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="late loser",
            assignee="stark",
            idempotency_key=key,
        )
        rows = conn.execute(
            "SELECT id, title FROM tasks WHERE idempotency_key = ? ORDER BY created_at ASC, id ASC",
            (key,),
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["title"] == "concurrent winner"
    assert task_id == rows[0]["id"]
