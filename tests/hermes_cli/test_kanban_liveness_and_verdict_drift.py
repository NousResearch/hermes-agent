from __future__ import annotations

import argparse
import json
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
        rows = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ?",
            (f"review-verdict:{review}:{source}:CHANGES_REQUESTED",),
        ).fetchall()
        assert len(rows) == 1
