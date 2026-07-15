from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.ship_crew_evidence import (
    build_evidence,
    evidence_sha256,
    write_evidence_artifact,
)
from hermes_cli.ship_crew_execution import (
    BudgetExceededError,
    compact_context,
    accept_output,
    resolve_execution_budget,
    retry_allowed,
    review_allowed,
)
from hermes_cli.ship_crew_notifications import (
    NotificationPolicy,
    notification_key,
    render_notification,
)
from hermes_cli.ship_crew_quota import (
    quota_available,
    quota_db_path,
    quota_state,
    record_quota_failure,
    record_quota_success,
    reset_quota,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_budgets_compact_and_bound_execution():
    budget = resolve_execution_budget("T2", {"max_output_chars": 10, "max_retries": 2})
    compacted = compact_context("0123456789" * 10, 40)
    assert len(compacted) <= 40
    assert "compacted" in compacted
    assert retry_allowed(0, budget)
    assert not retry_allowed(2, budget)
    assert review_allowed(0, budget)
    assert not review_allowed(2, resolve_execution_budget("T2", {"max_review_attempts": 2}))
    with pytest.raises(BudgetExceededError):
        accept_output("01234567890", budget)


def test_quota_circuit_is_shared_and_resets(kanban_home):
    with kb.connect_closing() as conn:
        assert quota_available(conn, "provider:model")
        assert quota_db_path().name == "ship-crew-quota.db"
        state = record_quota_failure(
            conn, "provider:model", threshold=2, retry_after_seconds=60, now=100
        )
        assert not state.open
        state = record_quota_failure(
            conn, "provider:model", threshold=2, retry_after_seconds=60, now=100
        )
        assert state.open_until == 160
        assert not quota_available(conn, "provider:model", now=120)
        assert quota_available(conn, "provider:model", now=160)
        reset = record_quota_success(conn, "provider:model", now=161)
        assert reset.failure_count == 0
        assert quota_state(conn, "provider:model").open_until == 0
        record_quota_failure(conn, "provider:model", threshold=1, retry_after_seconds=60, now=200)
        manual = reset_quota(conn, "provider:model", actor="captain", now=201)
        assert manual.failure_count == 0
        with sqlite3.connect(quota_db_path()) as quota_conn:
            assert quota_conn.execute(
                "SELECT COUNT(*) FROM ship_crew_quota_events WHERE event='manual_reset'"
            ).fetchone()[0] == 1


def test_evidence_is_canonical_redacted_and_atomic(tmp_path):
    record = build_evidence(
        task_id="t1",
        contract_version="1.0",
        outcome="completed",
        role="engineer",
        inputs={"prompt": "safe", "api_key": "do-not-write"},
        outputs={"summary": "done"},
        provenance={"executor": "local"},
    )
    assert record["inputs"]["api_key"] == "[REDACTED]"
    assert record["evidence_sha256"] == evidence_sha256({k: v for k, v in record.items() if k != "evidence_sha256"})
    path = write_evidence_artifact(tmp_path, record, "t1.json")
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert "do-not-write" not in path.read_text(encoding="utf-8")


def test_notifications_are_role_aware_and_deduplicable():
    event = {
        "task_id": "t1",
        "kind": "completed",
        "role": "engineer",
        "outcome": "completed",
        "evidence_sha256": "a" * 64,
        "summary": "ok",
    }
    rendered = render_notification(event)
    assert rendered and "t1" in rendered
    assert render_notification({**event, "role": "worker"}) == ""
    assert notification_key(task_id="t1", event_kind="completed", event_id=3) == notification_key(task_id="t1", event_kind="completed", event_id=3)
    assert render_notification(event, NotificationPolicy(max_chars=10)).endswith("…")
