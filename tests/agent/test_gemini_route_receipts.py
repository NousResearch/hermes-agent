from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.gemini_route_receipts import GeminiReceiptStore, routing_day_for


UTC = timezone.utc


def make_store(tmp_path: Path) -> GeminiReceiptStore:
    return GeminiReceiptStore(tmp_path / "profile" / "routing" / "gemini-routing.sqlite3")


def prepare(store: GeminiReceiptStore, *, receipt_id: str = "grt_test", started_at=None) -> str:
    return store.prepare_attempt(
        receipt_id=receipt_id,
        parent_session_id="parent",
        parent_turn_id="turn",
        child_session_id="child",
        task_index=0,
        route_requested="auto",
        route_decision="gemini",
        route_reason="eligible_leaf",
        data_classification="standard",
        output_contract="text",
        goal_text="Summarize this",
        context_text="context",
        requested_provider="antigravity-subscription",
        requested_model="gemini-3.8-flash-low",
        requested_effort="low",
        started_at=started_at,
    )


def test_store_creates_exact_schema_and_private_permissions(tmp_path: Path):
    store = make_store(tmp_path)

    assert store.path.exists()
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600

    with sqlite3.connect(store.path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"gemini_attempts", "daily_review_batches", "daily_review_items"} <= tables
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(gemini_attempts)")
        }
        assert {
            "receipt_id", "routing_day", "process_started_at_utc", "goal_sha256",
            "context_sha256", "prompt_sha256", "raw_envelope_json", "fallback_used",
        } <= columns
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() in {"wal", "delete"}


def test_attempt_lifecycle_hashes_and_success_payload(tmp_path: Path):
    store = make_store(tmp_path)
    receipt_id = prepare(store)
    store.mark_process_started(receipt_id, when=datetime(2026, 9, 11, 12, tzinfo=UTC))
    store.complete_attempt(
        receipt_id,
        worker_status="completed",
        response_text="summary",
        process_exit_code=0,
        duration_ms=42,
        conversation_id="conv-1",
        usage={"input": 1},
        raw_envelope={"status": "SUCCESS", "response": "summary"},
    )

    row = store.get_attempt(receipt_id)
    assert row["goal_text"] == "Summarize this"
    assert row["context_text"] == "context"
    assert len(row["goal_sha256"]) == 64
    assert len(row["context_sha256"]) == 64
    assert len(row["prompt_sha256"]) == 64
    assert len(row["response_sha256"]) == 64
    assert row["worker_status"] == "completed"
    assert row["process_started_at_utc"].startswith("2026-09-11T12:00:00")
    assert json.loads(row["usage_json"]) == {"input": 1}
    assert json.loads(row["raw_envelope_json"])["status"] == "SUCCESS"


def test_duplicate_receipt_id_is_rejected_and_terminal_row_cannot_be_rewritten(tmp_path: Path):
    store = make_store(tmp_path)
    prepare(store)
    with pytest.raises(sqlite3.IntegrityError):
        prepare(store)
    store.mark_process_started("grt_test")
    store.complete_attempt("grt_test", worker_status="failed", duration_ms=1, error_code="boom")
    with pytest.raises(ValueError, match="terminal"):
        store.complete_attempt("grt_test", worker_status="completed", duration_ms=2)


def test_started_cohort_includes_success_failure_and_fallback_but_not_unstarted(tmp_path: Path):
    store = make_store(tmp_path)
    started = datetime(2026, 9, 11, 19, tzinfo=UTC)
    for idx, status in enumerate(("completed", "failed", "timeout")):
        rid = prepare(store, receipt_id=f"grt_{idx}", started_at=started)
        store.mark_process_started(rid, when=started)
        store.complete_attempt(
            rid,
            worker_status=status,
            duration_ms=idx + 1,
            fallback_used=status != "completed",
        )
    prepare(store, receipt_id="grt_unstarted", started_at=started)

    rows = store.list_started_attempts_for_day(date(2026, 9, 11))
    assert [row["receipt_id"] for row in rows] == ["grt_0", "grt_1", "grt_2"]
    assert {row["worker_status"] for row in rows} == {"completed", "failed", "timeout"}


def test_routing_day_uses_los_angeles_midnight_and_dst_boundaries():
    assert routing_day_for(datetime(2026, 7, 1, 6, 59, tzinfo=UTC)) == "2026-06-30"
    assert routing_day_for(datetime(2026, 7, 1, 7, 0, tzinfo=UTC)) == "2026-07-01"
    assert routing_day_for(datetime(2026, 1, 1, 7, 59, tzinfo=UTC)) == "2025-12-31"
    assert routing_day_for(datetime(2026, 1, 1, 8, 0, tzinfo=UTC)) == "2026-01-01"
    assert routing_day_for(datetime(2026, 3, 8, 9, 59, tzinfo=UTC)) == "2026-03-08"
    assert routing_day_for(datetime(2026, 11, 1, 8, 30, tzinfo=UTC)) == "2026-11-01"


def test_concurrent_attempt_inserts_are_complete(tmp_path: Path):
    store = make_store(tmp_path)
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def writer(index: int) -> None:
        try:
            barrier.wait(timeout=5)
            prepare(store, receipt_id=f"grt_{index}")
        except BaseException as exc:  # pragma: no cover - diagnostic collection
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert store.count_attempts() == 8


def test_unique_daily_batch_reuses_persisted_seed_and_sample_under_race(tmp_path: Path):
    store = make_store(tmp_path)
    barrier = threading.Barrier(2)
    results: list[dict] = []

    def starter(seed: str) -> None:
        barrier.wait(timeout=5)
        results.append(
            store.create_or_get_review_batch(
                routing_day="2026-09-11",
                timezone_name="America/Los_Angeles",
                sample_size_requested=5,
                eligible_count=6,
                sample_seed_hex=seed,
                sample_receipt_ids=["a", "b", "c", "d", "e"],
            )
        )

    threads = [threading.Thread(target=starter, args=("11" * 32,)), threading.Thread(target=starter, args=("22" * 32,))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(results) == 2
    assert results[0]["batch_id"] == results[1]["batch_id"]
    assert results[0]["sample_seed_hex"] == results[1]["sample_seed_hex"]
    assert store.count_review_batches() == 1


def test_review_items_are_append_only(tmp_path: Path):
    store = make_store(tmp_path)
    prepare(store)
    store.mark_process_started("grt_test")
    store.complete_attempt("grt_test", worker_status="completed", response_text="ok", duration_ms=1)
    batch = store.create_or_get_review_batch(
        routing_day=store.get_attempt("grt_test")["routing_day"],
        timezone_name="America/Los_Angeles",
        sample_size_requested=5,
        eligible_count=1,
        sample_seed_hex="ab" * 32,
        sample_receipt_ids=["grt_test"],
    )
    store.add_review_item(
        batch_id=batch["batch_id"],
        receipt_id="grt_test",
        ordinal=0,
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        review_status="completed",
        verdict="pass",
        reason="Correct summary.",
        review_json={"verdict": "pass", "reason": "Correct summary."},
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.add_review_item(
            batch_id=batch["batch_id"], receipt_id="grt_test", ordinal=0,
            reviewer_provider="openai-codex", reviewer_model="gpt-5.6-sol",
            review_status="completed", verdict="fail", reason="rewrite",
        )
    items = store.list_review_items(batch["batch_id"])
    assert len(items) == 1
    assert items[0]["verdict"] == "pass"


def test_retention_redacts_raw_text_before_deleting_hashes(tmp_path: Path):
    store = make_store(tmp_path)
    old = datetime.now(UTC) - timedelta(days=31)
    prepare(store, started_at=old)
    store.mark_process_started("grt_test", when=old)
    store.complete_attempt(
        "grt_test", worker_status="completed", response_text="secret raw text",
        duration_ms=1, completed_at=old,
    )

    outcome = store.apply_retention(now=datetime.now(UTC), raw_days=30, aggregate_days=180)
    row = store.get_attempt("grt_test")
    assert outcome["raw_redacted"] == 1
    assert row["goal_text"] == ""
    assert row["context_text"] == ""
    assert row["response_text"] is None
    assert row["goal_sha256"]
    assert row["response_sha256"]


def test_retention_redacts_raw_reviewer_prose_at_raw_cutoff(tmp_path: Path):
    store = make_store(tmp_path)
    old = datetime.now(UTC) - timedelta(days=31)
    prepare(store, started_at=old)
    batch = store.create_or_get_review_batch(
        routing_day="2026-08-01",
        timezone_name="America/Los_Angeles",
        sample_size_requested=1,
        eligible_count=1,
        sample_seed_hex="11" * 32,
        sample_receipt_ids=["grt_test"],
        started_at=old,
    )
    store.add_review_item(
        batch_id=batch["batch_id"],
        receipt_id="grt_test",
        ordinal=0,
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        review_status="completed",
        verdict="fail",
        reason="PRIVATE REVIEW REASON",
        review_json={
            "verdict": "fail",
            "reason": "PRIVATE REVIEW REASON",
            "failure_kind": "correctness",
        },
    )

    outcome = store.apply_retention(now=datetime.now(UTC), raw_days=30, aggregate_days=180)
    item = store.list_review_items(batch["batch_id"])[0]

    assert outcome["review_raw_redacted"] == 1
    assert item["reason"] == ""
    assert item["review_json"] is None
    assert item["verdict"] == "fail"
    assert item["review_sha256"]


def test_receipt_store_fails_closed_when_private_modes_cannot_be_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(os, "chmod", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("denied")))

    with pytest.raises(RuntimeError, match="private permissions"):
        GeminiReceiptStore(tmp_path / "routing" / "routing.sqlite3")


def test_read_methods_use_read_only_sqlite_connections(tmp_path: Path, monkeypatch):
    store = make_store(tmp_path)
    prepare(store)
    real_connect = sqlite3.connect
    calls: list[tuple[object, dict]] = []

    def recording_connect(database, *args, **kwargs):
        calls.append((database, kwargs.copy()))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", recording_connect)
    assert store.get_attempt("grt_test")["receipt_id"] == "grt_test"
    assert any(str(database).startswith("file:") and "mode=ro" in str(database) and kwargs.get("uri") for database, kwargs in calls)
