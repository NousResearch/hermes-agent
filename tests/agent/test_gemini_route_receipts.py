from __future__ import annotations

import hashlib
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
            "terminal_worker_route", "terminal_provider", "terminal_model",
            "terminal_worker_status",
            "terminal_response_text", "terminal_response_sha256",
            "terminal_error_code",
        } <= columns
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() in {"wal", "delete"}


def test_legacy_schema_migration_holds_immediate_write_transaction(tmp_path: Path):
    db_path = tmp_path / "profile" / "routing" / "gemini-routing.sqlite3"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE daily_review_batches (batch_id TEXT PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE daily_review_items "
            "(batch_id TEXT NOT NULL, receipt_id TEXT NOT NULL)"
        )

    statements: list[str] = []

    class TracedStore(GeminiReceiptStore):
        def _open_write(self):
            conn = super()._open_write()
            conn.set_trace_callback(statements.append)
            return conn

    TracedStore(db_path)

    normalized = [statement.strip().upper() for statement in statements]
    first_alter = next(i for i, statement in enumerate(normalized) if statement.startswith("ALTER TABLE"))
    assert "BEGIN IMMEDIATE" in normalized[:first_alter]


def test_concurrent_connections_upgrade_one_legacy_schema(tmp_path: Path):
    db_path = tmp_path / "profile" / "routing" / "gemini-routing.sqlite3"
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE gemini_attempts "
            "(receipt_id TEXT PRIMARY KEY, routing_day TEXT NOT NULL, "
            "route_decision TEXT NOT NULL)"
        )
        conn.execute("CREATE TABLE daily_review_batches (batch_id TEXT PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE daily_review_items "
            "(batch_id TEXT NOT NULL, receipt_id TEXT NOT NULL)"
        )

    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def open_store() -> None:
        try:
            barrier.wait(timeout=5)
            GeminiReceiptStore(db_path)
        except BaseException as exc:  # pragma: no cover - diagnostic collection
            errors.append(exc)

    threads = [threading.Thread(target=open_store) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    with sqlite3.connect(db_path) as conn:
        batch_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(daily_review_batches)")
        }
        item_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(daily_review_items)")
        }
        attempt_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(gemini_attempts)")
        }
    assert {
        "alert_delivery_key",
        "alert_message",
        "slack_workspace_id",
        "alert_lease_token",
        "alert_lease_started_at_utc",
        "review_lease_token",
        "review_lease_started_at_utc",
    } <= batch_columns
    assert "failure_kind" in item_columns
    assert {
        "terminal_worker_route",
        "terminal_provider",
        "terminal_model",
        "terminal_worker_status",
        "terminal_response_text",
        "terminal_response_sha256",
        "terminal_error_code",
    } <= attempt_columns


def test_store_retries_locked_journal_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import agent.gemini_route_receipts as receipts_module

    real_apply = receipts_module.apply_wal_with_fallback
    attempts = 0

    def locked_once(conn, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_apply(conn, **kwargs)

    monkeypatch.setattr(receipts_module, "apply_wal_with_fallback", locked_once)

    make_store(tmp_path)

    assert attempts == 2


def test_stale_reviewer_cannot_overwrite_terminal_batch_or_outbox(tmp_path: Path):
    store = make_store(tmp_path)
    old = datetime(2026, 9, 11, 12, tzinfo=UTC)
    batch = store.create_or_get_review_batch(
        routing_day="2026-09-11",
        timezone_name="America/Los_Angeles",
        sample_size_requested=5,
        eligible_count=0,
        sample_seed_hex="11" * 32,
        sample_receipt_ids=[],
        started_at=old,
    )
    review_lease = "stale-runner"
    assert store.claim_review_batch(
        batch["batch_id"], lease_token=review_lease, now=old
    )
    assert store.fail_stale_review_batch(
        batch["batch_id"],
        stale_before=old + timedelta(seconds=1),
        pipeline_error="stale_review_lease",
        alert_message="authoritative stale-lease alert",
        slack_channel_id="C_AUTH",
        slack_workspace_id="T_AUTH",
        completed_at=old + timedelta(seconds=2),
    )

    with pytest.raises(KeyError):
        store.update_review_batch(
            batch["batch_id"],
            lease_token=review_lease,
            status="passed",
            alert_status="not_needed",
            alert_message="late runner payload",
            slack_channel_id="C_LATE",
            slack_workspace_id="T_LATE",
            completed_at=old + timedelta(seconds=3),
        )

    current = store.get_review_batch("2026-09-11")
    assert current is not None
    assert current["status"] == "pipeline_failed"
    assert current["pipeline_error"] == "stale_review_lease"
    assert current["alert_message"] == "authoritative stale-lease alert"
    assert current["slack_channel_id"] == "C_AUTH"
    assert current["slack_workspace_id"] == "T_AUTH"


def test_stale_reviewer_cannot_append_item_after_lease_terminalization(tmp_path: Path):
    store = make_store(tmp_path)
    old = datetime(2026, 9, 11, 12, tzinfo=UTC)
    receipt_id = prepare(store, receipt_id="grt_stale_item", started_at=old)
    batch = store.create_or_get_review_batch(
        routing_day="2026-09-11",
        timezone_name="America/Los_Angeles",
        sample_size_requested=1,
        eligible_count=1,
        sample_seed_hex="13" * 32,
        sample_receipt_ids=[receipt_id],
        started_at=old,
    )
    review_lease = "stale-runner"
    assert store.claim_review_batch(
        batch["batch_id"], lease_token=review_lease, now=old
    )
    assert store.fail_stale_review_batch(
        batch["batch_id"],
        stale_before=old + timedelta(seconds=1),
        pipeline_error="stale_review_lease",
        alert_message="authoritative stale-lease alert",
        slack_channel_id="C_AUTH",
        slack_workspace_id="T_AUTH",
        completed_at=old + timedelta(seconds=2),
    )

    with pytest.raises(KeyError, match="review authority lost"):
        store.add_review_item(
            batch_id=batch["batch_id"],
            lease_token=review_lease,
            receipt_id=receipt_id,
            ordinal=0,
            reviewer_provider="openai-codex",
            reviewer_model="gpt-5.6-sol",
            review_status="completed",
            verdict="pass",
            failure_kind="none",
            reason="late reviewer payload",
            review_json={
                "verdict": "pass",
                "reason": "late reviewer payload",
                "failure_kind": "none",
            },
            completed_at=old + timedelta(seconds=3),
        )

    assert store.list_review_items(batch["batch_id"]) == []


def test_review_batch_update_requires_current_lease_token(tmp_path: Path):
    store = make_store(tmp_path)
    now = datetime(2026, 9, 11, 12, tzinfo=UTC)
    batch = store.create_or_get_review_batch(
        routing_day="2026-09-11",
        timezone_name="America/Los_Angeles",
        sample_size_requested=0,
        eligible_count=0,
        sample_seed_hex="12" * 32,
        sample_receipt_ids=[],
        started_at=now,
    )
    assert store.claim_review_batch(
        batch["batch_id"], lease_token="runner-a", now=now
    )
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE daily_review_batches SET review_lease_token=? WHERE batch_id=?",
            ("runner-b", batch["batch_id"]),
        )

    with pytest.raises(KeyError):
        store.update_review_batch(
            batch["batch_id"],
            lease_token="runner-a",
            status="passed",
            alert_status="not_needed",
            completed_at=now,
        )
    store.update_review_batch(
        batch["batch_id"],
        lease_token="runner-b",
        status="passed",
        alert_status="not_needed",
        completed_at=now,
    )

    current = store.get_review_batch("2026-09-11")
    assert current is not None
    assert current["status"] == "passed"


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
    assert row["goal_text"] == ""
    assert row["context_text"] == ""
    assert len(row["goal_sha256"]) == 64
    assert len(row["context_sha256"]) == 64
    assert len(row["prompt_sha256"]) == 64
    assert len(row["response_sha256"]) == 64
    assert row["response_text"] is None
    assert row["conversation_id"] is None
    assert row["raw_envelope_json"] is None
    assert row["worker_status"] == "completed"
    assert row["process_started_at_utc"].startswith("2026-09-11T12:00:00")
    assert json.loads(row["usage_json"]) == {"input": 1}
    assert row["raw_envelope_json"] is None


@pytest.mark.parametrize(
    "gemini_status",
    ["timeout", "cancelled", "malformed", "denied", "oversized"],
)
def test_fallback_outcome_records_after_every_terminal_gemini_failure(
    tmp_path: Path, gemini_status: str
):
    store = make_store(tmp_path)
    receipt_id = prepare(store)
    store.mark_process_started(receipt_id)
    store.complete_attempt(
        receipt_id,
        worker_status=gemini_status,
        duration_ms=1,
        fallback_used=True,
    )
    pending_fallback = store.get_attempt(receipt_id)
    assert pending_fallback["terminal_worker_route"] is None
    assert pending_fallback["terminal_provider"] is None
    assert pending_fallback["terminal_model"] is None
    assert pending_fallback["terminal_worker_status"] is None

    store.record_fallback_outcome(
        receipt_id,
        worker_route="sol",
        provider="openai-codex",
        model="gpt-5.6-sol",
        worker_status="completed",
        response_text="fallback",
        error_code=None,
    )

    row = store.get_attempt(receipt_id)
    assert row is not None
    assert row["terminal_worker_route"] == "sol"
    assert row["terminal_worker_status"] == "completed"


def test_fallback_outcome_cannot_be_rewritten(tmp_path: Path):
    store = make_store(tmp_path)
    receipt_id = prepare(store)
    store.mark_process_started(receipt_id)
    store.complete_attempt(
        receipt_id,
        worker_status="failed",
        duration_ms=1,
        fallback_used=True,
    )
    store.record_fallback_outcome(
        receipt_id,
        worker_route="sol",
        provider="openai-codex",
        model="gpt-5.6-sol",
        worker_status="completed",
        response_text="authoritative",
        error_code=None,
    )

    with pytest.raises(KeyError):
        store.record_fallback_outcome(
            receipt_id,
            worker_route="sol",
            provider="anthropic",
            model="other-model",
            worker_status="failed",
            response_text=None,
            error_code="late",
        )

    row = store.get_attempt(receipt_id)
    assert row is not None
    assert row["terminal_provider"] == "openai-codex"
    assert row["terminal_response_text"] is None
    assert row["terminal_response_sha256"] == hashlib.sha256(b"authoritative").hexdigest()


def test_schema_upgrade_clears_legacy_provisional_gemini_terminal_truth(tmp_path: Path):
    store = make_store(tmp_path)
    receipt_id = prepare(store)
    store.mark_process_started(receipt_id)
    store.complete_attempt(
        receipt_id,
        worker_status="failed",
        duration_ms=1,
        fallback_used=True,
    )
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            """UPDATE gemini_attempts
               SET terminal_worker_route='gemini',
                   terminal_provider=requested_provider,
                   terminal_model=requested_model,
                   terminal_worker_status='failed'
               WHERE receipt_id=?""",
            (receipt_id,),
        )

    upgraded = GeminiReceiptStore(store.path).get_attempt(receipt_id)

    assert upgraded["terminal_worker_route"] is None
    assert upgraded["terminal_provider"] is None
    assert upgraded["terminal_model"] is None
    assert upgraded["terminal_worker_status"] is None


def test_schema_upgrade_removes_legacy_attempt_evidence_but_keeps_hashes(tmp_path: Path):
    store = make_store(tmp_path)
    receipt_id = prepare(store)
    complete_response = "🔥" * 25_000
    complete_error = "private-error-" * 10_000
    raw_envelope = json.dumps({"status": "SUCCESS", "response": complete_response})
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            """UPDATE gemini_attempts
               SET response_text=?, response_sha256=NULL, response_bytes=NULL,
                   error_message=?, error_message_sha256=NULL, error_message_bytes=NULL,
                   raw_envelope_json=?, terminal_response_text=?,
                   terminal_response_sha256=NULL, terminal_response_bytes=NULL
               WHERE receipt_id=?""",
            (
                complete_response,
                complete_error,
                raw_envelope,
                complete_response,
                receipt_id,
            ),
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS gemini_schema_migrations (
                   name TEXT PRIMARY KEY,
                   applied_at_utc TEXT NOT NULL
               )"""
        )
        conn.execute(
            "DELETE FROM gemini_schema_migrations WHERE name='bound_attempt_evidence_v1'"
        )

    upgraded = GeminiReceiptStore(store.path).get_attempt(receipt_id)

    assert upgraded["response_text"] is None
    assert upgraded["response_sha256"] == hashlib.sha256(
        complete_response.encode("utf-8")
    ).hexdigest()
    assert upgraded["response_bytes"] == len(complete_response.encode("utf-8"))
    assert upgraded["error_message"] is None
    assert upgraded["error_message_sha256"] == hashlib.sha256(
        complete_error.encode("utf-8")
    ).hexdigest()
    assert upgraded["error_message_bytes"] == len(complete_error.encode("utf-8"))
    assert upgraded["terminal_response_text"] is None
    assert upgraded["terminal_response_sha256"] == hashlib.sha256(
        complete_response.encode("utf-8")
    ).hexdigest()
    assert upgraded["terminal_response_bytes"] == len(complete_response.encode("utf-8"))
    assert upgraded["raw_envelope_json"] is None


def test_attempt_evidence_is_metadata_only_with_complete_hashes_and_byte_counts(tmp_path: Path):
    store = make_store(tmp_path)
    receipt_id = prepare(store)
    store.mark_process_started(receipt_id)
    complete_response = "🔥" * 50_000
    complete_error = "private-error-" * 20_000

    store.complete_attempt(
        receipt_id,
        worker_status="failed",
        duration_ms=1,
        response_text=complete_response,
        error_code="worker_failed",
        error_message=complete_error,
        raw_envelope={
            "status": "ERROR",
            "response": complete_response,
            "diagnostic": "x" * 100_000,
        },
    )

    row = store.get_attempt(receipt_id)
    assert row["response_text"] is None
    assert row["response_sha256"] == hashlib.sha256(
        complete_response.encode("utf-8")
    ).hexdigest()
    assert row["response_bytes"] == len(complete_response.encode("utf-8"))
    assert row["error_message"] is None
    assert row["error_message_sha256"] == hashlib.sha256(
        complete_error.encode("utf-8")
    ).hexdigest()
    assert row["error_message_bytes"] == len(complete_error.encode("utf-8"))
    assert row["raw_envelope_json"] is None


def test_receipt_and_review_prompt_never_persist_or_reproduce_sensitive_payloads(
    tmp_path: Path,
):
    from agent.gemini_daily_review import build_review_prompt

    store = make_store(tmp_path)
    receipt_id = store.prepare_attempt(
        receipt_id="grt_private",
        parent_session_id="parent",
        parent_turn_id="turn",
        child_session_id="child",
        task_index=0,
        route_requested="auto",
        route_decision="gemini",
        route_reason="eligible_leaf",
        data_classification="standard",
        output_contract="text",
        goal_text="PROMPT_SECRET",
        context_text="CONNECTION_STRING_SECRET",
        requested_provider="antigravity-subscription",
        requested_model="gemini-3.8-flash-low",
        requested_effort="low",
    )
    store.mark_process_started(receipt_id)
    store.complete_attempt(
        receipt_id,
        worker_status="failed",
        response_text="OUTPUT_SECRET",
        duration_ms=1,
        conversation_id="CONVERSATION_SECRET",
        raw_envelope={"diagnostic": "ENVELOPE_SECRET"},
        error_code="worker_failed",
        error_message="PROVIDER_ERROR_SECRET",
    )

    row = store.get_attempt(receipt_id)
    prompt = build_review_prompt(row)
    forbidden = {
        "PROMPT_SECRET",
        "CONNECTION_STRING_SECRET",
        "OUTPUT_SECRET",
        "CONVERSATION_SECRET",
        "ENVELOPE_SECRET",
        "PROVIDER_ERROR_SECRET",
    }
    assert all(secret not in json.dumps(row, sort_keys=True) for secret in forbidden)
    assert all(secret not in prompt for secret in forbidden)
    assert row["goal_sha256"] == hashlib.sha256(b"PROMPT_SECRET").hexdigest()
    assert row["response_sha256"] == hashlib.sha256(b"OUTPUT_SECRET").hexdigest()


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
    review_lease = "append-only-reviewer"
    assert store.claim_review_batch(
        batch["batch_id"], lease_token=review_lease, now=datetime.now(UTC)
    )
    store.add_review_item(
        batch_id=batch["batch_id"],
        lease_token=review_lease,
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
            batch_id=batch["batch_id"], lease_token=review_lease,
            receipt_id="grt_test", ordinal=0,
            reviewer_provider="openai-codex", reviewer_model="gpt-5.6-sol",
            review_status="completed", verdict="fail", reason="rewrite",
        )
    items = store.list_review_items(batch["batch_id"])
    assert len(items) == 1
    assert items[0]["verdict"] == "pass"
    assert items[0]["reason"] is None
    assert items[0]["review_json"] is None
    assert items[0]["review_sha256"] == hashlib.sha256(
        json.dumps(
            {"verdict": "pass", "reason": "Correct summary."},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_retention_finds_no_attempt_payload_after_metadata_only_writes(tmp_path: Path):
    store = make_store(tmp_path)
    old = datetime.now(UTC) - timedelta(days=31)
    prepare(store, started_at=old)
    store.mark_process_started("grt_test", when=old)
    store.complete_attempt(
        "grt_test", worker_status="completed", response_text="secret raw text",
        duration_ms=1, fallback_used=True, completed_at=old,
    )
    store.record_fallback_outcome(
        "grt_test",
        worker_route="sol",
        provider="openai-codex",
        model="gpt-5.6-sol",
        worker_status="completed",
        response_text="secret fallback text",
        error_code=None,
    )

    outcome = store.apply_retention(now=datetime.now(UTC), raw_days=30, aggregate_days=180)
    row = store.get_attempt("grt_test")
    assert outcome["raw_redacted"] == 0
    assert row["goal_text"] == ""
    assert row["context_text"] == ""
    assert row["response_text"] is None
    assert row["terminal_response_text"] is None
    assert row["goal_sha256"]
    assert row["response_sha256"]
    assert row["terminal_response_sha256"]


def test_reviewer_prose_is_absent_before_retention_runs(tmp_path: Path):
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
    review_lease = "retention-reviewer"
    assert store.claim_review_batch(
        batch["batch_id"], lease_token=review_lease, now=old
    )
    store.add_review_item(
        batch_id=batch["batch_id"],
        lease_token=review_lease,
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

    assert outcome["review_raw_redacted"] == 0
    assert item["reason"] is None
    assert item["review_json"] is None
    assert item["error_message"] is None
    assert item["review_sha256"]


def test_receipt_store_fails_closed_when_private_modes_cannot_be_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(os, "chmod", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("denied")))

    with pytest.raises(RuntimeError, match="private permissions"):
        GeminiReceiptStore(tmp_path / "routing" / "routing.sqlite3")


def test_receipt_store_rejects_symlink_and_non_regular_database_paths(tmp_path: Path):
    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"")
    symlink = tmp_path / "routing.sqlite3"
    symlink.symlink_to(target)

    with pytest.raises(RuntimeError, match="symlink"):
        GeminiReceiptStore(symlink)

    directory = tmp_path / "database-directory"
    directory.mkdir()
    with pytest.raises(RuntimeError, match="regular file"):
        GeminiReceiptStore(directory)


def test_write_connection_rechecks_sidecar_permissions_after_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = make_store(tmp_path)
    wal = Path(f"{store.path}-wal")
    shm = Path(f"{store.path}-shm")

    class FakeConnection:
        def close(self):
            for path in (wal, shm):
                path.write_bytes(b"test-sidecar")
                path.chmod(0o644)

    monkeypatch.setattr(store, "_open_write", lambda: FakeConnection())

    with store._write_connection():
        pass

    assert stat.S_IMODE(wal.stat().st_mode) == 0o600
    assert stat.S_IMODE(shm.stat().st_mode) == 0o600


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
