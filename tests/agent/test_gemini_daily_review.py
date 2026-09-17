from __future__ import annotations

import json
import hashlib
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.gemini_daily_review import (
    AlertDeliveryIndeterminateError,
    DailyReviewRunner,
    SolReviewer,
    _parse_verdict,
    build_review_prompt,
    sample_receipt_ids,
)
from agent.gemini_route_receipts import GeminiReceiptStore


UTC = timezone.utc


def test_verdict_parser_enforces_exact_fail_closed_contract():
    assert _parse_verdict(
        {"verdict": "pass", "reason": "matched the evidence", "failure_kind": "none"}
    ) == {
        "verdict": "pass",
        "reason": "matched the evidence",
        "failure_kind": "none",
    }
    with pytest.raises(ValueError, match="exactly"):
        _parse_verdict({"verdict": "faithful", "reason": "legacy alias"})
    with pytest.raises(ValueError, match="failure_kind"):
        _parse_verdict({"verdict": "fail", "reason": "missing field"})
    with pytest.raises(ValueError, match="failure_kind"):
        _parse_verdict(
            {"verdict": "pass", "reason": "wrong kind", "failure_kind": "correctness"}
        )


def add_attempt(
    store: GeminiReceiptStore,
    receipt_id: str,
    *,
    status: str = "completed",
    fallback_used: bool = False,
    started_at: datetime = datetime(2026, 9, 11, 12, tzinfo=UTC),
) -> None:
    store.prepare_attempt(
        receipt_id=receipt_id,
        parent_session_id="parent",
        parent_turn_id="turn",
        child_session_id=f"child-{receipt_id}",
        task_index=0,
        route_requested="auto",
        route_decision="gemini",
        route_reason="eligible_leaf",
        data_classification="standard",
        output_contract="text",
        goal_text=f"goal {receipt_id}",
        context_text="bounded context",
        requested_provider="antigravity-subscription",
        requested_model="gemini-3.8-flash-low",
        requested_effort="low",
        started_at=started_at,
    )
    store.mark_process_started(receipt_id, when=started_at)
    store.complete_attempt(
        receipt_id,
        worker_status=status,
        response_text=f"response {receipt_id}" if status == "completed" else None,
        duration_ms=1,
        fallback_used=fallback_used,
        error_code=None if status == "completed" else status,
        completed_at=started_at,
    )


def test_sampling_uses_exact_hmac_rank_vector():
    ids = ["grt_f", "grt_c", "grt_a", "grt_e", "grt_b", "grt_d"]
    seed = bytes.fromhex("42" * 32)

    assert sample_receipt_ids(ids, 5, seed) == [
        "grt_f",
        "grt_a",
        "grt_d",
        "grt_b",
        "grt_e",
    ]


def test_sampling_reviews_all_when_cohort_is_under_five():
    ids = ["grt_a", "grt_b", "grt_c"]
    assert set(sample_receipt_ids(ids, 5, bytes.fromhex("11" * 32))) == set(ids)


def test_review_prompt_contains_metadata_only_and_strict_schema(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_a")

    prompt = build_review_prompt(store.get_attempt("grt_a"))

    assert "goal grt_a" not in prompt
    assert "bounded context" not in prompt
    assert "response grt_a" not in prompt
    assert "goal_sha256" in prompt
    assert "response_sha256" in prompt
    assert "route_reason" in prompt
    assert '"verdict": "pass|fail"' in prompt
    assert '"failure_kind": "none|correctness|instruction|omission|hallucination|worker_failure|unreviewable"' in prompt
    assert "Return JSON only" in prompt


def test_sol_reviewer_creates_fresh_toolless_memoryless_agent_per_receipt():
    created: list[dict] = []

    class FakeAgent:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.provider = kwargs["provider"]
            self.model = kwargs["model"]

        def run_conversation(self, prompt):
            return {"final_response": '{"verdict":"pass","reason":"good","failure_kind":"none"}'}

        def close(self):
            pass

    reviewer = SolReviewer(
        provider="openai-codex",
        model="gpt-5.6-sol",
        agent_factory=FakeAgent,
    )
    reviewer("one")
    reviewer("two")

    assert len(created) == 2
    for kwargs in created:
        assert kwargs["enabled_toolsets"] == []
        assert kwargs["skip_memory"] is True
        assert kwargs["skip_context_files"] is True
        assert kwargs["load_soul_identity"] is False
        assert kwargs["session_db"] is None
        assert kwargs["save_trajectories"] is False
        assert kwargs["provider"] == "openai-codex"
        assert kwargs["model"] == "gpt-5.6-sol"


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("arbitrary-provider", "gpt-5.6-sol"),
        ("openai-codex", "arbitrary-model"),
    ],
)
def test_sol_reviewer_rejects_noncanonical_requested_identity(provider: str, model: str):
    with pytest.raises(ValueError, match="reviewer_identity_mismatch"):
        SolReviewer(provider=provider, model=model, agent_factory=lambda **_kwargs: None)


def test_sol_reviewer_fails_closed_on_actual_provider_or_model_mismatch():
    class WrongAgent:
        provider = "openrouter"
        model = "wrong-model"

        def __init__(self, **_kwargs):
            self.closed = False

        def close(self):
            self.closed = True

    reviewer = SolReviewer(
        provider="openai-codex",
        model="gpt-5.6-sol",
        agent_factory=WrongAgent,
    )

    with pytest.raises(RuntimeError, match="reviewer_identity_mismatch"):
        reviewer("review this")


def test_sol_reviewer_fails_closed_when_actual_identity_is_unavailable():
    class IdentitylessAgent:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    reviewer = SolReviewer(
        provider="openai-codex",
        model="gpt-5.6-sol",
        agent_factory=IdentitylessAgent,
    )

    with pytest.raises(RuntimeError, match="reviewer_identity_mismatch"):
        reviewer("review this")


def test_runner_reviews_full_cohort_but_worker_failures_fail_closed(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    for index, status in enumerate(("completed", "failed", "timeout")):
        add_attempt(store, f"grt_{index}", status=status, fallback_used=status != "completed")
    alerts: list[str] = []
    prompts: list[str] = []

    def reviewer(prompt: str) -> dict:
        prompts.append(prompt)
        return {"verdict": "pass", "reason": "acceptable", "failure_kind": "none"}

    result = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: reviewer,
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda message: alerts.append(message)
            or {"success": True, "chat_id": "C_ROUTE_FAILURES", "message_id": "1.1"},
        alert_channel_id="C_ROUTE_FAILURES",
    ).run(target_day=date(2026, 9, 11), seed=bytes.fromhex("33" * 32))

    assert result["status"] == "failed"
    assert result["eligible_count"] == 3
    assert result["reviewed_count"] == 3
    assert len(prompts) == 3
    assert any('"worker_status":"failed"' in prompt for prompt in prompts)
    assert any('"fallback_used":1' in prompt for prompt in prompts)
    assert len(alerts) == 1
    items = store.list_review_items(store.get_review_batch("2026-09-11")["batch_id"])
    assert [item["verdict"] for item in items].count("fail") == 2
    failed_worker_item = next(item for item in items if item["receipt_id"] == "grt_1")
    assert failed_worker_item["failure_kind"] == "worker_failure"
    expected_review = {"verdict": "pass", "reason": "acceptable", "failure_kind": "none"}
    assert failed_worker_item["reason"] is None
    assert failed_worker_item["review_json"] is None
    assert failed_worker_item["review_sha256"] == hashlib.sha256(
        json.dumps(expected_review, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_stale_runner_cannot_persist_item_after_review_lease_is_terminalized(
    tmp_path: Path,
):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    old = datetime(2026, 9, 11, 12, tzinfo=UTC)
    add_attempt(store, "grt_stale_runner", started_at=old)
    reviewer_entered = threading.Event()
    release_reviewer = threading.Event()
    results: list[dict] = []
    errors: list[BaseException] = []
    alerts: list[str] = []

    def reviewer(_prompt: str) -> dict[str, str]:
        reviewer_entered.set()
        assert release_reviewer.wait(timeout=5)
        return {"verdict": "pass", "reason": "late result", "failure_kind": "none"}

    runner = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: reviewer,
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda message: alerts.append(message) or {"success": True},
        alert_channel_id="C_ROUTE_FAILURES",
        alert_workspace_id="T_ROUTE_FAILURES",
        clock=lambda: old,
        lease_timeout_seconds=1,
    )

    def run_review() -> None:
        try:
            results.append(
                runner.run(target_day="2026-09-11", seed=bytes.fromhex("58" * 32))
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_review)
    thread.start()
    assert reviewer_entered.wait(timeout=5)
    batch = store.get_review_batch("2026-09-11")
    assert batch is not None
    assert store.fail_stale_review_batch(
        batch["batch_id"],
        stale_before=old + timedelta(seconds=1),
        pipeline_error="stale_review_lease",
        alert_message="authoritative stale-lease alert",
        slack_channel_id="C_AUTH",
        slack_workspace_id="T_AUTH",
        completed_at=old + timedelta(seconds=2),
    )
    release_reviewer.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert results[0]["status"] == "pipeline_failed"
    assert store.list_review_items(batch["batch_id"]) == []
    assert alerts == []


@pytest.mark.parametrize(
    ("pipeline_preflight_error", "expected_item_count"),
    [(None, 1), ("preflight_failed", 0)],
)
def test_stale_runner_losing_lease_before_terminal_update_returns_authoritative_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_preflight_error: str | None,
    expected_item_count: int,
):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    old = datetime(2026, 9, 11, 12, tzinfo=UTC)
    add_attempt(store, "grt_terminal_update_race", started_at=old)
    terminal_update_reached = threading.Event()
    release_terminal_update = threading.Event()
    original_update = store.update_review_batch
    results: list[dict] = []
    errors: list[BaseException] = []
    alerts: list[str] = []

    def paused_update(*args, **kwargs):
        terminal_update_reached.set()
        assert release_terminal_update.wait(timeout=5)
        return original_update(*args, **kwargs)

    monkeypatch.setattr(store, "update_review_batch", paused_update)
    runner = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: (
            lambda _prompt: {
                "verdict": "pass",
                "reason": "review completed before authority changed",
                "failure_kind": "none",
            }
        ),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda message: alerts.append(message) or {"success": True},
        alert_channel_id="C_ROUTE_FAILURES",
        alert_workspace_id="T_ROUTE_FAILURES",
        clock=lambda: old,
        lease_timeout_seconds=1,
    )

    def run_review() -> None:
        try:
            results.append(
                runner.run(
                    target_day="2026-09-11",
                    seed=bytes.fromhex("59" * 32),
                    pipeline_preflight_error=pipeline_preflight_error,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_review)
    thread.start()
    assert terminal_update_reached.wait(timeout=5)
    batch = store.get_review_batch("2026-09-11")
    assert batch is not None
    assert store.fail_stale_review_batch(
        batch["batch_id"],
        stale_before=old + timedelta(seconds=1),
        pipeline_error="stale_review_lease",
        alert_message="authoritative stale-lease alert",
        slack_channel_id="C_AUTH",
        slack_workspace_id="T_AUTH",
        completed_at=old + timedelta(seconds=2),
    )
    release_terminal_update.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert results[0]["status"] == "pipeline_failed"
    assert len(store.list_review_items(batch["batch_id"])) == expected_item_count
    authoritative = store.get_review_batch("2026-09-11")
    assert authoritative is not None
    assert authoritative["pipeline_error"] == "stale_review_lease"
    assert authoritative["slack_channel_id"] == "C_AUTH"
    assert authoritative["slack_workspace_id"] == "T_AUTH"
    assert alerts == []


def test_runner_alerts_only_sanitized_receipt_ids_and_reasons_on_quality_failure(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_a")
    alerts: list[str] = []

    result = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: (
            lambda _prompt: {"verdict": "fail", "reason": "unsupported conclusion", "failure_kind": "correctness"}
        ),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda message: alerts.append(message)
            or {"success": True, "chat_id": "C_ROUTE_FAILURES", "message_id": "1.1"},
        alert_channel_id="C_ROUTE_FAILURES",
    ).run(target_day="2026-09-11", seed=bytes.fromhex("44" * 32))

    assert result["status"] == "failed"
    assert len(alerts) == 1
    assert "grt_a" not in alerts[0]
    assert "unsupported conclusion" not in alerts[0]
    assert "goal grt_a" not in alerts[0]
    assert "response grt_a" not in alerts[0]


def test_successful_failure_alert_persists_exact_channel_and_message_ts(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_a")
    pre_send: list[dict] = []

    def sender(message: str) -> dict:
        batch = store.get_review_batch("2026-09-11")
        assert batch is not None
        pre_send.append(batch)
        assert batch["alert_status"] == "sending"
        assert batch["slack_channel_id"] == "C_ROUTE_FAILURES"
        assert batch["alert_message_sha256"] == hashlib.sha256(message.encode()).hexdigest()
        assert batch["alert_delivery_key"] == (
            f"gemini-daily-review:{batch['batch_id']}:{batch['alert_message_sha256']}"
        )
        return {
            "success": True,
            "chat_id": "C_ROUTE_FAILURES",
            "message_id": "1720000000.000001",
        }

    result = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: (
            lambda _prompt: {"verdict": "fail", "reason": "private review reason", "failure_kind": "correctness"}
        ),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=sender,
        alert_channel_id="C_ROUTE_FAILURES",
    ).run(target_day="2026-09-11", seed=bytes.fromhex("45" * 32))

    batch = store.get_review_batch("2026-09-11")
    assert batch is not None
    assert result["alert_status"] == "sent"
    assert len(pre_send) == 1
    assert batch["slack_channel_id"] == "C_ROUTE_FAILURES"
    assert batch["slack_message_ts"] == "1720000000.000001"
    assert batch["alert_status"] == "sent"


def test_alert_message_is_aggregate_only_and_excludes_review_content(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_private")
    alerts: list[str] = []

    DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: (
            lambda _prompt: {"verdict": "fail", "reason": "PRIVATE_REVIEW_REASON", "failure_kind": "correctness"}
        ),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda message: alerts.append(message)
        or {
            "success": True,
            "chat_id": "C_ROUTE_FAILURES",
            "message_id": "1720000000.000002",
        },
        alert_channel_id="C_ROUTE_FAILURES",
    ).run(target_day="2026-09-11", seed=bytes.fromhex("46" * 32))

    assert len(alerts) == 1
    assert alerts[0].startswith("Gemini daily review 2026-09-11: FAIL — 1/1")
    assert "PRIVATE_REVIEW_REASON" not in alerts[0]
    assert "grt_private" not in alerts[0]
    assert "routing.sqlite3 batch grb_" in alerts[0]


def test_wrong_channel_delivery_remains_pending_and_retries_same_batch(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_a")
    calls: list[str] = []

    def wrong_sender(message: str) -> dict:
        calls.append(message)
        return {"success": True, "chat_id": "C_WRONG", "message_id": "1.2"}

    runner = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: (
            lambda _prompt: {"verdict": "fail", "reason": "bad", "failure_kind": "correctness"}
        ),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=wrong_sender,
        alert_channel_id="C_ROUTE_FAILURES",
    )
    first = runner.run(target_day="2026-09-11", seed=bytes.fromhex("47" * 32))
    second = runner.run(target_day="2026-09-11", seed=bytes.fromhex("48" * 32))

    assert first["status"] == second["status"] == "failed"
    assert first["alert_status"] == second["alert_status"] == "pending"
    assert len(calls) == 2
    assert calls[0] == calls[1]


def test_indeterminate_alert_delivery_keeps_lease_and_blocks_immediate_retry(
    tmp_path: Path,
):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_a")
    calls: list[str] = []

    def indeterminate_sender(message: str) -> dict:
        calls.append(message)
        raise AlertDeliveryIndeterminateError("delivery outcome is unknown")

    runner = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: (
            lambda _prompt: {
                "verdict": "fail",
                "reason": "bad output",
                "failure_kind": "quality_failure",
            }
        ),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=indeterminate_sender,
        alert_channel_id="C_ROUTE_FAILURES",
        sample_size=1,
        clock=lambda: datetime(2026, 9, 12, 16, 0, tzinfo=UTC),
    )

    first = runner.run(target_day=date(2026, 9, 11), seed=bytes.fromhex("50" * 32))
    second = runner.run(target_day=date(2026, 9, 11), seed=bytes.fromhex("51" * 32))
    persisted = store.get_review_batch("2026-09-11")

    assert first["alert_status"] == "sending"
    assert second["alert_status"] == "sending"
    assert persisted is not None
    assert persisted["alert_status"] == "sending"
    assert len(calls) == 1


def test_pending_alert_retry_reuses_persisted_payload_and_destination(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_a")
    batch = store.create_or_get_review_batch(
        routing_day="2026-09-11",
        timezone_name="America/Los_Angeles",
        sample_size_requested=1,
        eligible_count=1,
        sample_seed_hex="49" * 32,
        sample_receipt_ids=["grt_a"],
    )
    review_lease = "review-lease"
    assert store.claim_review_batch(
        batch["batch_id"], lease_token=review_lease, now=datetime.now(UTC)
    )
    original = "original persisted alert payload"
    original_hash = hashlib.sha256(original.encode()).hexdigest()
    store.update_review_batch(
        batch["batch_id"],
        lease_token=review_lease,
        status="failed",
        alert_status="pending",
        alert_message=original,
        alert_delivery_key=f"gemini-daily-review:{batch['batch_id']}:{original_hash}",
        slack_channel_id="C_ROUTE_FAILURES",
    )

    sent: list[str] = []

    def sender(message: str) -> dict:
        sent.append(message)
        return {"success": True, "chat_id": "C_ROUTE_FAILURES", "message_id": "1.25"}

    result = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: pytest.fail("terminal batch must not be re-reviewed"),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=sender,
        alert_channel_id="C_CHANGED_AFTER_STAGING",
    ).run(target_day="2026-09-11")

    persisted = store.get_review_batch("2026-09-11")
    assert persisted is not None
    assert result["alert_status"] == "sent"
    assert sent == [original]
    assert persisted["alert_message"] == original
    assert persisted["slack_channel_id"] == "C_ROUTE_FAILURES"
    assert persisted["alert_message_sha256"] == original_hash
    assert persisted["alert_delivery_key"] == (
        f"gemini-daily-review:{batch['batch_id']}:{original_hash}"
    )


def test_concurrent_pending_alert_retries_admit_only_one_sender(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_a")
    failing_runner = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: (
            lambda _prompt: {"verdict": "fail", "reason": "bad", "failure_kind": "correctness"}
        ),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda _message: {"success": False},
        alert_channel_id="C_ROUTE_FAILURES",
    )
    assert failing_runner.run(target_day="2026-09-11")["alert_status"] == "pending"

    entered = threading.Event()
    release = threading.Event()
    sends: list[str] = []

    def sender(message: str) -> dict:
        sends.append(message)
        entered.set()
        assert release.wait(timeout=5)
        return {"success": True, "chat_id": "C_ROUTE_FAILURES", "message_id": "1.3"}

    runners = [
        DailyReviewRunner(
            store=store,
            reviewer_factory=lambda: pytest.fail("terminal batch must not be re-reviewed"),
            reviewer_provider="openai-codex",
            reviewer_model="gpt-5.6-sol",
            alert_sender=sender,
            alert_channel_id="C_ROUTE_FAILURES",
        )
        for _ in range(2)
    ]
    errors: list[BaseException] = []

    def retry(runner: DailyReviewRunner) -> None:
        try:
            runner.run(target_day="2026-09-11")
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=retry, args=(runners[0],))
    second = threading.Thread(target=retry, args=(runners[1],))
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    second.join(timeout=5)
    release.set()
    first.join(timeout=5)

    assert errors == []
    assert len(sends) == 1
    completed = store.get_review_batch("2026-09-11")
    assert completed is not None
    assert completed["alert_status"] == "sent"


def test_stale_outbox_initializer_cannot_reset_an_active_delivery_lease(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_a")
    now = datetime(2026, 9, 12, 12, tzinfo=UTC)
    runner = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: (
            lambda _prompt: {
                "verdict": "fail",
                "reason": "bad",
                "failure_kind": "correctness",
            }
        ),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda _message: {"success": False},
        alert_channel_id="C_ROUTE_FAILURES",
        clock=lambda: now,
    )
    assert runner.run(target_day="2026-09-11")["alert_status"] == "pending"
    stale = store.get_review_batch("2026-09-11")
    assert stale is not None
    assert store.claim_alert_delivery(
        stale["batch_id"],
        lease_token="owner-a",
        now=now,
        stale_before=now - timedelta(seconds=1),
    )

    runner._persist_alert_outbox(stale, stale["alert_message"])

    current = store.get_review_batch("2026-09-11")
    assert current is not None
    assert current["alert_status"] == "sending"
    assert current["alert_lease_token"] == "owner-a"
    assert not store.claim_alert_delivery(
        current["batch_id"],
        lease_token="owner-b",
        now=now,
        stale_before=now - timedelta(seconds=1),
    )


def test_stale_alert_delivery_lease_is_reclaimed_on_later_tick(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_a")
    now = datetime(2026, 9, 12, 12, tzinfo=UTC)
    runner = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: (
            lambda _prompt: {"verdict": "fail", "reason": "bad", "failure_kind": "correctness"}
        ),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda _message: {"success": False},
        alert_channel_id="C_ROUTE_FAILURES",
        clock=lambda: now,
        lease_timeout_seconds=10,
    )
    assert runner.run(target_day="2026-09-11")["alert_status"] == "pending"
    batch = store.get_review_batch("2026-09-11")
    assert batch is not None
    assert store.claim_alert_delivery(
        batch["batch_id"],
        lease_token="abandoned",
        now=now,
        stale_before=now,
    )

    recovered = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: pytest.fail("terminal batch must not be re-reviewed"),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda _message: {
            "success": True,
            "chat_id": "C_ROUTE_FAILURES",
            "message_id": "1.4",
        },
        alert_channel_id="C_ROUTE_FAILURES",
        clock=lambda: now + timedelta(seconds=11),
        lease_timeout_seconds=10,
    ).run(target_day="2026-09-11")

    assert recovered["alert_status"] == "sent"


def test_runner_fail_closes_and_alerts_on_malformed_reviewer_output(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_a")
    alerts: list[str] = []

    result = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: (lambda _prompt: "not-json"),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda message: alerts.append(message)
            or {"success": True, "chat_id": "C_ROUTE_FAILURES", "message_id": "1.1"},
        alert_channel_id="C_ROUTE_FAILURES",
    ).run(target_day="2026-09-11", seed=bytes.fromhex("55" * 32))

    assert result["status"] == "pipeline_failed"
    assert result["reviewed_count"] == 0
    assert len(alerts) == 1
    assert "PIPELINE FAIL — 0/1 reviews completed" in alerts[0]


def test_runner_terminalizes_and_alerts_when_sampled_receipt_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_missing")
    alerts: list[str] = []

    def missing_attempt(_receipt_id: str) -> dict:
        raise KeyError("receipt disappeared")

    monkeypatch.setattr(store, "get_attempt", missing_attempt)
    result = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: pytest.fail("missing receipt must not reach reviewer"),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda message: alerts.append(message)
        or {"success": True, "chat_id": "C_ROUTE_FAILURES", "message_id": "1.2"},
        alert_channel_id="C_ROUTE_FAILURES",
        alert_workspace_id="T_ROUTE",
    ).run(target_day="2026-09-11", seed=bytes.fromhex("54" * 32))

    assert result["status"] == "pipeline_failed"
    assert result["reviewed_count"] == 0
    batch = store.get_review_batch("2026-09-11")
    assert batch is not None
    expected_alert = (
        "Gemini daily review 2026-09-11: PIPELINE FAIL — 0/1 reviews completed. "
        f"Receipt: {store.path} batch {batch['batch_id']}"
    )
    assert alerts == [expected_alert]
    assert batch["status"] == "pipeline_failed"
    assert batch["pipeline_error"] == "sampled_receipt_missing"
    assert batch["alert_status"] == "sent"
    assert batch["slack_channel_id"] == "C_ROUTE_FAILURES"
    assert batch["slack_workspace_id"] == "T_ROUTE"
    assert batch["alert_message"] == expected_alert
    assert store.list_review_items(batch["batch_id"]) == []


def test_legacy_pending_pipeline_alert_migrates_exact_persisted_hash_and_delivers(
    tmp_path: Path,
):
    db_path = tmp_path / "routing.sqlite3"
    store = GeminiReceiptStore(db_path)
    add_attempt(store, "grt_a")
    DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: (lambda _prompt: "not-json"),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda _message: {"success": False},
        alert_channel_id="C_ROUTE_FAILURES",
    ).run(target_day="2026-09-11", seed=bytes.fromhex("56" * 32))
    batch = store.get_review_batch("2026-09-11")
    assert batch is not None
    current_alert = str(batch["alert_message"])
    legacy_alert = current_alert.replace("0/1 reviews completed", "1/1 reviews completed")
    assert legacy_alert != current_alert

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """UPDATE daily_review_batches
               SET alert_message=NULL, alert_delivery_key=NULL,
                   slack_channel_id=NULL, slack_workspace_id=NULL,
                   alert_message_sha256=?
               WHERE batch_id=?
            """,
            (hashlib.sha256(legacy_alert.encode()).hexdigest(), batch["batch_id"]),
        )

    sends: list[str] = []
    result = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: pytest.fail("terminal batch must not be re-reviewed"),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda message: sends.append(message)
        or {"success": True, "chat_id": "C_ROUTE_FAILURES", "message_id": "1.7"},
        alert_channel_id="C_ROUTE_FAILURES",
    ).run(target_day="2026-09-11")

    assert result["alert_status"] == "sent"
    assert sends == [legacy_alert]
    migrated = store.get_review_batch("2026-09-11")
    assert migrated is not None
    assert migrated["alert_message"] == legacy_alert
    assert migrated["slack_channel_id"] == "C_ROUTE_FAILURES"


def test_second_run_reuses_terminal_batch_without_duplicate_reviews_or_alerts(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_a")
    alerts: list[str] = []
    calls = 0

    def reviewer_factory():
        nonlocal calls

        def reviewer(_prompt: str) -> dict:
            nonlocal calls
            calls += 1
            return {"verdict": "fail", "reason": "bad", "failure_kind": "correctness"}

        return reviewer

    runner = DailyReviewRunner(
        store=store,
        reviewer_factory=reviewer_factory,
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda message: alerts.append(message)
            or {"success": True, "chat_id": "C_ROUTE_FAILURES", "message_id": "1.1"},
        alert_channel_id="C_ROUTE_FAILURES",
    )
    first = runner.run(target_day="2026-09-11", seed=bytes.fromhex("66" * 32))
    second = runner.run(target_day="2026-09-11", seed=bytes.fromhex("77" * 32))

    assert first["status"] == second["status"] == "failed"
    assert calls == 1
    assert len(alerts) == 1
    batch = store.get_review_batch("2026-09-11")
    assert batch["sample_seed_hex"] == "66" * 32


def test_concurrent_runs_claim_one_daily_batch_once(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_a")
    calls = 0
    calls_lock = threading.Lock()
    results: list[dict] = []

    def reviewer(_prompt: str) -> dict:
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.1)
        return {"verdict": "pass", "reason": "ok", "failure_kind": "none"}

    def run() -> None:
        results.append(
            DailyReviewRunner(
                store=store,
                reviewer_factory=lambda: reviewer,
                reviewer_provider="openai-codex",
                reviewer_model="gpt-5.6-sol",
                alert_sender=lambda _message: None,
                alert_channel_id="C_ROUTE_FAILURES",
            ).run(target_day="2026-09-11", seed=bytes.fromhex("88" * 32))
        )

    threads = [threading.Thread(target=run), threading.Thread(target=run)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert calls == 1
    assert {result["status"] for result in results} <= {"passed", "in_progress"}
    assert store.count_review_batches() == 1
    assert len(store.list_review_items(store.get_review_batch("2026-09-11")["batch_id"])) == 1


@pytest.mark.parametrize("stale_status", ["preparing", "reviewing"])
def test_stale_review_lease_becomes_pipeline_failure_without_rerun(
    tmp_path: Path, stale_status: str
):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    stale_started = datetime(2026, 9, 11, 7, 0, tzinfo=UTC)
    batch = store.create_or_get_review_batch(
        routing_day="2026-09-11",
        timezone_name="America/Los_Angeles",
        sample_size_requested=5,
        eligible_count=0,
        sample_seed_hex="44" * 32,
        sample_receipt_ids=[],
        started_at=stale_started,
    )
    if stale_status == "reviewing":
        assert store.claim_review_batch(
            batch["batch_id"], lease_token="stale-review", now=stale_started
        ) is True

    reviewer_calls: list[str] = []
    alerts: list[str] = []
    result = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: lambda prompt: reviewer_calls.append(prompt),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda message: alerts.append(message)
        or {"success": True, "chat_id": "C_ROUTE_FAILURES", "message_id": "1.1"},
        alert_channel_id="C_ROUTE_FAILURES",
        clock=lambda: datetime(2026, 9, 11, 7, 3, tzinfo=UTC),
        lease_timeout_seconds=150,
    ).run(target_day="2026-09-11")

    persisted = store.get_review_batch("2026-09-11")
    assert result["status"] == "pipeline_failed"
    assert persisted is not None
    assert persisted["status"] == "pipeline_failed"
    assert persisted["pipeline_error"] == "stale_review_lease"
    assert persisted["alert_status"] == "sent"
    assert len(alerts) == 1
    assert reviewer_calls == []


def test_started_nonterminal_attempt_is_a_pipeline_failure_not_a_sol_review(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    started = datetime(2026, 9, 11, 7, 0, tzinfo=UTC)
    store.prepare_attempt(
        receipt_id="grt_stale",
        parent_session_id="parent",
        parent_turn_id="turn",
        child_session_id="child",
        task_index=0,
        route_requested="auto",
        route_decision="gemini",
        route_reason="eligible_leaf",
        data_classification="standard",
        output_contract="text",
        goal_text="goal",
        context_text="context",
        requested_provider="antigravity-subscription",
        requested_model="gemini-3.8-flash-low",
        requested_effort="low",
        started_at=started,
    )
    store.mark_process_started("grt_stale", when=started)
    alerts: list[str] = []
    reviewer_calls: list[str] = []
    result = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: lambda prompt: reviewer_calls.append(prompt),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda message: alerts.append(message)
        or {"success": True, "chat_id": "C_ROUTE_FAILURES", "message_id": "1.1"},
        alert_channel_id="C_ROUTE_FAILURES",
    ).run(target_day="2026-09-11", seed=bytes.fromhex("99" * 32))

    assert result["status"] == "pipeline_failed"
    assert reviewer_calls == []
    assert len(alerts) == 1
