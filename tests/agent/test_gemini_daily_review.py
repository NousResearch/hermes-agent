from __future__ import annotations

import json
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from agent.gemini_daily_review import (
    DailyReviewRunner,
    SolReviewer,
    _parse_verdict,
    build_review_prompt,
    sample_receipt_ids,
)
from agent.gemini_route_receipts import GeminiReceiptStore


UTC = timezone.utc


def test_verdict_parser_maps_faithful_contract_to_internal_pass_fail():
    assert _parse_verdict({"verdict": "faithful", "reason": "matched the evidence"}) == {
        "verdict": "pass",
        "reason": "matched the evidence",
    }
    assert _parse_verdict(
        {"verdict": "unfaithful", "reason": "invented an unsupported result"}
    ) == {"verdict": "fail", "reason": "invented an unsupported result"}


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


def test_sampling_is_deterministic_unbiased_shape_and_exactly_five_of_six():
    ids = [f"grt_{index}" for index in range(6)]
    seed = bytes.fromhex("42" * 32)
    first = sample_receipt_ids(ids, 5, seed)
    second = sample_receipt_ids(ids, 5, seed)

    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5
    assert set(first) <= set(ids)


def test_sampling_reviews_all_when_cohort_is_under_five():
    ids = ["grt_a", "grt_b", "grt_c"]
    assert set(sample_receipt_ids(ids, 5, bytes.fromhex("11" * 32))) == set(ids)


def test_review_prompt_contains_full_attempt_and_strict_schema(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_a")

    prompt = build_review_prompt(store.get_attempt("grt_a"))

    assert "goal grt_a" in prompt
    assert "bounded context" in prompt
    assert "response grt_a" in prompt
    assert "route_reason" in prompt
    assert '"verdict": "faithful|unfaithful"' in prompt
    assert "Return JSON only" in prompt


def test_sol_reviewer_creates_fresh_toolless_memoryless_agent_per_receipt():
    created: list[dict] = []

    class FakeAgent:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def run_conversation(self, prompt):
            return {"final_response": '{"verdict":"pass","reason":"good"}'}

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


def test_runner_reviews_started_failures_and_fallbacks_and_stays_silent_on_pass(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    for index, status in enumerate(("completed", "failed", "timeout")):
        add_attempt(store, f"grt_{index}", status=status, fallback_used=status != "completed")
    alerts: list[str] = []
    prompts: list[str] = []

    def reviewer(prompt: str) -> dict:
        prompts.append(prompt)
        return {"verdict": "pass", "reason": "acceptable"}

    result = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: reviewer,
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda message: alerts.append(message)
            or {"success": True, "chat_id": "C_ROUTE_FAILURES", "message_id": "1.1"},
        alert_channel_id="C_ROUTE_FAILURES",
    ).run(target_day=date(2026, 9, 11), seed=bytes.fromhex("33" * 32))

    assert result["status"] == "passed"
    assert result["eligible_count"] == 3
    assert result["reviewed_count"] == 3
    assert len(prompts) == 3
    assert any('"worker_status":"failed"' in prompt for prompt in prompts)
    assert any('"fallback_used":1' in prompt for prompt in prompts)
    assert alerts == []


def test_runner_alerts_only_sanitized_receipt_ids_and_reasons_on_quality_failure(tmp_path: Path):
    store = GeminiReceiptStore(tmp_path / "routing.sqlite3")
    add_attempt(store, "grt_a")
    alerts: list[str] = []

    result = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: (
            lambda _prompt: {"verdict": "fail", "reason": "unsupported conclusion"}
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

    result = DailyReviewRunner(
        store=store,
        reviewer_factory=lambda: (
            lambda _prompt: {"verdict": "fail", "reason": "private review reason"}
        ),
        reviewer_provider="openai-codex",
        reviewer_model="gpt-5.6-sol",
        alert_sender=lambda _message: {
            "success": True,
            "chat_id": "C_ROUTE_FAILURES",
            "message_id": "1720000000.000001",
        },
        alert_channel_id="C_ROUTE_FAILURES",
    ).run(target_day="2026-09-11", seed=bytes.fromhex("45" * 32))

    batch = store.get_review_batch("2026-09-11")
    assert batch is not None
    assert result["alert_status"] == "sent"
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
            lambda _prompt: {"verdict": "fail", "reason": "PRIVATE_REVIEW_REASON"}
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
            lambda _prompt: {"verdict": "fail", "reason": "bad"}
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
    assert result["reviewed_count"] == 1
    assert len(alerts) == 1
    assert "PIPELINE FAIL" in alerts[0]


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
            return {"verdict": "fail", "reason": "bad"}

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
        return {"verdict": "pass", "reason": "ok"}

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
        assert store.claim_review_batch(batch["batch_id"]) is True

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
