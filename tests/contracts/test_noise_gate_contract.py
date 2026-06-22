"""Reusable contract tests for Hermes/OpenClaw noise-gate backends.

These tests standardize the BE-REL-NOISE-001 / BE-NG-QA-SUPPORT-001
acceptance checklist. They are intentionally implementation-agnostic: point the
suite at a candidate backend with:

    HERMES_NOISE_GATE_EVALUATE_EVENT="package.module:evaluate_event"

The target callable must accept one event dict and return a dict-like
NoiseGateResult. If the env var is unset, the suite skips so the repository CI
can carry the standard before an implementation exists.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any, Callable
from uuid import uuid4

import pytest


pytestmark = pytest.mark.contract

TARGET_ENV = "HERMES_NOISE_GATE_EVALUATE_EVENT"
SAFE_SOURCE_SCOPE_HASH = hashlib.sha256(b"qa-noise-gate-contract-source").hexdigest()
ACCEPTED_DECISIONS = {"accepted", "accepted_after_window"}
REJECTED_DECISIONS = {"rejected_invalid_key", "rejected_invalid_ttl"}


def _normalize_semantic(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.strip().split())
    if isinstance(value, list):
        return [_normalize_semantic(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_semantic(value[key]) for key in sorted(value)}
    return value


def _canonical_json(value: Any) -> str:
    normalized = _normalize_semantic(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _load_evaluate_event() -> Callable[[dict[str, Any]], Any]:
    target = os.environ.get(TARGET_ENV)
    if not target:
        pytest.skip(
            f"Set {TARGET_ENV}=package.module:evaluate_event to run the "
            "standard noise-gate contract suite against an implementation."
        )
    if ":" not in target:
        raise AssertionError(f"{TARGET_ENV} must use package.module:function form")
    module_name, function_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    evaluate_event = getattr(module, function_name)
    if not callable(evaluate_event):
        raise AssertionError(f"{target} is not callable")
    return evaluate_event


@pytest.fixture()
def evaluate_event() -> Callable[[dict[str, Any]], dict[str, Any]]:
    target = _load_evaluate_event()

    def call(event: dict[str, Any]) -> dict[str, Any]:
        result = target(copy.deepcopy(event))
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        elif hasattr(result, "to_dict"):
            result = result.to_dict()
        elif hasattr(result, "__dict__") and not isinstance(result, dict):
            result = vars(result)
        assert isinstance(result, dict), "NoiseGateResult must be dict-like"
        return result

    return call


@pytest.fixture()
def lane() -> str:
    return f"qa-noise-gate-contract-{uuid4().hex}"


def _event(
    lane: str,
    *,
    profile: str = "backend",
    event_type: str = "standup_update",
    payload: dict[str, Any] | None = None,
    ttl_seconds: int = 60,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    payload = payload or {
        "action_item_id": "BE-NG-QA-SUPPORT-001",
        "state_before": "pending",
        "state_after": "ready_for_qa",
        "target_channel_class": "audit_log",
        "business_key": "noise-gate-contract-proof",
        "normalized_summary": "backend contract event ready for QA",
    }
    created_at = created_at or datetime(2026, 5, 12, 17, 0, tzinfo=UTC)
    return {
        "profile": profile,
        "lane": lane,
        "event_type": event_type,
        "source_scope_hash": SAFE_SOURCE_SCOPE_HASH,
        "semantic_payload": payload,
        "semantic_event_hash": _semantic_hash(payload),
        "ttl_seconds": ttl_seconds,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
    }


def _decision(result: dict[str, Any]) -> str:
    decision = result.get("decision")
    assert isinstance(decision, str), "NoiseGateResult.decision must be a string"
    return decision


def _assert_no_side_effect(result: dict[str, Any]) -> None:
    assert result.get("side_effect_allowed") is False


def test_duplicate_replay_accepts_first_event_and_suppresses_replay(
    evaluate_event: Callable[[dict[str, Any]], dict[str, Any]], lane: str
) -> None:
    event = _event(lane)

    first = evaluate_event(event)
    replay = evaluate_event(event)

    assert _decision(first) == "accepted"
    assert first.get("side_effect_allowed") is True
    assert _decision(replay) == "suppressed_duplicate"
    _assert_no_side_effect(replay)
    assert replay.get("dedupe_key_hash") == first.get("dedupe_key_hash")


def test_canonicalization_suppresses_formatting_only_payload_changes(
    evaluate_event: Callable[[dict[str, Any]], dict[str, Any]], lane: str
) -> None:
    first_payload = {
        "state_after": "ready_for_qa",
        "action_item_id": "BE-NG-QA-SUPPORT-001",
        "normalized_summary": "backend contract event ready for QA",
        "tags": ["qa", "noise-gate", "backend"],
    }
    equivalent_payload = {
        "tags": ["qa", "noise-gate", "backend"],
        "normalized_summary": "  backend   contract event ready for QA  ",
        "action_item_id": "BE-NG-QA-SUPPORT-001",
        "state_after": "ready_for_qa",
    }

    first = evaluate_event(_event(lane, payload=first_payload))
    replay = evaluate_event(_event(lane, payload=equivalent_payload))

    assert _decision(first) == "accepted"
    assert _decision(replay) == "suppressed_duplicate"
    _assert_no_side_effect(replay)


def test_ttl_window_suppresses_inside_window_and_accepts_after_expiry(
    evaluate_event: Callable[[dict[str, Any]], dict[str, Any]], lane: str
) -> None:
    start = datetime(2026, 5, 12, 17, 0, tzinfo=UTC)
    payload = {"action_item_id": "BE-NG-QA-SUPPORT-001", "normalized_summary": "ttl probe"}

    first = evaluate_event(_event(lane, payload=payload, ttl_seconds=60, created_at=start))
    inside_window = evaluate_event(
        _event(lane, payload=payload, ttl_seconds=60, created_at=start + timedelta(seconds=30))
    )
    after_window = evaluate_event(
        _event(lane, payload=payload, ttl_seconds=60, created_at=start + timedelta(seconds=90))
    )

    assert _decision(first) == "accepted"
    assert _decision(inside_window) == "suppressed_duplicate"
    _assert_no_side_effect(inside_window)
    assert _decision(after_window) in ACCEPTED_DECISIONS
    assert after_window.get("side_effect_allowed") is True


def test_cross_profile_independence_accepts_same_semantics_for_different_profiles(
    evaluate_event: Callable[[dict[str, Any]], dict[str, Any]], lane: str
) -> None:
    payload = {
        "action_item_id": "BE-NG-QA-SUPPORT-001",
        "normalized_summary": "same semantic event from separate profiles",
    }

    backend_result = evaluate_event(_event(lane, profile="backend", payload=payload))
    qa_result = evaluate_event(_event(lane, profile="qa", payload=payload))

    assert _decision(backend_result) == "accepted"
    assert backend_result.get("side_effect_allowed") is True
    assert _decision(qa_result) == "accepted"
    assert qa_result.get("side_effect_allowed") is True
    assert backend_result.get("dedupe_key_hash") != qa_result.get("dedupe_key_hash")


@pytest.mark.parametrize(
    ("mutation", "expected_reason_fragment"),
    [
        (lambda event: event.pop("profile"), "profile"),
        (lambda event: event.pop("event_type"), "event_type"),
        (lambda event: event.pop("semantic_event_hash"), "semantic_event_hash"),
        (
            lambda event: (
                event.pop("source_scope_hash"),
                event.update({"source_scope": "telegram:-1001234567890:secret-thread"}),
            ),
            "privacy",
        ),
        (lambda event: event.update({"ttl_seconds": 0}), "ttl"),
    ],
)
def test_invalid_events_are_rejected_before_side_effects(
    evaluate_event: Callable[[dict[str, Any]], dict[str, Any]],
    lane: str,
    mutation: Callable[[dict[str, Any]], Any],
    expected_reason_fragment: str,
) -> None:
    event = _event(lane)
    mutation(event)

    result = evaluate_event(event)

    assert _decision(result) in REJECTED_DECISIONS
    _assert_no_side_effect(result)
    serialized = json.dumps(result, sort_keys=True, default=str).lower()
    assert expected_reason_fragment in serialized


def test_observable_metadata_is_audit_safe_and_hash_only(
    evaluate_event: Callable[[dict[str, Any]], dict[str, Any]], lane: str
) -> None:
    event = _event(lane)

    result = evaluate_event(event)

    assert _decision(result) == "accepted"
    assert result.get("dedupe_key_hash")
    metadata = result.get("observable_metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("source_scope_hash") == SAFE_SOURCE_SCOPE_HASH
    assert metadata.get("semantic_event_hash") == event["semantic_event_hash"]
    assert metadata.get("raw_private_fields_present") is False
    serialized = json.dumps(result, sort_keys=True, default=str)
    assert "telegram:-100" not in serialized
    assert "@" not in serialized
    assert "gho_" not in serialized
    assert "token" not in serialized.lower()


def test_concurrent_duplicate_submissions_have_one_winner(
    evaluate_event: Callable[[dict[str, Any]], dict[str, Any]], lane: str
) -> None:
    event = _event(
        lane,
        payload={
            "action_item_id": "BE-NG-QA-SUPPORT-001",
            "normalized_summary": "concurrent duplicate submission",
        },
        ttl_seconds=300,
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: evaluate_event(event), range(8)))

    decisions = [_decision(result) for result in results]
    assert decisions.count("accepted") == 1
    assert decisions.count("suppressed_duplicate") == 7
    assert sum(result.get("side_effect_allowed") is True for result in results) == 1
    assert sum(result.get("side_effect_allowed") is False for result in results) == 7
