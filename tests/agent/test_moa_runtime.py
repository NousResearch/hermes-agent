import json
import os
from pathlib import Path
from decimal import Decimal

import pytest

from agent.moa_runtime import (
    apply_route_feedback,
    circuit_status,
    mark_slot_failure,
    mark_slot_success,
    read_moa_telemetry,
    record_moa_telemetry,
    record_route_feedback,
    reset_runtime_state_for_tests,
)


@pytest.fixture(autouse=True)
def _isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    reset_runtime_state_for_tests()
    yield
    reset_runtime_state_for_tests()


class RateLimitError(Exception):
    status_code = 429


def test_circuit_breaker_opens_and_success_closes_it():
    slot = {"provider": "xai-oauth", "model": "grok-4.5"}

    opened = mark_slot_failure(slot, RateLimitError("rate limit"), cooldown_seconds=20)

    assert opened["active"] is True
    assert opened["reason"] == "rate_limit"
    assert circuit_status(slot)["retry_after_seconds"] > 0

    mark_slot_success(slot)
    assert circuit_status(slot)["active"] is False


def test_telemetry_drops_content_and_error_text():
    record_moa_telemetry({
        "preset": "max",
        "status": "ok",
        "aggregator": "openai-codex:gpt-5.6-sol",
        "latency_ms": 1200,
        "reference_cost_usd": Decimal("0.125"),
        "prompt": "private prompt",
        "output": "private output",
        "error": "secret exception",
    })

    summary = read_moa_telemetry()
    assert summary["events"] == 1
    assert summary["presets"]["max"]["average_latency_ms"] == 1200
    assert summary["presets"]["max"]["known_reference_cost_usd"] == 0.125

    path = Path(os.environ["HERMES_HOME"]) / "logs" / "moa_runtime.jsonl"
    raw = path.read_text(encoding="utf-8")
    assert "private prompt" not in raw
    assert "private output" not in raw
    assert "secret exception" not in raw


def test_route_feedback_changes_route_only_after_stable_signal():
    assert apply_route_feedback("balanced") == "balanced"
    for _ in range(3):
        record_route_feedback("balanced", "research")

    assert apply_route_feedback("balanced") == "research"
    feedback_path = Path(os.environ["HERMES_HOME"]) / "state" / "moa_router_feedback.json"
    data = json.loads(feedback_path.read_text(encoding="utf-8"))
    assert data == {"routes": {"balanced": {"research": 3}}}
