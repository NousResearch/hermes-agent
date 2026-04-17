"""FASE 3 — Routing telemetry tests (RED suite)."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Add agent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agent"))

from routing_telemetry import (
    TelemetryEvent,
    PREMIUM_MULTIPLIERS,
    multiplier_for,
    build_event,
    record_event,
    load_events,
    summarize,
    wrap_resolve_turn_route,
    DEFAULT_STORE,
)


@pytest.fixture
def isolated_store(tmp_path):
    """Return a Path for an isolated JSONL store."""
    store = tmp_path / "telemetry.jsonl"
    return store


def test_record_and_load_event_roundtrip(isolated_store):
    """3 eventos → load devuelve 3 con mismos valores."""
    events = [
        build_event(
            model="gpt-5-mini",
            provider="copilot",
            domain="code",
            decision_source="cheap_route",
            turn_kind="simple_turn",
            success=True,
            tokens_in=100,
            tokens_out=200,
        ),
        build_event(
            model="claude-opus-4.6",
            provider="copilot",
            domain="code",
            decision_source="primary_default",
            turn_kind="complex_task",
            success=False,
            error_type="timeout",
            tokens_in=500,
            tokens_out=0,
        ),
        build_event(
            model="o3-mini",
            provider="copilot",
            domain="reasoning",
            decision_source="reasoning_fallback",
            turn_kind="reasoning_task",
            success=True,
            latency_ms=150.0,
            tokens_in=300,
            tokens_out=400,
        ),
    ]
    for ev in events:
        record_event(ev, store=isolated_store)
    
    loaded = load_events(store=isolated_store)
    assert len(loaded) == 3
    for orig, loaded_ev in zip(events, loaded):
        assert orig.model == loaded_ev.model
        assert orig.provider == loaded_ev.provider
        assert orig.success == loaded_ev.success
        assert orig.tokens_in == loaded_ev.tokens_in
        assert orig.tokens_out == loaded_ev.tokens_out


def test_build_event_computes_premium_units():
    """model="claude-opus-4.6", tokens_in=1000, tokens_out=0 → premium_units == 5.0"""
    ev = build_event(
        model="claude-opus-4.6",
        provider="copilot",
        domain="code",
        decision_source="primary_default",
        turn_kind="complex_task",
        success=True,
        tokens_in=1000,
        tokens_out=0,
    )
    # multiplier=5.0, tokens=1000, premium_units = 5.0 * 1000 / 1000 = 5.0
    assert ev.premium_units == 5.0


def test_multiplier_for_unknown_model_defaults_to_1():
    """multiplier_for("foo-bar") == 1.0"""
    assert multiplier_for("foo-bar") == 1.0


def test_summarize_identifies_worst_model(isolated_store):
    """10 eventos gpt-5-mini (9 success) + 10 claude-opus-4.6 (3 success) → worst_model == claude-opus-4.6"""
    # gpt-5-mini: 9 success, 1 failure
    for i in range(9):
        ev = build_event(
            model="gpt-5-mini",
            provider="copilot",
            domain="code",
            decision_source="cheap_route",
            turn_kind="simple_turn",
            success=True,
            tokens_in=100,
            tokens_out=100,
        )
        record_event(ev, store=isolated_store)
    ev_fail = build_event(
        model="gpt-5-mini",
        provider="copilot",
        domain="code",
        decision_source="cheap_route",
        turn_kind="simple_turn",
        success=False,
        error_type="rate_limit",
        tokens_in=100,
        tokens_out=0,
    )
    record_event(ev_fail, store=isolated_store)
    
    # claude-opus-4.6: 3 success, 7 failures
    for i in range(3):
        ev = build_event(
            model="claude-opus-4.6",
            provider="copilot",
            domain="code",
            decision_source="primary_default",
            turn_kind="complex_task",
            success=True,
            tokens_in=500,
            tokens_out=500,
        )
        record_event(ev, store=isolated_store)
    for i in range(7):
        ev = build_event(
            model="claude-opus-4.6",
            provider="copilot",
            domain="code",
            decision_source="primary_default",
            turn_kind="complex_task",
            success=False,
            error_type="timeout",
            tokens_in=500,
            tokens_out=0,
        )
        record_event(ev, store=isolated_store)
    
    events = load_events(store=isolated_store)
    summary = summarize(events)
    assert summary["worst_model"] == "claude-opus-4.6"


def test_summarize_identifies_optimal_model(isolated_store):
    """gpt-5-mini 20/20 success latency 100ms + claude-sonnet-4.6 20/20 success latency 500ms → optimal_model == gpt-5-mini"""
    # gpt-5-mini: 20 success, latency 100ms
    for i in range(20):
        ev = build_event(
            model="gpt-5-mini",
            provider="copilot",
            domain="code",
            decision_source="cheap_route",
            turn_kind="simple_turn",
            success=True,
            latency_ms=100.0,
            tokens_in=100,
            tokens_out=100,
        )
        record_event(ev, store=isolated_store)
    
    # claude-sonnet-4.6: 20 success, latency 500ms
    for i in range(20):
        ev = build_event(
            model="claude-sonnet-4.6",
            provider="copilot",
            domain="code",
            decision_source="primary_default",
            turn_kind="complex_task",
            success=True,
            latency_ms=500.0,
            tokens_in=500,
            tokens_out=500,
        )
        record_event(ev, store=isolated_store)
    
    events = load_events(store=isolated_store)
    summary = summarize(events)
    assert summary["optimal_model"] == "gpt-5-mini"


def test_summarize_worst_requires_min_5_requests(isolated_store):
    """modelo con 2 requests ambos fallidos no debe ser worst si hay otro con ≥5"""
    # Model with only 2 requests, both failures
    for i in range(2):
        ev = build_event(
            model="rare-model",
            provider="copilot",
            domain="code",
            decision_source="fallback",
            turn_kind="simple_turn",
            success=False,
            error_type="unknown",
            tokens_in=100,
            tokens_out=0,
        )
        record_event(ev, store=isolated_store)
    
    # Model with 5 requests, 3 failures (60% failure rate, but more requests)
    for i in range(2):
        ev = build_event(
            model="common-model",
            provider="copilot",
            domain="code",
            decision_source="primary",
            turn_kind="simple_turn",
            success=True,
            tokens_in=100,
            tokens_out=100,
        )
        record_event(ev, store=isolated_store)
    for i in range(3):
        ev = build_event(
            model="common-model",
            provider="copilot",
            domain="code",
            decision_source="primary",
            turn_kind="simple_turn",
            success=False,
            error_type="timeout",
            tokens_in=100,
            tokens_out=0,
        )
        record_event(ev, store=isolated_store)
    
    events = load_events(store=isolated_store)
    summary = summarize(events)
    # rare-model has 100% failure but only 2 requests (< 5 min)
    # common-model has 60% failure rate with 5 requests
    # worst_model should be common-model since rare-model doesn't meet min threshold
    assert summary["worst_model"] == "common-model"


def test_summarize_premium_savings(isolated_store):
    """mezcla calculada manual; savings == (baseline - actual)/baseline"""
    # Create events with known token counts
    # gpt-5-mini: multiplier=0.0, 1000 tokens total
    ev1 = build_event(
        model="gpt-5-mini",
        provider="copilot",
        domain="code",
        decision_source="cheap_route",
        turn_kind="simple_turn",
        success=True,
        tokens_in=500,
        tokens_out=500,
    )
    record_event(ev1, store=isolated_store)
    
    # claude-opus-4.6: multiplier=5.0, 1000 tokens total
    ev2 = build_event(
        model="claude-opus-4.6",
        provider="copilot",
        domain="code",
        decision_source="primary_default",
        turn_kind="complex_task",
        success=True,
        tokens_in=500,
        tokens_out=500,
    )
    record_event(ev2, store=isolated_store)
    
    events = load_events(store=isolated_store)
    summary = summarize(events)
    
    # Total tokens = 2000
    # baseline_all_opus_units = 2000 * 5 / 1000 = 10.0
    # actual: gpt-5-mini = 0.0 * 1000/1000 = 0, opus = 5.0 * 1000/1000 = 5.0
    # total_premium_units = 5.0
    # savings = (10.0 - 5.0) / 10.0 = 0.5
    assert summary["baseline_all_opus_units"] == 10.0
    assert summary["total_premium_units"] == 5.0
    assert abs(summary["savings_vs_baseline"] - 0.5) < 0.001


def test_storage_is_jsonl_append_only(isolated_store):
    """escribir 2 eventos, abrir archivo, contar líneas == 2, json.loads cada una"""
    ev1 = build_event(
        model="gpt-5-mini",
        provider="copilot",
        domain="code",
        decision_source="cheap_route",
        turn_kind="simple_turn",
        success=True,
        tokens_in=100,
        tokens_out=100,
    )
    ev2 = build_event(
        model="claude-opus-4.6",
        provider="copilot",
        domain="code",
        decision_source="primary_default",
        turn_kind="complex_task",
        success=False,
        tokens_in=200,
        tokens_out=0,
    )
    record_event(ev1, store=isolated_store)
    record_event(ev2, store=isolated_store)
    
    content = isolated_store.read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    assert len(lines) == 2
    
    # Verify each line is valid JSON
    for line in lines:
        data = json.loads(line)
        assert "model" in data
        assert "timestamp" in data


def test_since_filter_drops_old_events(isolated_store):
    """usa datetime aware, evento viejo filtrado"""
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(hours=2)
    
    # Create event with old timestamp manually
    old_ev = TelemetryEvent(
        timestamp=old_time.isoformat(),
        model="gpt-5-mini",
        provider="copilot",
        domain="code",
        decision_source="cheap_route",
        turn_kind="simple_turn",
        success=True,
        error_type=None,
        latency_ms=50.0,
        tokens_in=100,
        tokens_out=100,
        premium_multiplier=0.0,
        premium_units=0.0,
    )
    
    new_ev = build_event(
        model="claude-opus-4.6",
        provider="copilot",
        domain="code",
        decision_source="primary_default",
        turn_kind="complex_task",
        success=True,
        tokens_in=200,
        tokens_out=200,
    )
    
    # Write both events
    record_event(old_ev, store=isolated_store)
    record_event(new_ev, store=isolated_store)
    
    # Load with since=1 hour ago, should only get new event
    since_time = now - timedelta(hours=1)
    loaded = load_events(store=isolated_store, since=since_time)
    
    assert len(loaded) == 1
    assert loaded[0].model == "claude-opus-4.6"


def test_wrap_resolve_turn_route_records_success(isolated_store):
    """decora una fn fake, llama, verifica 1 evento grabado con success=True y latency_ms > 0"""
    call_count = 0
    
    @wrap_resolve_turn_route(store=isolated_store)
    def fake_resolve_turn(user_message, routing_config, primary):
        nonlocal call_count
        call_count += 1
        return {"model": "gpt-5-mini", "runtime": {"provider": "copilot"}, "label": None}
    
    result = fake_resolve_turn("hello", {}, {"model": "gpt-5-mini"})
    
    assert call_count == 1
    assert result == {"model": "gpt-5-mini", "runtime": {"provider": "copilot"}, "label": None}
    
    # Check event was recorded
    events = load_events(store=isolated_store)
    assert len(events) == 1
    ev = events[0]
    assert ev.success is True
    assert ev.latency_ms is not None
    assert ev.latency_ms > 0
    assert ev.turn_kind == "primary"
