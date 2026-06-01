"""Scenario F: engine selector determinism (WP11).

Tests that select_call_engine returns the correct engine name in all cases,
including the critical default-to-turn_based guarantee.
"""
from __future__ import annotations

import pytest

from gateway.calls.native.streaming.engine import select_call_engine, TURN_BASED, STREAMING


def test_engine_turn_based_explicit():
    assert select_call_engine({"calls": {"native": {"engine": "turn_based"}}}) == TURN_BASED


def test_engine_streaming_explicit():
    assert select_call_engine({"calls": {"native": {"engine": "streaming"}}}) == STREAMING


def test_engine_default_when_missing():
    assert select_call_engine({}) == TURN_BASED
    assert select_call_engine({"calls": {}}) == TURN_BASED
    assert select_call_engine({"calls": {"native": {}}}) == TURN_BASED


def test_engine_unknown_falls_back():
    assert select_call_engine({"calls": {"native": {"engine": "bogus"}}}) == TURN_BASED


def test_engine_none_config():
    assert select_call_engine(None) == TURN_BASED


def test_engine_case_insensitive():
    assert select_call_engine({"calls": {"native": {"engine": "STREAMING"}}}) == STREAMING


def test_engine_whitespace_stripped():
    assert select_call_engine({"calls": {"native": {"engine": "  streaming  "}}}) == STREAMING
    assert select_call_engine({"calls": {"native": {"engine": "  turn_based  "}}}) == TURN_BASED


def test_engine_build_pipeline_turn_based_delegates():
    """build_native_pipeline delegates to turn_based_factory unchanged (zero behavior change)."""
    from gateway.calls.native.streaming.engine import build_native_pipeline

    sentinel = object()
    result = build_native_pipeline(
        {},
        turn_based_factory=lambda: sentinel,
    )
    assert result is sentinel


def _streaming_cfg():
    return {"calls": {"native": {"engine": "streaming"}}}


def test_streaming_missing_extra_raises_clear_error(monkeypatch):
    from gateway.calls.native.streaming import engine as eng

    monkeypatch.setattr(eng, "pipecat_available", lambda: False)
    with pytest.raises(eng.StreamingExtraNotInstalled) as ei:
        eng.build_native_pipeline(_streaming_cfg(), turn_based_factory=lambda: object())
    assert "simplex-streaming" in str(ei.value)


def test_streaming_present_defers_until_transport(monkeypatch):
    from gateway.calls.native.streaming import engine as eng
    from gateway.calls.native.streaming.pipecat_transport import PipecatIntegrationDeferred

    monkeypatch.setattr(eng, "pipecat_available", lambda: True)
    with pytest.raises(PipecatIntegrationDeferred):
        eng.build_native_pipeline(_streaming_cfg(), turn_based_factory=lambda: object())


def test_turn_based_default_returns_factory_product():
    from gateway.calls.native.streaming import engine as eng

    sentinel = object()
    result = eng.build_native_pipeline({}, turn_based_factory=lambda: sentinel)
    assert result is sentinel
