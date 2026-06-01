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


def test_streaming_without_aiortc_raises_clear_error(monkeypatch):
    """Slice 6: STREAMING now depends on aiortc (not pipecat). Absent → clear raise."""
    from gateway.calls.native.streaming import engine as eng

    monkeypatch.setattr(eng, "_aiortc_available", lambda: False)
    with pytest.raises(eng.StreamingExtraNotInstalled) as ei:
        eng.build_native_pipeline(_streaming_cfg(), turn_based_factory=lambda: object())
    msg = str(ei.value)
    assert "aiortc" in msg
    assert "simplex-native-calls" in msg


def test_streaming_with_aiortc_unavailable_via_monkeypatch_constructs(monkeypatch):
    """With aiortc 'available' (monkeypatched), the STREAMING branch un-defers and
    constructs a StreamingPipeline using the fake cognitive ports (no pipecat,
    no real aiortc import — the core is pure asyncio)."""
    from gateway.calls.native.streaming import engine as eng
    from gateway.calls.native.streaming.aiortc_transport import StreamingPipeline

    monkeypatch.setattr(eng, "_aiortc_available", lambda: True)
    pipe = eng.build_native_pipeline(
        _streaming_cfg(), turn_based_factory=lambda: object(), cognitive="fake"
    )
    assert isinstance(pipe, StreamingPipeline)
    assert pipe.is_streaming is True
    assert hasattr(pipe, "process_pcm16")


def test_turn_based_default_returns_factory_product():
    from gateway.calls.native.streaming import engine as eng

    sentinel = object()
    result = eng.build_native_pipeline({}, turn_based_factory=lambda: sentinel)
    assert result is sentinel
