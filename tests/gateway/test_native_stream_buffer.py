"""Behavioral tests for the optional Rust-backed stream accumulator.

This slice keeps the gateway's Python control plane intact while providing a
small, optional native data-plane backend for append-heavy stream storage.
"""

from __future__ import annotations

import pytest

from gateway.native_stream_buffer import StreamBuffer, rust_stream_buffer_available


def test_python_backend_preserves_active_and_ledger_separately() -> None:
    buffer = StreamBuffer(backend="python")

    buffer.append("hello")
    buffer.append(" world")
    buffer.set_active("tail")

    assert buffer.backend == "python"
    assert buffer.active() == "tail"
    assert buffer.ledger() == "hello world"


def test_python_backend_clear_releases_both_views() -> None:
    buffer = StreamBuffer(backend="python")
    buffer.append("payload")

    buffer.clear()

    assert buffer.active() == ""
    assert buffer.ledger() == ""


@pytest.mark.skipif(not rust_stream_buffer_available(), reason="Rust stream buffer is not built")
def test_rust_backend_matches_stream_contract() -> None:
    buffer = StreamBuffer(backend="rust")

    buffer.append("前缀")
    buffer.append(" + suffix")
    buffer.set_active("tail")

    assert buffer.backend == "rust"
    assert buffer.active() == "tail"
    assert buffer.ledger() == "前缀 + suffix"

    buffer.clear()
    assert buffer.active() == ""
    assert buffer.ledger() == ""


def test_forced_rust_backend_fails_closed_without_artifact(monkeypatch) -> None:
    monkeypatch.setattr("gateway.native_stream_buffer._load_rust_api", lambda: None)

    with pytest.raises(RuntimeError, match="Rust stream buffer is unavailable"):
        StreamBuffer(backend="rust")


def test_consumer_compatibility_properties_use_the_python_backend() -> None:
    from gateway.stream_consumer import GatewayStreamConsumer

    consumer = object.__new__(GatewayStreamConsumer)
    consumer._stream_buffer = StreamBuffer(backend="python")
    consumer._tool_progress_lines = []
    consumer._tool_progress_active = False
    consumer._accumulated = "active"
    consumer._stream_ledger = "ledger"

    consumer._append_accumulated(" + delta")

    assert consumer._stream_text_buffer().active() == "active + delta"
    assert consumer._stream_text_buffer().ledger() == "ledger + delta"


@pytest.mark.skipif(not rust_stream_buffer_available(), reason="Rust stream buffer is not built")
def test_consumer_uses_rust_buffer_when_available() -> None:
    from gateway.stream_consumer import GatewayStreamConsumer

    consumer = object.__new__(GatewayStreamConsumer)
    consumer._tool_progress_lines = []
    consumer._tool_progress_active = False

    consumer._append_accumulated("前缀")

    assert consumer._stream_text_buffer().backend == "rust"
    assert consumer._accumulated == "前缀"
    assert consumer._stream_ledger == "前缀"
