"""Behavioral coverage for the optional Rust-backed stream accumulator.

The native component owns only append-heavy stream storage. Delivery policy,
rendering, and lifecycle remain in the Python gateway.
"""

from __future__ import annotations

from typing import cast

import pytest

from gateway.native_stream_buffer import StreamBuffer, rust_stream_buffer_available


def test_python_backend_keeps_active_and_ledger_views_independent() -> None:
    buffer = StreamBuffer(backend="python")

    buffer.append("hello")
    buffer.append(" 世界")
    buffer.set_active("tail")

    assert buffer.backend == "python"
    assert buffer.active() == "tail"
    assert buffer.ledger() == "hello 世界"


def test_python_backend_clear_resets_both_views() -> None:
    buffer = StreamBuffer(backend="python")
    buffer.append("payload")

    buffer.clear()

    assert buffer.active() == ""
    assert buffer.ledger() == ""


def test_consumer_compatibility_surface_preserves_append_contract() -> None:
    from gateway.stream_consumer import GatewayStreamConsumer

    consumer = object.__new__(GatewayStreamConsumer)
    consumer = cast(GatewayStreamConsumer, consumer)
    consumer._stream_buffer = StreamBuffer(backend="python")
    consumer._tool_progress_lines = []
    consumer._tool_progress_active = False
    consumer._accumulated = "active"
    consumer._stream_ledger = "ledger"

    consumer._append_accumulated(" + delta")

    assert consumer._accumulated == "active + delta"
    assert consumer._stream_ledger == "ledger + delta"


@pytest.mark.skipif(not rust_stream_buffer_available(), reason="Rust stream buffer is not built")
def test_rust_backend_matches_python_stream_contract() -> None:
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


@pytest.mark.skipif(not rust_stream_buffer_available(), reason="Rust stream buffer is not built")
def test_consumer_uses_rust_buffer_for_append_fast_path() -> None:
    from gateway.stream_consumer import GatewayStreamConsumer

    consumer = cast(GatewayStreamConsumer, object.__new__(GatewayStreamConsumer))
    consumer._tool_progress_lines = []
    consumer._tool_progress_active = False

    consumer._append_accumulated("前缀")

    assert consumer._stream_text_buffer().backend == "rust"
    assert consumer._accumulated == "前缀"
    assert consumer._stream_ledger == "前缀"


def test_native_write_failure_demotes_without_losing_delta(monkeypatch) -> None:
    class FailingApi:
        def new(self):
            return object()

        def free(self, _handle):
            return None

        def append(self, _handle, _data, _length):
            return 0

    monkeypatch.setattr("gateway.native_stream_buffer._load_rust_api", lambda: FailingApi())

    buffer = StreamBuffer(backend="rust")
    buffer.append("delta")

    assert buffer.backend == "python"
    assert buffer.active() == "delta"
    assert buffer.ledger() == "delta"
