"""Regression coverage for summary transport-failure classification (#124077).

The Codex Responses no-progress guard raises a builtin ``TimeoutError`` whose message reads
``"Codex auxiliary Responses stream stalled: no new output for 60.0s ..."``. That text contains
neither ``"timeout"`` nor ``"timed out"``, so ``_classify_summary_failure`` used to miss the
timeout class while ``_is_connection_error`` matched the exception type name and flagged the
failure as a terminal premature-stream close. The terminal flag armed an unconditional
compression abort that bypassed the retry ladder and the deterministic fallback, and on
turn-start preflight compression that ends as "Auto-resetting session after compression
exhaustion" — the session is wiped.

These tests pin the timeout-first precedence: a timeout stays on the timeout ladder with the
deterministic fallback reachable, and only genuine premature closes are terminal.
"""

from agent.auxiliary_client import _is_connection_error, _is_timeout_error
from agent.context_compressor import ContextCompressor, _classify_summary_failure

_CODEX_STREAM_STALL = (
    "Codex auxiliary Responses stream stalled: no new output for 60.0s (67.7s elapsed)"
)


class _StatusError(RuntimeError):
    """Stands in for a provider error object carrying an HTTP status code."""

    def __init__(self, status_code: int):
        super().__init__(f"upstream returned {status_code}")
        self.status_code = status_code


def _compressor() -> ContextCompressor:
    return ContextCompressor(
        model="test/model", quiet_mode=True, config_context_length=100_000
    )


def test_codex_stream_guard_timeout_is_timeout_not_terminal_network_failure():
    kind = _classify_summary_failure(TimeoutError(_CODEX_STREAM_STALL))

    assert kind.timeout is True
    assert kind.streaming_closed is False
    assert kind.json_decode is False


def test_codex_stream_guard_timeout_is_recognised_by_the_transport_owner():
    """The stall is recognised by the shared helper, so the compressor must not rebuild the taxonomy."""
    assert _is_timeout_error(TimeoutError(_CODEX_STREAM_STALL)) is True
    # The transport helper intentionally overlaps timeouts — that overlap is exactly why the
    # compressor has to give the timeout class precedence over streaming_closed.
    assert _is_connection_error(TimeoutError(_CODEX_STREAM_STALL)) is True


def test_plain_connection_drop_remains_terminal_network_failure():
    kind = _classify_summary_failure(ConnectionError("Connection error."))

    assert kind.timeout is False
    assert kind.streaming_closed is True


def test_premature_close_message_remains_terminal_network_failure():
    import httpx

    kind = _classify_summary_failure(
        httpx.RemoteProtocolError("peer closed connection unexpectedly")
    )

    assert kind.timeout is False
    assert kind.streaming_closed is True


def test_timeout_text_on_non_timeout_exception_keeps_existing_timeout_semantics():
    kind = _classify_summary_failure(
        RuntimeError("summary request timed out after 120s")
    )

    assert kind.timeout is True
    assert kind.streaming_closed is False


def test_generic_stalled_word_does_not_redefine_the_taxonomy():
    """A 'stalled' message on a non-timeout type is not a timeout: only the transport owner decides."""
    kind = _classify_summary_failure(
        RuntimeError("summary parser stalled on malformed payload")
    )

    assert kind.timeout is False
    assert kind.streaming_closed is False


def test_http_gateway_timeout_status_stays_a_timeout_not_terminal():
    for status in (408, 429, 502, 504):
        exc = _StatusError(status)
        kind = _classify_summary_failure(exc)

        assert kind.timeout is True, status
        assert kind.streaming_closed is False, status


def test_codex_stall_arms_timeout_ladder_without_terminal_abort(monkeypatch):
    compressor = _compressor()
    monkeypatch.setattr(
        "agent.context_compressor.call_llm",
        lambda *a, **kw: (_ for _ in ()).throw(TimeoutError(_CODEX_STREAM_STALL)),
    )

    result = compressor._on_summary_failure(
        TimeoutError(_CODEX_STREAM_STALL),
        turns_to_summarize=[
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ],
        focus_topic=None,
        memory_context="",
    )

    assert result is None
    assert compressor._consecutive_timeout_failures == 1
    assert compressor._last_summary_network_failure is False
    assert compressor._last_summary_auth_failure is False
    # The default gate stays open so the deterministic fallback path is reachable instead of an
    # unconditional terminal abort.
    assert compressor._abort_on_summary_failure({}, 3, None) is False
