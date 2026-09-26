"""Regression coverage for summary transport-failure classification (#124077).

The Codex Responses no-progress guard raises builtin TimeoutError with a "stream stalled"
message.  Timeout and connection taxonomies intentionally overlap in auxiliary transport
code, but summary compression must give timeout semantics precedence so the timeout ladder
and normal deterministic fallback remain reachable.
"""

from agent.context_compressor import _classify_summary_failure


_CODEX_STREAM_STALL = (
    "Codex auxiliary Responses stream stalled: no new output for 60.0s "
    "(last event=response.in_progress)"
)


def test_codex_stream_guard_timeout_is_timeout_not_terminal_network_failure():
    kind = _classify_summary_failure(TimeoutError(_CODEX_STREAM_STALL))

    assert kind.timeout is True
    assert kind.streaming_closed is False


def test_plain_connection_drop_remains_terminal_network_failure():
    kind = _classify_summary_failure(ConnectionError("peer closed connection unexpectedly"))

    assert kind.timeout is False
    assert kind.streaming_closed is True


def test_timeout_text_on_non_timeout_exception_keeps_existing_timeout_semantics():
    kind = _classify_summary_failure(RuntimeError("summary request timed out while reading response"))

    assert kind.timeout is True
    assert kind.streaming_closed is False


def test_stalled_word_alone_does_not_redefine_transport_taxonomy():
    kind = _classify_summary_failure(RuntimeError("summary parser stalled on malformed provider payload"))

    assert kind.timeout is False
    assert kind.streaming_closed is False
