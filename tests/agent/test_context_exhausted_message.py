"""The context-exhausted error must describe the request, not just history (2026-09-14).

A WebUI session sent one ~84K-token message to a 65,536-token local model. History
was 44 tokens, so the old error read "Context length exceeded (44 tokens)" — true
of history, useless to the user.
"""

from agent.conversation_loop import format_context_exhausted_message


def test_reports_request_size_window_and_uncompressible_share():
    msg = format_context_exhausted_message(84_152, 44, 65_536)
    assert "84,152" in msg and "65,536" in msg
    assert "44" in msg  # history size still visible, but not the headline
    assert "84,108" in msg  # non-history share that compression cannot touch
    assert "smaller message" in msg or "larger-context" in msg


def test_unknown_window_does_not_crash():
    msg = format_context_exhausted_message(1_000, 500, None)
    assert "model's-token window" in msg and "1,000" in msg


def test_uncompressible_overflow_fails_over_only_while_chain_remains():
    from types import SimpleNamespace
    from agent.conversation_loop import uncompressible_overflow_can_failover

    assert uncompressible_overflow_can_failover(SimpleNamespace(_fallback_chain=[{"p": 1}], _fallback_index=0))
    assert not uncompressible_overflow_can_failover(SimpleNamespace(_fallback_chain=[{"p": 1}], _fallback_index=1))
    assert not uncompressible_overflow_can_failover(SimpleNamespace(_fallback_chain=[], _fallback_index=0))
    assert not uncompressible_overflow_can_failover(SimpleNamespace())
