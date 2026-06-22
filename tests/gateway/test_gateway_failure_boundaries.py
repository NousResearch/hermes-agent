"""Gateway failure-boundary tests for user-facing safe fallback messages."""

from gateway.run import _normalize_empty_agent_response


def test_normalize_empty_agent_response_context_overflow_returns_bounded_guidance():
    """Context-window failures should return the new bounded guidance, not old/reset guidance or raw provider text."""
    agent_result = {
        "failed": True,
        "error": "HTTP 400: maximum context window exceeded; prompt is too long; request_id=req_ctx_123 traceback=stack boom",
    }

    message = _normalize_empty_agent_response(
        agent_result,
        "",
        history_len=100,
    )

    lowered = message.lower()
    assert "conversation got too large" in lowered or "fresh, short message" in lowered
    assert "/compact" not in message
    assert "/reset" not in message
    assert "model's context window" not in lowered
    assert "HTTP 400" not in message
    assert "maximum context window" not in lowered
    assert "traceback" not in lowered
    assert "request_id" not in lowered
