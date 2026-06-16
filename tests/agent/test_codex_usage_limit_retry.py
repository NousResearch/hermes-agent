from agent.conversation_loop import _is_usage_limit_exhaustion


class DummyError(Exception):
    pass


def test_usage_limit_exhaustion_detects_codex_429_body():
    exc = DummyError("HTTP 429: {'error': {'type': 'usage_limit_reached', 'message': 'The usage limit has been reached'}}")
    assert _is_usage_limit_exhaustion(exc)


def test_usage_limit_exhaustion_detects_context_reason():
    assert _is_usage_limit_exhaustion(
        DummyError("rate limit"),
        {"reason": "goUsageLimit", "message": "The usage limit has been reached"},
    )


def test_usage_limit_exhaustion_ignores_plain_transient_429():
    assert not _is_usage_limit_exhaustion(DummyError("HTTP 429: try again later"))
