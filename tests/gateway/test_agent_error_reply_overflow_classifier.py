"""_hmwa_agent_error_reply (the ``except Exception`` fallback reply) must reach the same
context-overflow verdict as ``is_context_overflow_failure_result`` — the classifier
``949724b321`` hoisted so the #1630 transcript skip and the failed-``agent_result`` user-facing
rewrite can never disagree. This except-handler path is a third, independent site: it fires when
``run_conversation`` raises instead of returning a failed result dict (auth/billing/content-policy/
internal-server exceptions all take this path), and it used to decide "Session too large" from
nothing but ``status_code in {400, 500}`` and a long history — no text classification at all,
reproducing the exact misclassification bug the shared classifier exists to prevent.
"""

import asyncio

from gateway.run import GatewayRunner


class _FakeAPIError(Exception):
    """A provider exception carrying the ``status_code`` attribute the except-handler reads."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)

    async def _noop_stop_typing(event, source):
        return None

    runner._hmwa_stop_typing_for_turn = _noop_stop_typing
    return runner


def _prepared(runner: GatewayRunner, history_len: int):
    # message_text=None skips the input-ownership/transcript-persistence block, which needs a
    # real SessionStore this test has no reason to build.
    return runner._PreparedTurn(
        [{"role": "user", "content": "x"}] * history_len, "", None, None, None, None,
    )


def _reply(e: Exception, history_len: int) -> str:
    runner = _runner()
    prepared = _prepared(runner, history_len)
    return asyncio.run(
        runner._hmwa_agent_error_reply(e, None, None, None, "fixture-session", prepared)
    )


def test_content_policy_400_on_long_session_keeps_its_own_reply():
    """A validation/content-policy 400 names no overflow phrase and no bare "400" digits — a long
    session must not steer it into the misleading "Session too large" rewrite."""
    reply = _reply(_FakeAPIError(400, "content flagged by moderation policy, please rephrase"), 138)
    assert "Session too large" not in reply
    assert "unexpected error" in reply


def test_generic_500_on_long_session_keeps_its_own_reply():
    """A bare 500 (no overflow phrase, no "400" substring) on a long session is the exact
    regression shape ``949724b321`` fixed for the other two call sites."""
    reply = _reply(_FakeAPIError(500, "internal server error, please retry"), 138)
    assert "Session too large" not in reply
    assert "unexpected error" in reply


def test_genuine_overflow_phrase_still_gets_the_compact_reply():
    reply = _reply(_FakeAPIError(400, "This model's maximum context length is 128000 tokens"), 138)
    assert "Session too large" in reply
    assert "/compact" in reply


def test_bare_400_envelope_on_long_session_still_gets_the_compact_reply():
    """Preserves the historical "any bare 400 on a long session" fallback for provider envelopes
    that carry no recognizable overflow phrase at all."""
    reply = _reply(_FakeAPIError(400, 'HTTP 400: {"object":"error","model":"deepseek-v4-flash"}'), 138)
    assert "Session too large" in reply


def test_short_session_400_never_gets_the_compact_reply():
    """The history_len > 50 gate is unchanged by this fix."""
    reply = _reply(_FakeAPIError(400, 'HTTP 400: {"object":"error"}'), 10)
    assert "Session too large" not in reply
    assert "request was rejected" in reply
