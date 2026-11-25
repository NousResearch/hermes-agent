"""A summary call that fails after the host cancelled the compression must unwind, not count as a route failure.

The gateway's hygiene timeout cancels the compression while the summary call is still in flight. When that call
then fails, ``_on_summary_failure`` treated it like any other failure: it demoted the session's summary route to
the main model and, on the main route, armed a cooldown and stamped terminal failure flags, all for an attempt
nobody waits for any more.
"""

from __future__ import annotations

from unittest.mock import patch

from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.context_compressor import ContextCompressor

TURNS = [{"role": "user", "content": "hello " * 50}, {"role": "assistant", "content": "hi " * 50}]


def _compressor(summary_model: str) -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(model="main/model", quiet_mode=True)
        _ = c.context_length
    c.summary_model = summary_model
    c._summary_model_fallen_back = False
    return c


def _run(c: ContextCompressor, *, cancel_during_call: bool) -> tuple[int, bool]:
    """Run a real summary attempt whose provider call always fails; return (provider calls, unwound as a cancel)."""
    cancelled = False
    c._compression_cancelled_check = lambda: cancelled
    calls = 0

    def _failing_call(**_kw):
        nonlocal cancelled, calls
        calls += 1
        cancelled = cancel_during_call
        raise RuntimeError("upstream connection reset")

    with patch("agent.context_compressor.call_llm", side_effect=_failing_call):
        try:
            c._generate_summary(TURNS)
        except AuxiliaryExplicitCancellation:
            return calls, True
    return calls, False


def test_cancelled_attempt_keeps_the_summary_route():
    c = _compressor("summary/model")

    assert _run(c, cancel_during_call=True) == (1, True)
    assert c.summary_model == "summary/model"
    assert c._summary_model_fallen_back is False


def test_cancelled_attempt_on_the_main_route_leaves_no_failure_state():
    c = _compressor("")

    assert _run(c, cancel_during_call=True) == (1, True)
    assert c._summary_failure_cooldown_until == 0.0
    assert c._last_summary_error is None
    assert not getattr(c, "_last_summary_network_failure", False)


def test_live_attempt_still_falls_back_to_the_main_model_once():
    c = _compressor("summary/model")

    assert _run(c, cancel_during_call=False) == (2, False)
    assert c._summary_model_fallen_back is True
    assert c._summary_failure_cooldown_until > 0
