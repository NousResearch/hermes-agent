"""Summary-failure classification for transport errors.

Sibling of ``test_context_compressor.py`` (kept separate because that file is at
its line cap — see ``scripts/check`` health rules).

``_classify_summary_failure`` sets two flags that overlap:

    timeout=… or "timeout" in err or "timed out" in err,
    streaming_closed=_is_connection_error(e) and not stall,

``_is_connection_error`` returns True for ``openai.APITimeoutError`` (an
``APIConnectionError`` subclass) *and* for the ``"timed out"`` substring, so a
plain deadline exhaustion sets BOTH. Every consumer of the pair used to assume
``streaming_closed`` meant "the peer dropped the connection". These tests pin the
two places where that assumption produced a wrong answer.
"""

import pytest
import openai
from unittest.mock import patch

from agent.context_compressor import (
    ContextCompressor,
    _classify_summary_failure,
)
from agent.auxiliary_client import CODEX_STREAM_STALL_MARKER

_REQ = None


def _msgs():
    return [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": "ok"},
    ]


class TestTransportFailureClassification:
    """A transport failure must be reported under its own class."""

    @pytest.mark.parametrize(
        "err,expected",
        [
            (openai.APITimeoutError(request=_REQ), "timed out"),
            (openai.APIConnectionError(message="Connection error.", request=_REQ),
             "closed stream prematurely"),
            # A Codex stream-guard stall is deliberately a ladder timeout, not a
            # transport error (#124077), even though it arrives as a connection error.
            (openai.APIConnectionError(
                message=f"upstream {CODEX_STREAM_STALL_MARKER}", request=_REQ),
             "closed stream prematurely"),
        ],
        ids=["api_timeout", "connection_error", "connection_error_with_stall_text"],
    )
    def test_transport_errors_report_the_right_failure_reason(self, err, expected):
        """``fallback_reason()`` must not label a deadline exhaustion as a mid-stream drop.

        The reason string is what the one-shot main-model retry logs and what an
        operator reads to decide what to do. "The peer closed the stream" points at
        the provider's connection handling; "the deadline expired" points at the
        summarizer route. Only the second is true for an ``APITimeoutError``.

        A plain ``APIConnectionError`` keeps reporting the premature close — that is
        genuinely what it is. The stall-marker case is pinned so this fix cannot
        silently reclassify it.
        """
        kind = _classify_summary_failure(err)
        assert kind.streaming_closed is True  # both are terminal network failures
        assert kind.fallback_reason() == expected

    def test_deadline_exhaustion_keeps_the_timeout_ladder(self):
        """A deadline exhaustion stays on the timeout ladder, not the 30s rung.

        A timeout is the structural repeat-offender class: a transcript too large to
        summarize inside the deadline fails identically every turn, and the
        60s→300s→900s ladder exists to stop re-burning the full timeout each time
        (#62452). This pins that invariant so a future reorder of the ``elif`` chain
        — where ``streaming_closed`` shares the 30s transient bucket and an
        ``APITimeoutError`` satisfies both predicates — cannot silently move a
        deadline exhaustion onto the short rung.

        Measured on one compressor instance across turns with no distinct summary
        model — the shape the ladder is designed for. The main-model fallback path
        deliberately clears the cooldown for its immediate retry, so a cross-turn
        streak is not observable through it.
        """
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="main-model", quiet_mode=True)

        with patch("agent.context_compressor.call_llm",
                   side_effect=openai.APITimeoutError(request=_REQ)), \
             patch("agent.context_compressor.time.monotonic", return_value=1000.0):
            assert c._generate_summary(_msgs()) is None
        assert c._consecutive_timeout_failures == 1
        # First rung of the ladder (60s), strictly above the 30s transient rung.
        assert c._summary_failure_cooldown_until - 1000.0 >= 60.0

        # A genuine premature close keeps the SHORT rung — different root cause.
        with patch("agent.context_compressor.call_llm",
                   side_effect=Exception("RemoteProtocolError: response ended prematurely")), \
             patch("agent.context_compressor.time.monotonic", return_value=2000.0):
            assert c._generate_summary(_msgs()) is None
        assert 0 < c._summary_failure_cooldown_until - 2000.0 < 60.0

    def test_classification_change_does_not_weaken_the_abort(self):
        """The terminal-abort behavior is unchanged: context is still preserved.

        Reordering the flags changes only the *diagnosis*, never the *outcome*.
        This is the discrimination check for the PR.
        """
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test", quiet_mode=True, protect_first_n=2,
                protect_last_n=2, abort_on_summary_failure=False,
            )
        msgs = [{"role": "system", "content": "sys"}] + [
            m for i in range(12) for m in (
                {"role": "user", "content": f"u{i} " + "x" * 200},
                {"role": "assistant", "content": f"a{i} " + "y" * 200},
            )
        ]
        with patch("agent.context_compressor.call_llm",
                   side_effect=openai.APITimeoutError(request=_REQ)):
            res = c.compress(msgs, current_tokens=999999, force=True)

        assert res == msgs
        assert c._last_summary_network_failure is True
        assert c._last_compress_aborted is True
        assert c._last_summary_fallback_used is False
        assert c._last_summary_dropped_count == 0
