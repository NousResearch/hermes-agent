"""Compaction serialization must be bounded and cancellable (issue #99255).

``_generate_summary`` consults the host-owned cooperative cancellation signal
once (right before the prompt is built) and again after the auxiliary call
returns.  Everything between those two points — ``_serialize_for_summary``,
which strict-redacts every message in the compression window — ran with no
cancellation checkpoint at all, and handed the redactor each message body *in
full* before truncating the result to ``_CONTENT_MAX``.

On a large session with MB-scale tool results that combination let a worker
whose host had already timed out (fence cancelled at
``compression.context_total_ceiling_seconds``) keep burning CPU for hours,
holding the GIL and starving the gateway event loop.

These tests pin both halves of the contract:

* the serializer polls the cancellation signal per message and unwinds with
  ``AuxiliaryExplicitCancellation`` (a ``BaseException``, so no broad
  ``except Exception`` can swallow it), and
* the bytes handed to the redactor per message are bounded, so that
  per-message checkpoint is actually reachable on MB-scale bodies.
"""

import time
from unittest.mock import patch

import pytest

import agent.context_compressor as cc
from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.context_compressor import ContextCompressor

SECRET = "sk-proj-" + ("a" * 40)


def _compressor() -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100000,
    ):
        return ContextCompressor(model="test/model", quiet_mode=True)


def _turns(n: int, body: str):
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"step {i}"})
        out.append(
            {"role": "tool", "tool_call_id": f"call_{i}", "content": body}
        )
    return out


@pytest.fixture
def fed(monkeypatch):
    """Count the bytes handed to the compaction redactor."""
    counter = {"bytes": 0, "calls": 0}
    real = cc._redact_compaction_text

    def _counting(text):
        counter["bytes"] += len(text or "")
        counter["calls"] += 1
        return real(text)

    monkeypatch.setattr(cc, "_redact_compaction_text", _counting)
    return counter


class TestSerializerIsCancellable:
    def test_cancelled_host_stops_the_serializer(self):
        """A fence cancelled mid-serialization must abort the walk."""
        comp = _compressor()
        turns = _turns(50, "payload " * 100)

        seen = {"n": 0}

        def _check():
            seen["n"] += 1
            # Host gives up after the 5th message.
            return seen["n"] > 5

        comp._compression_cancelled_check = _check

        with pytest.raises(AuxiliaryExplicitCancellation):
            comp._serialize_for_summary(turns)

        # It must stop near the cancellation point, not walk all 100 rows.
        assert seen["n"] <= 10, f"kept walking after cancel: {seen['n']} checks"

    def test_cancellation_is_checked_once_per_message(self):
        """The checkpoint is inside the loop, not merely at its entry."""
        comp = _compressor()
        turns = _turns(20, "payload")
        seen = {"n": 0}

        def _check():
            seen["n"] += 1
            return False

        comp._compression_cancelled_check = _check
        comp._serialize_for_summary(turns)

        assert seen["n"] >= len(turns), (
            f"only {seen['n']} checks for {len(turns)} messages — the "
            "serializer is not polling per message"
        )

    def test_uncancelled_serialization_is_unaffected(self):
        """No signal installed → ordinary behavior, no exception."""
        comp = _compressor()
        turns = _turns(3, "hello world")
        out = comp._serialize_for_summary(turns)
        assert "[TOOL RESULT call_0]" in out
        assert "hello world" in out

    def test_cancellation_escapes_broad_exception_handlers(self):
        """AuxiliaryExplicitCancellation must survive `except Exception`."""
        comp = _compressor()
        comp._compression_cancelled_check = lambda: True

        with pytest.raises(AuxiliaryExplicitCancellation):
            try:
                comp._serialize_for_summary(_turns(5, "x"))
            except Exception:  # noqa: BLE001 - deliberately broad
                pytest.fail("cancellation was swallowed by `except Exception`")


class TestRedactionInputIsBounded:
    def test_large_tool_body_does_not_flood_the_redactor(self, fed):
        """Only ~_CONTENT_MAX chars survive; don't redact a megabyte to get them."""
        comp = _compressor()
        big = "A" * 1_000_000 + "=="
        comp._serialize_for_summary([
            {"role": "tool", "tool_call_id": "c1", "content": big}
        ])

        assert fed["bytes"] < 100_000, (
            f"redactor was handed {fed['bytes']:,} chars to produce at most "
            f"{comp._CONTENT_MAX:,} — redact-before-truncate is still live"
        )

    def test_large_tool_args_do_not_flood_the_redactor(self, fed):
        comp = _compressor()
        comp._serialize_for_summary([
            {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [
                    {
                        "function": {
                            "name": "terminal",
                            "arguments": "B" * 1_000_000 + "==",
                        }
                    }
                ],
            }
        ])
        assert fed["bytes"] < 100_000, (
            f"tool arguments handed {fed['bytes']:,} chars to the redactor"
        )

    def test_non_string_content_still_reaches_the_redactor_coercion(self):
        """Dict/int bodies must not die on a slice in the input bound.

        ``redact_sensitive_text`` coerces non-str values itself; the bound
        has to pass them through rather than call len()/slice on them.
        """
        comp = _compressor()
        out = comp._serialize_for_summary([
            {"role": "user", "content": 12345},
            {"role": "tool", "tool_call_id": "c1", "content": {"a": "b"}},
            {
                "role": "assistant",
                "content": "x",
                "tool_calls": [
                    {"function": {"name": "t", "arguments": {"k": "v"}}}
                ],
            },
        ])
        assert "12345" in out
        assert "[TOOL RESULT c1]" in out

    def test_bounded_input_keeps_the_serialized_shape(self):
        """Truncation markers and labels survive the input bound."""
        comp = _compressor()
        big = "A" * 1_000_000
        out = comp._serialize_for_summary([
            {"role": "tool", "tool_call_id": "c1", "content": big}
        ])
        assert out.startswith("[TOOL RESULT c1]: ")
        assert "...[truncated]..." in out
        assert len(out) < comp._CONTENT_MAX + 200

    def test_redactor_load_is_independent_of_raw_body_size(self, fed):
        """Total redactor input scales with message COUNT, not transcript size.

        This is the invariant that decouples compaction prep cost from how
        big the tool outputs were.  Asserted in bytes rather than seconds so
        it does not depend on the redactor's own complexity class (the
        quadratic assignment scan is tracked separately in #99265 / #91672).
        """
        comp = _compressor()
        n = 4
        small = _turns(n, "A" * 1_000)
        huge = _turns(n, "A" * 500_000)

        comp._serialize_for_summary(small)
        small_bytes = fed["bytes"]

        fed["bytes"] = 0
        comp._serialize_for_summary(huge)
        huge_bytes = fed["bytes"]

        window = comp._REDACT_INPUT_HEAD + comp._REDACT_INPUT_TAIL
        # 500x more raw text must not mean 500x more redaction work.
        assert huge_bytes <= n * (window + 64) + small_bytes, (
            f"redactor load grew with body size: {small_bytes:,} -> "
            f"{huge_bytes:,} bytes for a 500x larger transcript"
        )

    def test_cancellation_latency_is_bounded_by_one_message(self):
        """After the host cancels, the worker stops within one message."""
        comp = _compressor()
        turns = _turns(200, "authorization=Bearer abc\n" + ("A" * 4_000) + "==")
        calls = {"n": 0}

        def _check():
            calls["n"] += 1
            return calls["n"] > 2

        comp._compression_cancelled_check = _check
        t0 = time.perf_counter()
        with pytest.raises(AuxiliaryExplicitCancellation):
            comp._serialize_for_summary(turns)
        elapsed = time.perf_counter() - t0

        # Without the checkpoint this walks all 400 rows; with it, the walk
        # ends on the 3rd. Bound the wall clock generously — the point is
        # "one message", not a performance number.
        assert calls["n"] <= 4
        assert elapsed < 60.0, f"cancel took {elapsed:.1f}s to take effect"


class TestBoundingNeverLeaksSecrets:
    """The input bound must never let a secret reach the summarizer prompt."""

    @pytest.fixture(autouse=True)
    def _redaction_globally_disabled(self, monkeypatch):
        # force=True at the compaction boundary must still win.
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)

    def test_secret_at_head_of_huge_body_is_redacted(self):
        comp = _compressor()
        body = f"token={SECRET}\n" + ("A" * 1_000_000)
        out = comp._serialize_for_summary([
            {"role": "tool", "tool_call_id": "c1", "content": body}
        ])
        assert SECRET not in out

    def test_secret_at_tail_of_huge_body_is_redacted(self):
        comp = _compressor()
        body = ("A" * 1_000_000) + f"\ntoken={SECRET}"
        out = comp._serialize_for_summary([
            {"role": "tool", "tool_call_id": "c1", "content": body}
        ])
        assert SECRET not in out

    def test_secret_in_huge_tool_args_is_redacted(self):
        comp = _compressor()
        out = comp._serialize_for_summary([
            {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [
                    {
                        "function": {
                            "name": "terminal",
                            "arguments": f'{{"cmd":"export token={SECRET}"}}'
                            + ("A" * 500_000),
                        }
                    }
                ],
            }
        ])
        assert SECRET not in out
