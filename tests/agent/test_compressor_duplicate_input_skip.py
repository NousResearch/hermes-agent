"""Duplicate-input summary skip: only an attempt that actually delivered a compression
may arm it.

The skip exists so a caller re-presenting a transcript the compressor already
summarized does not pay for the summary LLM again. It must NOT fire for an input
whose earlier attempt was abandoned before it produced anything: the stall-recovery
path (agent/conversation_compression.py, #78981) deliberately re-runs the SAME worker
with the SAME messages on the configured fallback chain, and the host's
commit-refusal path restores attempt state so the next attempt recomputes it.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import (
    _restore_compressor_attempt_state,
    _snapshot_compressor_attempt_state,
    run_compress_context_with_progress_timeout,
)

CURRENT_TOKENS = 90_000
CHAIN_ENTRY = {
    "provider": "custom",
    "model": "backup-summarizer",
    "base_url": "https://fallback.invalid/v1",
    "api_key": "sk-fallback",
    "timeout": 5,
}


def _ok_response(content="SUMMARY BODY: earlier turns condensed. " + "s" * 200):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _make_compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(
            model="main-model", quiet_mode=True, threshold_percent=0.50,
            protect_first_n=2, protect_last_n=2, summary_model_override="aux-summarizer",
        )
        _ = compressor.context_length
        return compressor


def _transcript(n=30, size=2000):
    messages = [{"role": "system", "content": "You are Hermes."}]
    for i in range(n):
        messages.append({"role": "user", "content": f"u{i} " + "x" * size})
        messages.append({"role": "assistant", "content": f"a{i} " + "y" * size})
    return messages


def _failure_class(compressor: ContextCompressor):
    return (compressor._last_compression_telemetry or {}).get("failure_class")


# ---------------------------------------------------------------------------
# The intended skip: the caller re-presents a transcript already summarized
# ---------------------------------------------------------------------------


def test_repeated_successful_summary_input_is_skipped():
    compressor = _make_compressor()
    messages = _transcript()
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        compressed = compressor.compress(messages, current_tokens=CURRENT_TOKENS)
        assert len(calls) == 1
        assert compressed is not messages, "the first call must produce a compression"

        again = compressor.compress(messages, current_tokens=CURRENT_TOKENS)

    assert again is messages, "an unchanged, already-summarized input is handed back untouched"
    assert len(calls) == 1, "the duplicate input must not pay for a second summary call"
    assert _failure_class(compressor) == "duplicate_input_skipped"


def test_reordered_transcript_is_not_a_duplicate():
    """Order is part of the identity: a reordered thread must be summarized."""
    compressor = _make_compressor()
    messages = _transcript(n=6)
    reordered = messages[:1] + list(reversed(messages[1:]))
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        compressor.compress(messages, current_tokens=CURRENT_TOKENS)
        compressor.compress(reordered, current_tokens=CURRENT_TOKENS)

    assert len(calls) == 2, "a reordered transcript is a new input, never a duplicate"


# ---------------------------------------------------------------------------
# An abandoned attempt must not arm the skip
# ---------------------------------------------------------------------------


def test_stalled_attempt_retry_resummarizes_on_the_fallback_route():
    """The stall-fallback retry re-runs the same worker with the same messages.

    Nothing was published by the stalled primary, so the retry MUST reach the summary
    LLM on the pinned fallback route instead of reporting "no compression".
    """
    release = threading.Event()
    messages = _transcript()
    compressor = _make_compressor()
    calls = []
    timeouts = []

    def _fake_call_llm(**kwargs):
        index = len(calls)
        calls.append(kwargs)
        if index == 0:
            # Provider holds the connection open: zero tokens, zero fence progress.
            release.wait(timeout=30)
            return _ok_response("LATE PRIMARY SUMMARY")
        return _ok_response("FALLBACK SUMMARY BODY " + "f" * 200)

    def worker(fence):
        out = compressor.compress(messages, current_tokens=CURRENT_TOKENS)
        if out is messages or not fence.begin_commit():
            return messages, "degraded-prompt"
        try:
            return out, "summarized-prompt"
        finally:
            fence.finish_commit()

    try:
        with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm), patch(
            "agent.auxiliary_client._get_auxiliary_task_config", return_value={"fallback_chain": [CHAIN_ENTRY]},
        ):
            out_messages, out_prompt = run_compress_context_with_progress_timeout(
                worker=worker, messages=messages, system_prompt_fallback="degraded-prompt",
                idle_timeout_seconds=0.3, total_ceiling_seconds=1.5,
                on_timeout=lambda *args: timeouts.append(args),
            )
    finally:
        release.set()

    assert len(calls) == 2, (
        "the stalled attempt's retry must re-run the summary; dedup skipped it"
    )
    assert calls[1].get("provider") == "custom", "the retry must use the pinned fallback route"
    assert out_messages is not messages, "the fallback route's compression must be published"
    assert out_prompt == "summarized-prompt"
    assert not timeouts, "no continue-without-compression degrade after a recovery"


def test_rolled_back_attempt_does_not_arm_the_skip():
    """Host rollback (commit refused) restores attempt state, including the fingerprint."""
    compressor = _make_compressor()
    messages = _transcript()
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        snapshot = _snapshot_compressor_attempt_state(compressor)
        compressor.compress(messages, current_tokens=CURRENT_TOKENS)
        assert len(calls) == 1
        _restore_compressor_attempt_state(compressor, snapshot)
        compressor.compress(messages, current_tokens=CURRENT_TOKENS)

    assert len(calls) == 2, (
        "an attempt the host rolled back never delivered a compression, so the next "
        "attempt on the same input must run the summary"
    )
