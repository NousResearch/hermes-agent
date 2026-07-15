"""Tests for cross-session _previous_summary contamination bug (#38788).

ContextCompressor._previous_summary is an instance variable that stores the
previous compaction summary for iterative updates.  It is cleared by
on_session_reset() which is called for /new and /reset, but NOT when a cron
session ends naturally.  A cron session's compaction sets _previous_summary,
then the cron session ends.  A subsequent live messaging session inherits this
stale summary, and _generate_summary() injects it as "PREVIOUS SUMMARY:" into
the summarizer prompt — contaminating the live session's context.

Fix: compress() guards against this by clearing _previous_summary when no
handoff summary is found in the current messages.
"""

import sys
import types
from unittest.mock import patch


# Stub out optional heavy dependencies not installed in the test environment
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from agent.context_compressor import ContextCompressor


def _make_compressor():
    """Build a ContextCompressor with enough state to pass compress() guards."""
    c = ContextCompressor.__new__(ContextCompressor)
    c.quiet_mode = True
    c.model = "test/model"
    c.provider = "test"
    c.base_url = "http://test"
    c.api_key = "test-key"
    c.api_mode = ""
    c.context_length = 128000
    c.threshold_tokens = 64000
    c.threshold_percent = 0.50
    c.tail_token_budget = 20000
    c.protect_last_n = 12
    c.summary_model = ""
    c.last_prompt_tokens = 100000
    c.last_completion_tokens = 0
    c._summary_failure_cooldown_until = 0.0
    c._max_compaction_summary_tokens = 0
    c.summary_budget_tokens = 0
    c.abort_on_summary_failure = False
    c._last_compress_aborted = False
    c._summary_model_fallen_back = False
    c.compression_count = 0
    c._context_probed = False
    c._last_compression_savings_pct = 100.0
    c._ineffective_compression_count = 0
    c._last_summary_error = None
    c._last_summary_dropped_count = 0
    c._last_summary_fallback_used = False
    c._last_aux_model_failure_error = None
    c._last_aux_model_failure_model = None
    c.last_real_prompt_tokens = 0
    c.last_compression_rough_tokens = 0
    c.last_rough_tokens_when_real_prompt_fit = 0
    c.awaiting_real_usage_after_compression = False
    return c


def _conversation_without_handoff(n_exchanges=12):
    """Build message list with no compaction handoff in it."""
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": f"Question {i}"})
        msgs.append({"role": "assistant", "content": f"Answer {i}"})
    return msgs


def _conversation_with_handoff(n_exchanges=12):
    """Build message list WITH a compaction handoff in protected head."""
    from agent.context_compressor import SUMMARY_PREFIX
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    msgs.append({"role": "user", "content": SUMMARY_PREFIX + "\nPrevious summary."})
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": f"Question {i}"})
        msgs.append({"role": "assistant", "content": f"Answer {i}"})
    return msgs


def test_stale_previous_summary_cleared_when_no_handoff():
    """Cross-session guard: stale _previous_summary cleared when no handoff."""
    c = _make_compressor()
    # Simulate state left by a prior cron session's compaction
    c._previous_summary = "STALE CRON SUMMARY - this must not leak"

    messages = _conversation_without_handoff()

    with patch.object(c, "_generate_summary",
                      return_value="[CONTEXT COMPACTION] Fresh summary."):
        result = c.compress(messages)

    assert c._previous_summary is None, (
        "compress() must clear stale _previous_summary when no handoff "
        f"summary exists in current messages. Got: {c._previous_summary!r}"
    )
    assert result != messages
    assert any(
        "[CONTEXT COMPACTION]" in (m.get("content", "") or "") for m in result
    )


def test_previous_summary_preserved_when_handoff_found():
    """When a handoff IS found, _previous_summary should be preserved for
    iterative update within the same session."""
    c = _make_compressor()
    c._previous_summary = "Summary from earlier compaction in same session"

    messages = _conversation_with_handoff()

    with patch.object(c, "_generate_summary",
                      return_value="[CONTEXT COMPACTION] Updated summary."):
        c.compress(messages)

    # When a handoff IS found, the staleness guard must NOT fire.
    # _previous_summary should be updated, not cleared.
    assert c._previous_summary is not None, (
        "compress() must NOT clear _previous_summary when handoff summary "
        "exists in current messages"
    )


def test_no_false_positive_when_previous_summary_already_none():
    """When _previous_summary is already None and no handoff found, nothing
    should break (the guard is a no-op in this case)."""
    c = _make_compressor()
    c._previous_summary = None

    messages = _conversation_without_handoff()

    with patch.object(c, "_generate_summary",
                      return_value="[CONTEXT COMPACTION] Fresh summary."):
        c.compress(messages)

    # Should still be None — guard is no-op
    assert c._previous_summary is None
