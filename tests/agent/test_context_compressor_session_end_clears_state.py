"""Tests for on_session_end() clearing all per-session compressor state.

Bug: on_session_end() (added in #38788) only cleared _previous_summary, but
on_session_reset() clears 14+ per-session variables. When a session ends
(cron exit, gateway expiry, session-id rotation) and the compressor instance
is reused, these stale values survive:

- _ineffective_compression_count: can suppress compression in next session
- _summary_failure_cooldown_until: can block summary generation
- _last_compress_aborted: can make callers think compression is aborted
- _last_aux_model_failure_*: can surface stale error warnings
- _last_summary_dropped_count / _last_summary_fallback_used: misleading warnings
- _context_probed / _context_probe_persistable: stale context probe state

Fix: on_session_end() now clears all per-session state, matching
on_session_reset()'s surface.
"""

import sys
import types
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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
    c.last_total_tokens = 100000
    c._summary_failure_cooldown_until = 0.0
    c._max_compaction_summary_tokens = 0
    c.summary_budget_tokens = 0
    c.abort_on_summary_failure = False
    c._last_compress_aborted = False
    c._summary_model_fallen_back = False
    c.compression_count = 0
    c._context_probed = False
    c._context_probe_persistable = False
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
    c._previous_summary = None
    c._summary_has_user_turn = None
    return c


def _simulate_cron_session_state(c):
    """Simulate per-session state that a cron compaction would leave behind."""
    c._previous_summary = "Cron session summary that must not leak"
    c._summary_has_user_turn = False
    c._last_summary_error = "Cron session summary error"
    c._last_summary_dropped_count = 5
    c._last_summary_fallback_used = True
    c._last_aux_model_failure_error = "Cron aux model error"
    c._last_aux_model_failure_model = "cron-model/v1"
    c._last_compression_savings_pct = 3.0
    c._ineffective_compression_count = 2
    c._summary_failure_cooldown_until = 9999999999.0
    c._last_compress_aborted = True
    c._context_probed = True
    c._context_probe_persistable = True
    c.last_real_prompt_tokens = 50000
    c.last_compression_rough_tokens = 60000
    c.last_rough_tokens_when_real_prompt_fit = 55000
    c.awaiting_real_usage_after_compression = True






def test_ineffective_compression_count_does_not_leak_across_sessions():
    """A cron session that hit ineffective compression limits must not
    suppress compression in a subsequent live session."""
    c = _make_compressor()
    c._ineffective_compression_count = 2  # hit the anti-thrashing limit
    c._last_compression_savings_pct = 3.0

    c.on_session_end("cron-session", [])

    # After session end, the next session must start with a clean slate
    assert c._ineffective_compression_count == 0
    assert c._last_compression_savings_pct == 100.0


def test_summary_failure_cooldown_does_not_leak_across_sessions():
    """A cron session's summary failure cooldown must not block summary
    generation in a subsequent live session."""
    c = _make_compressor()
    c._summary_failure_cooldown_until = 9999999999.0

    c.on_session_end("cron-session", [])

    assert c._summary_failure_cooldown_until == 0.0


def test_compress_aborted_flag_does_not_leak_across_sessions():
    """A cron session's _last_compress_aborted flag must not make callers
    think compression is still aborted in a subsequent live session."""
    c = _make_compressor()
    c._last_compress_aborted = True

    c.on_session_end("cron-session", [])

    assert c._last_compress_aborted is False


def test_aux_model_failure_does_not_leak_across_sessions():
    """Stale aux model failure info from a cron session must not produce
    misleading error warnings in a subsequent live session."""
    c = _make_compressor()
    c._last_aux_model_failure_error = "cron-model/v1 failed"
    c._last_aux_model_failure_model = "cron-model/v1"

    c.on_session_end("cron-session", [])

    assert c._last_aux_model_failure_error is None
    assert c._last_aux_model_failure_model is None
