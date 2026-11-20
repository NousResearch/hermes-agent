"""Tests for #62452 — compression fallback timeout independence + escalating
timeout cooldown.

Two failure amplifiers when the auxiliary compression route times out:

1. Fallback candidates inherited the primary's exact ``effective_timeout``.
   A fallback tuned differently (or simply slower-but-healthy) died on the
   same clock the primary just burned, guaranteeing chain exhaustion.
   Fix: ``auxiliary.<task>.fallback_chain`` entries may declare their own
   ``timeout``; ``_call_fallback_candidate_sync/async`` resolve it via
   ``_fallback_entry_timeout``.

2. A session whose transcript structurally cannot be summarised within the
   deadline re-attempted every 60s, re-burning the full timeout on every
   subsequent turn. Fix: consecutive timeout-class failures escalate the
   cooldown 60s → 300s → 900s; any successful summary resets the streak.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.auxiliary_client import _fallback_entry_timeout, _call_fallback_candidate_sync
from agent.context_compressor import ContextCompressor


# ---------------------------------------------------------------------------
# _fallback_entry_timeout — label parsing + config resolution
# ---------------------------------------------------------------------------


def _patch_task_config(chain):
    return patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value={"fallback_chain": chain},
    )






def test_fallback_candidate_call_uses_entry_timeout():
    """The wire call for a configured-chain candidate carries the entry's own
    timeout, not the task-level one the primary just burned."""
    seen = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    fb_client = SimpleNamespace(
        base_url="https://example.invalid/v1",
        chat=SimpleNamespace(completions=_FakeCompletions()),
    )
    chain = [{"provider": "custom", "timeout": 240}]
    with _patch_task_config(chain):
        resp = _call_fallback_candidate_sync(
            fb_client, "deepseek-v4-flash", "fallback_chain[0](custom)",
            task="compression", messages=[{"role": "user", "content": "hi"}],
            temperature=None, max_tokens=None, tools=None,
            effective_timeout=30.0,  # the primary's burned budget
            effective_extra_body={}, reasoning_config=None,
        )
    assert resp is not None
    assert seen.get("timeout") == 240.0




# ---------------------------------------------------------------------------
# Escalating timeout cooldown — 60s → 300s → 900s, reset on success
# ---------------------------------------------------------------------------


def _make_compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(model="main-model", quiet_mode=True)


def _msgs():
    return [
        {"role": "user", "content": "u1 " + "x" * 200},
        {"role": "assistant", "content": "a1 " + "y" * 200},
        {"role": "user", "content": "u2 " + "z" * 200},
    ]


def _fail_with_timeout(compressor, now):
    with patch(
        "agent.context_compressor.call_llm",
        side_effect=TimeoutError("Request timed out."),
    ), patch("agent.context_compressor.time.monotonic", return_value=now):
        return compressor._generate_summary(_msgs())




def test_timeout_streak_resets_on_success():
    c = _make_compressor()
    assert _fail_with_timeout(c, 1000.0) is None
    assert _fail_with_timeout(c, 2000.0) is None
    assert c._consecutive_timeout_failures == 2

    # A successful summary clears the cooldown AND the streak.
    c._clear_compression_failure_cooldown()
    assert c._consecutive_timeout_failures == 0

    # The next timeout starts back at the 60s rung.
    assert _fail_with_timeout(c, 5000.0) is None
    assert c._summary_failure_cooldown_until == 5000.0 + 60


def test_non_timeout_transient_errors_keep_flat_cooldown():
    """Rate-limit / generic connection errors keep the flat 60s cooldown —
    escalation is scoped to timeout-class failures only."""
    c = _make_compressor()
    with patch(
        "agent.context_compressor.call_llm",
        side_effect=RuntimeError("rate limit exceeded"),
    ), patch("agent.context_compressor.time.monotonic", return_value=1000.0):
        assert c._generate_summary(_msgs()) is None
    assert c._summary_failure_cooldown_until == 1000.0 + 60

    # And it does not advance the timeout streak.
    assert getattr(c, "_consecutive_timeout_failures", 0) == 0


