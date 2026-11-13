"""Tests for ``compression.max_summary_tokens`` — an explicit summary-size target.

Motivation: the summary budget is derived from the compressed content
(``min(content × 0.20, min(context_length × 5%, 10000))``), which is fine for a hosted
API decoding at 400+ tok/s but not for a local route at 30-60 tok/s: compaction time is
``output tokens ÷ decode speed``, so a 30K-token summary costs ~15 minutes and always
exceeds the compression deadline. The knob shrinks the model-facing target, and because
the budget alone is advisory (models overshoot it ~3x) it also makes the prompt carry an
explicit hard ceiling.

Two invariants matter here:

1. Default behaviour is unchanged — no override, same derived ceiling, same prompt.
2. The override never becomes a wire ``max_tokens``: a length-stopped summary is treated
   as PARTIAL by ``_response_finish_reason``/``_TRUNCATED_SUMMARY_MARKER``, which discards
   the compaction. Only the prompt-side target moves.
"""

from agent.context_compressor import (
    _LEAN_SESSION_LOG_BUDGET_TOKENS,
    _SUMMARY_TOKENS_CEILING,
    ContextCompressor,
)
from hermes_cli.config import DEFAULT_CONFIG


def _compressor(context_length=1_000_000, **kwargs):
    return ContextCompressor(model="deepseek-flash", config_context_length=context_length,
                             quiet_mode=True, **kwargs)


def _prompt(compressor, budget):
    compressor._previous_summary = None
    return compressor._build_summary_prompt("transcript sample " * 100, budget, None, "", True)


def _target_line(prompt):
    return next(line for line in prompt.splitlines() if line.startswith("Target ~"))


# ---------------------------------------------------------------------------
# Default path: no config key → nothing changes
# ---------------------------------------------------------------------------

def test_default_target_is_derived_from_the_window():
    c = _compressor(context_length=1_000_000)
    assert c.summary_tokens_override is None
    assert c.max_summary_tokens == _SUMMARY_TOKENS_CEILING


def test_small_window_derives_five_percent():
    c = _compressor(context_length=100_000)
    assert c.max_summary_tokens == int(100_000 * 0.05)


def test_default_prompt_has_no_hard_ceiling():
    c = _compressor()
    budget = c._compute_summary_budget([{"role": "tool", "content": "x" * 40_000}])
    prompt = _prompt(c, budget)
    assert "HARD CEILING" not in prompt
    assert f"Target ~{budget + _LEAN_SESSION_LOG_BUDGET_TOKENS} tokens." in _target_line(prompt)


# ---------------------------------------------------------------------------
# Override: the explicit target wins, and the prompt actually says so
# ---------------------------------------------------------------------------

def test_override_wins_over_the_window_ceiling():
    c = _compressor()
    c.summary_tokens_override = 6_000
    assert c.max_summary_tokens == 6_000


def test_override_survives_a_cached_ceiling_recompute():
    """update_model() refills the cached value; the override must still win."""
    c = _compressor()
    c.max_summary_tokens = 9_999          # simulate a cached derived value
    c.summary_tokens_override = 6_000
    assert c.max_summary_tokens == 6_000


def test_budget_is_clamped_to_the_override():
    c = _compressor()
    c.summary_tokens_override = 6_000
    # 64K tokens of content would derive 12.8K; the override clamps it to 6K.
    big = c._compute_summary_budget([{"role": "tool", "content": "x" * 160_000}])
    assert big == 6_000
    # Small content keeps its own smaller budget — the override is a ceiling, not a floor.
    small = c._compute_summary_budget([{"role": "user", "content": "y" * 4_000}])
    assert small < 6_000


def test_override_prompt_states_the_target_and_a_hard_ceiling():
    c = _compressor()
    c.summary_tokens_override = 6_000
    budget = c._compute_summary_budget([{"role": "tool", "content": "x" * 160_000}])
    prompt = _prompt(c, budget)
    line = _target_line(prompt)
    # The session-log section's 4K allowance must not be added on top of an explicit target.
    assert line.startswith("Target ~6000 tokens. HARD CEILING")
    assert f"{6000 + _LEAN_SESSION_LOG_BUDGET_TOKENS}" not in line
    # Overshoot is the failure mode being fixed, so the prompt also caps the two sections
    # that grow without bound.
    assert "Completed Actions: at most 20 entries" in prompt
    assert "Relevant Files: only files actually read or modified" in prompt


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

def test_default_config_exposes_the_key():
    assert "max_summary_tokens" in DEFAULT_CONFIG["compression"]
    assert DEFAULT_CONFIG["compression"]["max_summary_tokens"] is None
