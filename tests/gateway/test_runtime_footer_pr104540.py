"""Tests for PR #104540: runtime footer effort + routed_model + context_window fields.

These test the NEW behavior introduced by the PR — the model renderer now shows
the full model id (not vendor-stripped), context_pct shows "X% of Y", and the
effort field resolves reasoning effort from config.
"""

from __future__ import annotations

import pytest

from gateway.runtime_footer import (
    build_footer_line,
    format_runtime_footer,
)


# ---------------------------------------------------------------------------
# effort field — resolves reasoning effort from user_config
# ---------------------------------------------------------------------------


def test_effort_medium_when_unset():
    """No agent config → default 'medium' effort."""
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=0,
        context_length=None,
        cwd="",
        user_config={},
        fields=("effort",),
    )
    assert out == "medium"


def test_effort_none_when_disabled():
    """reasoning disabled via string override → 'none'."""
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=0,
        context_length=None,
        cwd="",
        user_config={"agent": {"reasoning_effort": "high", "reasoning_overrides": {"gpt-5.4": "none"}}},
        fields=("effort",),
    )
    assert out == "none"


def test_effort_high_from_config():
    """Explicit high effort → 'high'."""
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=0,
        context_length=None,
        cwd="",
        user_config={"agent": {"reasoning_effort": "high"}},
        fields=("effort",),
    )
    assert out == "high"


def test_effort_low_from_config():
    """Explicit low effort → 'low'."""
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=0,
        context_length=None,
        cwd="",
        user_config={"agent": {"reasoning_effort": "low"}},
        fields=("effort",),
    )
    assert out == "low"


def test_effort_override_wins():
    """Per-model reasoning_overrides (string value) take precedence over global effort."""
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=0,
        context_length=None,
        cwd="",
        user_config={
            "agent": {
                "reasoning_effort": "low",
                "reasoning_overrides": {"gpt-5.4": "high"},
            }
        },
        fields=("effort",),
    )
    assert out == "high"


def test_effort_graceful_on_bad_config():
    """Malformed config doesn't crash — returns empty string."""
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=0,
        context_length=None,
        cwd="",
        user_config={"agent": "not-a-dict"},
        fields=("effort",),
    )
    # Should not crash; returns empty or "medium"
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# model field — now shows FULL model id (not vendor-stripped)
# ---------------------------------------------------------------------------


def test_model_full_shows_full_id():
    """model field now shows the full model id, not vendor-stripped."""
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=0,
        context_length=None,
        cwd="",
        fields=("model",),
    )
    assert out == "openai/gpt-5.4"


def test_model_full_empty_when_none():
    """None model → empty string."""
    out = format_runtime_footer(
        model=None,
        context_tokens=0,
        context_length=None,
        cwd="",
        fields=("model",),
    )
    assert out == ""


def test_model_full_empty_when_empty():
    """Empty model → empty string."""
    out = format_runtime_footer(
        model="",
        context_tokens=0,
        context_length=None,
        cwd="",
        fields=("model",),
    )
    assert out == ""


# ---------------------------------------------------------------------------
# context_window field — "X% of Y" format
# ---------------------------------------------------------------------------


def test_context_window_shows_pct_and_total():
    """context_window shows 'X% of Y' with human-readable total."""
    out = format_runtime_footer(
        model="m",
        context_tokens=270_000,
        context_length=1_000_000,
        cwd="",
        fields=("context_pct",),
    )
    assert out == "27% of 1M"


def test_context_window_100k_shows_k():
    """context_window with <1M total shows k suffix."""
    out = format_runtime_footer(
        model="m",
        context_tokens=68_000,
        context_length=100_000,
        cwd="",
        fields=("context_pct",),
    )
    assert out == "68% of 100k"


def test_context_window_no_total_returns_empty():
    """No context_length → empty string (context_pct returns '' when missing)."""
    out = format_runtime_footer(
        model="m",
        context_tokens=500,
        context_length=None,
        cwd="",
        fields=("context_pct",),
    )
    assert out == ""


def test_context_window_zero_tokens():
    """Zero tokens → 0% of Y."""
    out = format_runtime_footer(
        model="m",
        context_tokens=0,
        context_length=100_000,
        cwd="",
        fields=("context_pct",),
    )
    assert out == "0% of 100k"


def test_context_window_negative_length_returns_empty():
    """Negative context_length → empty (context_pct returns '' when length is falsy)."""
    out = format_runtime_footer(
        model="m",
        context_tokens=500,
        context_length=-1,
        cwd="",
        fields=("context_pct",),
    )
    assert out == ""


# ---------------------------------------------------------------------------
# build_footer_line — routed_model passthrough
# ---------------------------------------------------------------------------


def test_build_footer_line_uses_routed_model():
    """build_footer_line uses routed_model when provided."""
    out = build_footer_line(
        user_config={"display": {"runtime_footer": {"enabled": True, "fields": ["model"]}}},
        platform_key="discord",
        model="openai/gpt-5.4",
        routed_model="openai/gpt-5.4-0314",
        context_tokens=0,
        context_length=None,
        cwd="",
    )
    assert "openai/gpt-5.4-0314" in out
    assert "gpt-5.4" in out  # full model, not short


def test_build_footer_line_falls_back_to_model():
    """build_footer_line falls back to model when routed_model is None."""
    out = build_footer_line(
        user_config={"display": {"runtime_footer": {"enabled": True, "fields": ["model"]}}},
        platform_key="discord",
        model="openai/gpt-5.4",
        routed_model=None,
        context_tokens=0,
        context_length=None,
        cwd="",
    )
    assert "openai/gpt-5.4" in out


# ---------------------------------------------------------------------------
# Combined field rendering
# ---------------------------------------------------------------------------


def test_effort_and_model_and_context_window_together():
    """Multiple new fields render together correctly."""
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=270_000,
        context_length=1_000_000,
        cwd="",
        user_config={"agent": {"reasoning_effort": "high"}},
        fields=("model", "effort", "context_pct"),
    )
    assert "openai/gpt-5.4" in out
    assert "high" in out
    assert "27% of 1M" in out
