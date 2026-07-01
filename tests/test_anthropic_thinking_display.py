"""Unit tests for the Anthropic thinking.display resolver (Gap A).

reasoning_config["display"] controls whether Claude returns the reasoning text
("summarized") or suppresses it on the wire ("omitted"). Default is summarized so
Hermes always has reasoning to surface in its CLI + API-exposure paths.

Run: python -m pytest tests/test_anthropic_thinking_display.py -q
"""
from __future__ import annotations

import pytest

from agent.anthropic_adapter import _resolve_thinking_display, build_anthropic_kwargs


# ── _resolve_thinking_display ──────────────────────────────────────────

def test_default_is_summarized_when_no_config():
    assert _resolve_thinking_display(None) == "summarized"
    assert _resolve_thinking_display({}) == "summarized"
    assert _resolve_thinking_display({"effort": "high"}) == "summarized"


def test_explicit_omitted():
    assert _resolve_thinking_display({"display": "omitted"}) == "omitted"


def test_explicit_summarized():
    assert _resolve_thinking_display({"display": "summarized"}) == "summarized"


def test_case_and_whitespace_normalized():
    assert _resolve_thinking_display({"display": "  OMITTED "}) == "omitted"
    assert _resolve_thinking_display({"display": "Summarized"}) == "summarized"


def test_invalid_value_falls_back_to_summarized():
    # never produce an invalid wire value
    assert _resolve_thinking_display({"display": "verbose"}) == "summarized"
    assert _resolve_thinking_display({"display": 123}) == "summarized"
    assert _resolve_thinking_display({"display": None}) == "summarized"


# ── end-to-end through build_anthropic_kwargs (adaptive model) ─────────

def _thinking_for(reasoning_config):
    kw = build_anthropic_kwargs(
        model="claude-opus-4-7",          # adaptive-thinking model
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=4096,
        reasoning_config=reasoning_config,
    )
    return kw.get("thinking")


def test_kwargs_default_display_summarized():
    th = _thinking_for({"effort": "high"})
    assert th and th["type"] == "adaptive"
    assert th["display"] == "summarized"


def test_kwargs_display_omitted_passes_through():
    th = _thinking_for({"effort": "high", "display": "omitted"})
    assert th and th["type"] == "adaptive"
    assert th["display"] == "omitted"


def test_kwargs_display_unchanged_when_reasoning_disabled():
    # enabled:False → no thinking kwarg at all (display moot)
    th = _thinking_for({"enabled": False, "effort": "high"})
    assert th is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
