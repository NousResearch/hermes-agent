"""Tests for explicit Opus-on-demand keyword routing — RED PHASE.

Verifies that [OPUS]/[HEAVY]/[CRÍTICO]/[ARCHITECTURE] markers force the
primary (expensive) model path, and that conversational continuation turns
stick to the previous tier instead of silently downgrading to cheap.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# has_opus_keyword
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message",
    [
        "[OPUS] refactor this architecture",
        "[HEAVY] critical security review",
        "[CRÍTICO] análisis de vulnerabilidades",
        "[ARCHITECTURE] design new system",
        "please [opus] check this",  # case-insensitive
    ],
)
def test_has_opus_keyword_detects_markers(message):
    # RED: function does not exist yet → ImportError
    from agent.smart_model_routing import has_opus_keyword

    assert has_opus_keyword(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "refactor this architecture",
        "opus theatre tickets",  # bare word, no brackets
        "",
        "hello",
    ],
)
def test_has_opus_keyword_ignores_non_markers(message):
    from agent.smart_model_routing import has_opus_keyword

    assert has_opus_keyword(message) is False


# ---------------------------------------------------------------------------
# choose_cheap_model_route must bail when OPUS keyword present
# ---------------------------------------------------------------------------
def _cheap_cfg():
    return {
        "enabled": True,
        "max_simple_chars": 96,
        "max_simple_words": 16,
        "cheap_model": {"provider": "copilot", "model": "gpt-5-mini"},
    }


def test_opus_keyword_blocks_cheap_routing():
    from agent.smart_model_routing import choose_cheap_model_route

    # Short message that would otherwise match cheap heuristic
    msg = "[OPUS] hola"
    assert choose_cheap_model_route(msg, _cheap_cfg()) is None


def test_heavy_keyword_blocks_cheap_routing():
    from agent.smart_model_routing import choose_cheap_model_route

    msg = "[HEAVY] quick"
    assert choose_cheap_model_route(msg, _cheap_cfg()) is None


# ---------------------------------------------------------------------------
# is_continuation_turn
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message",
    ["continúa", "sigue", "resume", "dale", "mismo tema", "continua por favor"],
)
def test_is_continuation_turn_detects_spanish_markers(message):
    # RED: function does not exist yet
    from agent.smart_model_routing import is_continuation_turn

    assert is_continuation_turn(message) is True


@pytest.mark.parametrize(
    "message",
    ["hola", "qué tal", "escribe un poema", ""],
)
def test_is_continuation_turn_ignores_non_markers(message):
    from agent.smart_model_routing import is_continuation_turn

    assert is_continuation_turn(message) is False


def test_continuation_turn_blocks_cheap_routing():
    """Continuation should inherit previous tier, not downgrade blindly."""
    from agent.smart_model_routing import choose_cheap_model_route

    assert choose_cheap_model_route("continúa", _cheap_cfg()) is None
    assert choose_cheap_model_route("sigue", _cheap_cfg()) is None
