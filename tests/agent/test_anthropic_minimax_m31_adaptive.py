"""MiniMax M3.1 speaks the adaptive-thinking contract on the Anthropic-Messages wire.

The M3.1 endpoint REQUIRES adaptive thinking:

    thinking: {"type": "disabled"}          -> HTTP 400 "requires adaptive thinking"
    thinking: {"type": "enabled", "budget_tokens": N}
                                          -> accepted, but the budget is NOT a ceiling

That second case is the expensive one. Under the manual form the server emits an
unbounded reasoning block: a 4k-token request came back with 8k-12k thinking
tokens, and the response ran to the output ceiling. The model decides how much
to think; the request's ``budget_tokens`` only signals "thinking on". Measured on
one fixed prompt, the same request cost 42.8s-51.0s of wall time and ~3x the
output tokens of the adaptive form.

So M3.1 needs the same treatment as Claude 4.6+ and Kimi/Moonshot: adaptive
thinking plus ``output_config.effort``, where the effort dial is a real control.

``MiniMax-M3`` (the older generation) still accepts a disable and keeps the
legacy manual path — it must stay untouched.
"""

from __future__ import annotations

import pytest

from agent.anthropic_adapter import build_anthropic_kwargs

MESSAGES = [{"role": "user", "content": "hello"}]

# Both the GA id and the retired preview id are served by the M3.1 endpoint.
MINIMAX_M31 = [
    "MiniMax-M3.1-Flash",
    "MiniMax-M3.1-Flash-Preview",
    "minimax-m3.1-flash",
]


def _kwargs(model: str, reasoning_config: dict | None, **extra):
    return build_anthropic_kwargs(
        model=model,
        messages=MESSAGES,
        tools=None,
        max_tokens=4096,
        reasoning_config=reasoning_config,
        **extra,
    )


class TestMinimaxM31SendsAdaptiveThinking:
    @pytest.mark.parametrize("model", MINIMAX_M31)
    def test_enable_sends_adaptive_plus_effort(self, model: str) -> None:
        kwargs = _kwargs(model, {"enabled": True, "effort": "medium"})
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kwargs["output_config"] == {"effort": "medium"}

    @pytest.mark.parametrize("model", MINIMAX_M31)
    def test_enable_never_sends_manual_budget_tokens(self, model: str) -> None:
        """The manual form is accepted but its budget is not a ceiling — that is
        the whole bug: the server thinks unbounded and the turn costs 3x more."""
        kwargs = _kwargs(model, {"enabled": True, "effort": "high"})
        assert kwargs["thinking"]["type"] == "adaptive"
        assert "budget_tokens" not in kwargs["thinking"]

    @pytest.mark.parametrize("model", MINIMAX_M31)
    def test_disable_is_never_sent(self, model: str) -> None:
        """The endpoint answers a disable with HTTP 400."""
        kwargs = _kwargs(model, {"enabled": False})
        assert kwargs.get("thinking", {}).get("type") != "disabled"

    @pytest.mark.parametrize("model", MINIMAX_M31)
    def test_disable_lands_on_the_lowest_adaptive_effort(self, model: str) -> None:
        """M3.1 can neither be disabled nor left unconfigured: with no thinking
        block the server falls back to its own (unbounded) default. Its documented
        levels start at ``low``, so "off" lands there."""
        kwargs = _kwargs(model, {"enabled": False})
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kwargs["output_config"] == {"effort": "low"}

    @pytest.mark.parametrize(
        "effort,expected",
        [("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "xhigh")],
    )
    def test_every_documented_level_reaches_the_wire(self, effort: str, expected: str) -> None:
        """MiniMax documents low/medium/high/xhigh/max for M3.1."""
        kwargs = _kwargs("MiniMax-M3.1-Flash", {"enabled": True, "effort": effort})
        assert kwargs["output_config"] == {"effort": expected}


class TestOlderMinimaxIsUnchanged:
    """``MiniMax-M3`` accepts a disable and keeps the legacy manual path."""

    def test_m3_still_sends_budget_tokens(self) -> None:
        kwargs = _kwargs("MiniMax-M3", {"enabled": True, "effort": "high"})
        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["thinking"]["budget_tokens"] > 0
        assert "output_config" not in kwargs

    def test_m3_keeps_the_omission_on_disable(self) -> None:
        """Pre-M3.1: thinking is opt-in via budget_tokens, so absence IS off."""
        kwargs = _kwargs("MiniMax-M3", {"enabled": False})
        assert "thinking" not in kwargs

    def test_m2_7_is_untouched(self) -> None:
        kwargs = _kwargs("MiniMax-M2.7", {"enabled": True, "effort": "medium"})
        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["thinking"]["budget_tokens"] > 0


class TestHelperPredicate:
    def test_only_m31_is_in_the_adaptive_class(self) -> None:
        from agent.anthropic_endpoints import _model_name_is_minimax_adaptive

        assert _model_name_is_minimax_adaptive("MiniMax-M3.1-Flash") is True
        assert _model_name_is_minimax_adaptive("MiniMax-M3.1-Flash-Preview") is True
        assert _model_name_is_minimax_adaptive("minimax/minimax-m3.1-flash") is True
        assert _model_name_is_minimax_adaptive("MiniMax-M3-1-Flash") is True
        assert _model_name_is_minimax_adaptive("MiniMax-M3.1") is True
        assert _model_name_is_minimax_adaptive("MiniMax-M3") is False
        assert _model_name_is_minimax_adaptive("MiniMax-M2.7") is False
        assert _model_name_is_minimax_adaptive("claude-opus-4-8") is False

    def test_version_lookalikes_do_not_match(self) -> None:
        """A future ``minimax-m3.10`` must not inherit the M3.1 contract."""
        from agent.anthropic_endpoints import _model_name_is_minimax_adaptive

        for lookalike in ("MiniMax-M3.10", "minimax-m3-10", "MiniMax-M31", "MiniMax-M3-Flash"):
            assert _model_name_is_minimax_adaptive(lookalike) is False, lookalike

    def test_m31_is_adaptive_for_the_adapter(self) -> None:
        from agent.anthropic_adapter import _supports_adaptive_thinking

        assert _supports_adaptive_thinking("MiniMax-M3.1-Flash") is True
        assert _supports_adaptive_thinking("MiniMax-M3") is False
