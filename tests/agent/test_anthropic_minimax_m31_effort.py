"""MiniMax M3.1 speaks the adaptive-thinking contract on the Anthropic-Messages wire.

MiniMax's Anthropic-compatible endpoint (``https://api.minimax.io/anthropic``) documents
``thinking`` and ``output_config.effort``; the M3.1 line implements the adaptive contract:

    thinking: {"type": "disabled"}          -> HTTP 400 "requires adaptive thinking"
    thinking: {"type": "enabled", "budget_tokens": N}
                                          -> accepted, but the budget is NOT a ceiling
    thinking: {"type": "adaptive"}         -> accepted; depth comes from output_config.effort

That second case is the expensive one: MiniMax parses ``budget_tokens`` and never enforces it,
so the server decides how much to think. Measured live on one fixed prompt, budget 300 / 8000 /
32000 all produced the same thinking volume, and the request ran at the server default (``max``)
whatever level the user picked. The effort dial only reaches the wire through
``output_config.effort`` (low/medium/high/xhigh/max), which does move thinking volume.

``MiniMax-M3`` (the older generation) is deliberately NOT covered: measured live on the same
endpoint, ``output_config.effort`` does not change its thinking volume either (n=6 per level,
one band, no gradient), so it keeps the legacy manual path.
"""

from __future__ import annotations

import pytest

from agent.anthropic_adapter import build_anthropic_kwargs

MESSAGES = [{"role": "user", "content": "hello"}]
MINIMAX_BASE_URL = "https://api.minimax.io/anthropic"

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
        base_url=MINIMAX_BASE_URL,
        **extra,
    )


class TestMinimaxM31SendsAdaptiveThinking:
    @pytest.mark.parametrize("model", MINIMAX_M31)
    def test_enable_sends_adaptive_plus_effort(self, model: str) -> None:
        kwargs = _kwargs(model, {"enabled": True, "effort": "medium"})
        assert kwargs["thinking"]["type"] == "adaptive"
        assert kwargs["output_config"] == {"effort": "medium"}

    @pytest.mark.parametrize("model", MINIMAX_M31)
    def test_enable_never_sends_manual_budget_tokens(self, model: str) -> None:
        """The manual form is accepted but its budget is not a ceiling — that is
        the whole bug: the server thinks unbounded and ignores the requested level."""
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
        assert kwargs["thinking"]["type"] == "adaptive"
        assert kwargs["output_config"] == {"effort": "low"}

    @pytest.mark.parametrize(
        "effort,expected",
        [("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "xhigh"), ("max", "max")],
    )
    def test_every_documented_level_reaches_the_wire(self, effort: str, expected: str) -> None:
        """MiniMax documents low/medium/high/xhigh/max for M3.1."""
        kwargs = _kwargs("MiniMax-M3.1-Flash", {"enabled": True, "effort": effort})
        assert kwargs["output_config"] == {"effort": expected}

    def test_hermes_internal_levels_map_onto_minimaxes_ladder(self) -> None:
        """``ultra`` is a Hermes-internal step with no MiniMax counterpart (its ladder
        stops at ``max``); ``minimal`` is the legacy alias for ``low``."""
        assert _kwargs("MiniMax-M3.1-Flash", {"enabled": True, "effort": "ultra"})[
            "output_config"
        ] == {"effort": "max"}
        assert _kwargs("MiniMax-M3.1-Flash", {"enabled": True, "effort": "minimal"})[
            "output_config"
        ] == {"effort": "low"}

    def test_output_cap_leaves_room_for_the_reasoning_pass_and_the_answer(self) -> None:
        """Thinking tokens count against ``max_tokens`` on MiniMax too. Measured live:
        a 4096 cap on an ``effort: high`` request answered with a thinking block and
        NO text (stop_reason=max_tokens), while 8192 and up returned the answer."""
        kwargs = _kwargs("MiniMax-M3.1-Flash", {"enabled": True, "effort": "high"})
        assert kwargs["max_tokens"] >= 32768


class TestOlderMinimaxIsUnchanged:
    """``MiniMax-M3`` accepts a disable, and its effort knob does not move the needle
    (measured live), so it keeps the legacy manual path."""

    def test_m3_still_sends_budget_tokens(self) -> None:
        kwargs = _kwargs("MiniMax-M3", {"enabled": True, "effort": "high"})
        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["thinking"]["budget_tokens"] > 0
        assert "output_config" not in kwargs

    def test_m27_still_sends_budget_tokens(self) -> None:
        kwargs = _kwargs("MiniMax-M2.7", {"enabled": True, "effort": "high"})
        assert kwargs["thinking"]["type"] == "enabled"
        assert "output_config" not in kwargs


class TestOnlyTheM31LineMatches:
    """The match is a release-line prefix, so lookalikes never inherit it."""

    @pytest.mark.parametrize(
        "model",
        ["minimax-m3", "minimax-m3.10", "minimax-m31", "MiniMax-M3", "MiniMax-M2.7"],
    )
    def test_lookalikes_keep_the_manual_path(self, model: str) -> None:
        kwargs = _kwargs(model, {"enabled": True, "effort": "high"})
        assert kwargs["thinking"]["type"] == "enabled"
        assert "output_config" not in kwargs


class TestOtherEndpointsAreUnaffected:
    """The knob belongs to MiniMax's own endpoint. A relay serving the same model slug
    speaks a different contract and must keep the payload it had."""

    def test_the_same_model_on_another_host_keeps_the_manual_path(self) -> None:
        kwargs = build_anthropic_kwargs(
            model="MiniMax-M3.1-Flash",
            messages=MESSAGES,
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "high"},
            base_url="https://openrouter.ai/api/v1",
        )
        assert kwargs["thinking"]["type"] == "enabled"
        assert "output_config" not in kwargs

    def test_a_request_without_a_base_url_keeps_the_manual_path(self) -> None:
        kwargs = build_anthropic_kwargs(
            model="MiniMax-M3.1-Flash",
            messages=MESSAGES,
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert "output_config" not in kwargs


class TestPickerHonesty:
    """The route declares its own ladder, so the Desktop picker never claims a level the
    wire does not have (``ultra`` sends ``max``; ``minimal`` sends ``low``)."""

    def test_the_route_declares_minimaxes_ladder(self) -> None:
        from agent.reasoning_effort import route_supported_efforts
        supported = route_supported_efforts("minimax", "MiniMax-M3.1-Flash")
        assert supported == ("low", "medium", "high", "xhigh", "max")

    def test_a_relay_serving_the_same_slug_keeps_the_wide_ladder(self) -> None:
        from agent.reasoning_effort import route_supported_efforts
        assert "ultra" not in route_supported_efforts("openrouter", "minimax/minimax-m3.1")

    def test_the_label_reports_the_clamp_instead_of_pretending(self) -> None:
        from agent.reasoning_effort import effort_display_label
        assert effort_display_label("ultra", "minimax", "MiniMax-M3.1-Flash") == (
            "ultra (sends max on this route)"
        )
        assert effort_display_label("high", "minimax", "MiniMax-M3.1-Flash") == "high"
