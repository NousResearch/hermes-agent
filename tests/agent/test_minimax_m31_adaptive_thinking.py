"""MiniMax M3.1 adaptive thinking (2026-10-28).

MiniMax's M3.1 family documents ``output_config.effort`` (low/medium/high/xhigh/max)
as the ONLY working thinking-depth control on the Anthropic-compatible endpoint, and
rejects ``thinking.type=disabled`` with HTTP 400. The legacy manual-thinking path
Hermes uses for every non-Claude model sends ``thinking.type=enabled`` +
``budget_tokens`` — measured against the live API, budget_tokens does NOT constrain
M3.1's thinking volume (1000/4000/32000 -> 971/541/900 thinking tokens, no monotonic
relation), while output_config.effort does (low=105 / medium=347 / max=1955).
(A second measurement run counted thinking-block characters instead of provider
reported thinking tokens and reproduced the same shape: effort low < medium < max
monotonically, while budget_tokens stayed uncorrelated.)

So M3.1 must be routed down the adaptive path, exactly like Claude 4.6+ and Kimi.
M3 and the M2.x family keep the manual path: their behaviour is unchanged and
already verified.
"""

import pytest


class TestMinimaxM31AdaptiveClassification:
    """M3.1 family classifies as adaptive; M3 / M2.x keep the manual path."""

    def test_m31_family_is_adaptive(self):
        from agent.anthropic_adapter import _supports_adaptive_thinking

        for m in (
            "MiniMax-M3.1-Flash",
            "minimax-m3.1-flash",
            "MiniMax-M3.1-Flash-Preview",
            "minimax/MiniMax-M3.1-Flash",
        ):
            assert _supports_adaptive_thinking(m) is True, m

    def test_m31_supports_xhigh_effort(self):
        """xhigh is one of M3.1's documented effort levels — it must not be downgraded."""
        from agent.anthropic_adapter import _supports_xhigh_effort

        assert _supports_xhigh_effort("MiniMax-M3.1-Flash") is True

    def test_m3_and_m2x_stay_on_manual_path(self):
        """Only the M3.1 family moves. M3 / M2.x behaviour must not change."""
        from agent.anthropic_adapter import _supports_adaptive_thinking

        for m in ("MiniMax-M3", "minimax-m3", "MiniMax-M2.7", "MiniMax-M2.5", "minimax-m2"):
            assert _supports_adaptive_thinking(m) is False, m

    def test_m31_does_not_forbid_sampling_params(self):
        """M3.1 accepts temperature/top_p alongside adaptive thinking (measured)."""
        from agent.anthropic_adapter import _forbids_sampling_params

        assert _forbids_sampling_params("MiniMax-M3.1-Flash") is False


class TestMinimaxM31ThinkingKwargs:
    """The wire shape must be adaptive + output_config.effort, never budget_tokens."""

    @pytest.mark.parametrize(
        "hermes_effort,expected",
        [
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "xhigh"),
            ("max", "max"),
            ("ultra", "max"),
            ("minimal", "low"),
        ],
    )
    def test_effort_maps_to_output_config(self, hermes_effort, expected):
        from agent.anthropic_adapter import _thinking_kwargs

        kwargs = _thinking_kwargs(
            {"enabled": True, "effort": hermes_effort}, "MiniMax-M3.1-Flash", 4096
        )
        assert kwargs.get("thinking") == {"type": "adaptive", "display": "summarized"}
        assert kwargs.get("output_config") == {"effort": expected}
        # The manual path's budget_tokens shape must never reach M3.1.
        assert "budget_tokens" not in kwargs.get("thinking", {})

    def test_effort_none_is_not_sent(self):
        """M3.1 400s on effort=none / thinking disabled — the disable branch must omit
        the thinking block entirely rather than send an explicit disable."""
        from agent.anthropic_adapter import _thinking_kwargs

        kwargs = _thinking_kwargs(
            {"enabled": False, "effort": "none"}, "MiniMax-M3.1-Flash", 4096
        )
        assert kwargs.get("thinking", {}).get("type") != "disabled"
        assert "output_config" not in kwargs

    def test_m3_keeps_manual_budget_tokens_shape(self):
        """Regression guard: M3 must keep the legacy manual-thinking shape."""
        from agent.anthropic_adapter import _thinking_kwargs

        kwargs = _thinking_kwargs({"enabled": True, "effort": "medium"}, "MiniMax-M3", 4096)
        assert kwargs.get("thinking") == {"type": "enabled", "budget_tokens": 8000}
        assert "output_config" not in kwargs
