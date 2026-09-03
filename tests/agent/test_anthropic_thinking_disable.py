"""'Thinking off' on the native Anthropic Messages wire.

Adaptive Claude models (4.6+) think by DEFAULT.  Omitting the ``thinking``
parameter is therefore NOT a disable — it leaves the upstream default in
place and the model keeps thinking, which is exactly what a user who turned
thinking off is trying to stop paying for.  The disable has to be sent:

    thinking: {"type": "disabled"}

Reasoning-mandatory families (claude-fable) reject that with an HTTP 400
("Thinking is mandatory for this model"), so they keep the omission — a
silently-ignored disable is a much better failure than a dead turn.

Legacy manual-thinking Claude (<= 4.5) needs no disable at all: thinking is
opt-in there via ``budget_tokens``, so sending nothing already means off.

Sibling contract on the chat_completions wire: hermes-agent#90412.
"""

from __future__ import annotations

import pytest

from agent.anthropic_adapter import build_anthropic_kwargs

MESSAGES = [{"role": "user", "content": "hello"}]

# Portal ids and their bare Anthropic equivalents: the disable verdict is a
# property of the model family, not of which route serves it.
ADAPTIVE_DISABLEABLE = [
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-4.8",
    "anthropic/claude-opus-4.7",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4.6",
    "claude-opus-4-6",
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


class TestThinkingOffIsSentExplicitly:
    @pytest.mark.parametrize("model", ADAPTIVE_DISABLEABLE)
    def test_adaptive_models_receive_an_explicit_disable(self, model: str) -> None:
        """The whole point: omission would leave thinking ON for these."""
        kwargs = _kwargs(model, {"enabled": False})
        assert kwargs["thinking"] == {"type": "disabled"}

    @pytest.mark.parametrize("model", ADAPTIVE_DISABLEABLE)
    def test_disable_carries_no_effort_dial(self, model: str) -> None:
        """``output_config.effort`` describes how hard to think — meaningless
        alongside a disable, and it is what the enable path sets."""
        kwargs = _kwargs(model, {"enabled": False, "effort": "high"})
        assert kwargs["thinking"] == {"type": "disabled"}
        assert "output_config" not in kwargs

    def test_mandatory_thinking_models_keep_the_omission(self) -> None:
        """claude-fable answers a disable with HTTP 400, so don't send one."""
        kwargs = _kwargs("anthropic/claude-fable-5", {"enabled": False})
        assert "thinking" not in kwargs

    def test_legacy_manual_thinking_models_keep_the_omission(self) -> None:
        """Pre-4.6 thinking is opt-in via budget_tokens: absence IS off."""
        kwargs = _kwargs("claude-sonnet-4-5", {"enabled": False})
        assert "thinking" not in kwargs

    def test_haiku_keeps_the_omission(self) -> None:
        """Haiku is legacy-manual, so absence already means off."""
        kwargs = _kwargs("anthropic/claude-haiku-4.5", {"enabled": False})
        assert "thinking" not in kwargs

    def test_kimi_keeps_its_documented_omission(self) -> None:
        """Kimi speaks the adaptive contract but is out of scope here (#13848)."""
        kwargs = _kwargs(
            "kimi-k2.5", {"enabled": False}, base_url="https://api.kimi.com/coding"
        )
        assert "thinking" not in kwargs


class TestEnablePathIsUnchanged:
    """The disable branch must not disturb the thinking-ON contract."""

    def test_adaptive_enable_still_sends_adaptive_plus_effort(self) -> None:
        kwargs = _kwargs("anthropic/claude-opus-5", {"enabled": True, "effort": "high"})
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kwargs["output_config"] == {"effort": "high"}

    def test_mandatory_model_still_thinks_when_asked_to(self) -> None:
        kwargs = _kwargs("anthropic/claude-fable-5", {"enabled": True, "effort": "max"})
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kwargs["output_config"] == {"effort": "max"}

    def test_legacy_enable_still_sends_budget_tokens(self) -> None:
        kwargs = _kwargs("claude-sonnet-4-5", {"enabled": True, "effort": "high"})
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 16000}

    def test_haiku_still_never_gets_thinking_on_the_enable_path(self) -> None:
        kwargs = _kwargs("anthropic/claude-haiku-4.5", {"enabled": True, "effort": "high"})
        assert "thinking" not in kwargs

    def test_no_reasoning_config_sends_no_thinking_field(self) -> None:
        assert "thinking" not in _kwargs("anthropic/claude-opus-5", None)


class TestDisableVerdictHelper:
    """``_accepts_thinking_disable`` is the single source of the verdict."""

    def test_verdict_matches_the_mandatory_flag_the_catalog_publishes(self) -> None:
        from agent.anthropic_adapter import _accepts_thinking_disable

        # Portal catalog: reasoning.mandatory is true for fable, false for these.
        assert _accepts_thinking_disable("anthropic/claude-opus-5") is True
        assert _accepts_thinking_disable("anthropic/claude-sonnet-5") is True
        assert _accepts_thinking_disable("anthropic/claude-fable-5") is False

    def test_non_claude_models_are_left_alone(self) -> None:
        from agent.anthropic_adapter import _accepts_thinking_disable

        for model in ("minimax-m2.7", "qwen3-max", "kimi-k2.5"):
            assert _accepts_thinking_disable(model) is False, model

    def test_unknown_claude_releases_default_to_disableable(self) -> None:
        """Mirrors _supports_adaptive_thinking: new Claude gets the modern
        contract without a code change."""
        from agent.anthropic_adapter import _accepts_thinking_disable

        assert _accepts_thinking_disable("anthropic/claude-opus-6") is True
