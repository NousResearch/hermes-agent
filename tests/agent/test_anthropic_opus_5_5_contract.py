"""Claude Opus 5.5's request-shape contract on the native Anthropic wire.

Anthropic's Opus 5.5 migration guide names two request-shape breaking changes
that a harness sending the Opus 5 shape hits as a hard HTTP 400:

1. **Thinking can't be disabled.** ``thinking: {"type": "disabled"}`` (and the
   legacy ``{"type": "enabled", "budget_tokens": N}``) both answer
   ``"thinking.type.disabled" is not supported for this model.``  Thinking is
   always on; ``output_config.effort`` is the only control.
2. **Forced tool use is not supported.** ``tool_choice`` types ``any`` and
   ``tool`` answer ``tool_choice: type "tool" and "any" are not supported for
   this model.`` — on the Messages API, the Batches API *and* token counting.
   The documented replacement is ``auto`` plus strict tool use.

Both are properties of the model FAMILY, not of a single id: Fable 5.1 and
Mythos 5.1 shipped the identical restrictions a month earlier.  So the adapter
carries one family set and both guards read it, exactly the way
``_MANDATORY_THINKING_CLAUDE_SUBSTRINGS`` already handles ``claude-fable``.

Sibling contract: ``test_anthropic_thinking_disable.py`` (the disable verdict
for every other Claude family).
"""

from __future__ import annotations

import pytest

from agent.anthropic_adapter import (
    _accepts_forced_tool_choice,
    _accepts_thinking_disable,
    build_anthropic_kwargs,
)

MESSAGES = [{"role": "user", "content": "hello"}]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]

# Every id in the 5.5-generation contract: forced tool choice 400s and the
# thinking disable 400s.  Bare and Portal-namespaced spellings both, because
# the verdict is a property of the model, not of the route that serves it.
NO_FORCED_TOOLS = [
    "claude-opus-5-5",
    "anthropic/claude-opus-5-5",
    "claude-apr/claude-opus-5-5",
    "claude-opus-5-5-fast",
    "anthropic/claude-opus-5.5",  # dot spelling normalizes onto the same family
    "claude-fable-5-1",
    "claude-mythos-5-1",
]

# Models that still accept a forced tool call.  Opus 5 is the control: the
# card's whole premise is that 5.5 diverges from it.
FORCED_TOOLS_OK = [
    "anthropic/claude-opus-5",
    "claude-opus-5",
    "anthropic/claude-sonnet-5",
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-sonnet-4-5",
]


def _kwargs(model: str, **extra):
    return build_anthropic_kwargs(
        model=model,
        messages=MESSAGES,
        tools=extra.pop("tools", None),
        max_tokens=4096,
        reasoning_config=extra.pop("reasoning_config", None),
        **extra,
    )


class TestForcedToolChoiceIsDowngraded:
    """``any``/``tool`` would 400 — send ``auto`` instead, never the 400."""

    @pytest.mark.parametrize("model", NO_FORCED_TOOLS)
    def test_required_downgrades_to_auto(self, model: str) -> None:
        kwargs = _kwargs(model, tools=TOOLS, tool_choice="required")
        assert kwargs["tool_choice"] == {"type": "auto"}

    @pytest.mark.parametrize("model", NO_FORCED_TOOLS)
    def test_named_tool_downgrades_to_auto(self, model: str) -> None:
        """A forced tool NAME is the other 400 shape, and the more common one:
        it is how callers ask for structured output."""
        kwargs = _kwargs(model, tools=TOOLS, tool_choice="get_weather")
        assert kwargs["tool_choice"] == {"type": "auto"}

    @pytest.mark.parametrize("model", NO_FORCED_TOOLS)
    def test_downgrade_leaves_the_tool_schemas_untouched(self, model: str) -> None:
        """Anthropic's documented replacement is "auto + strict tool use", and
        we deliberately stop at ``auto``.

        Strict mode constrains sampling to a restricted JSON Schema subset.
        Measured against this tree's default registry
        (``model_tools.get_tool_definitions()``): every tool schema omits the
        mandatory ``additionalProperties: false``, and some use ``minimum`` /
        ``maximum`` / ``minItems > 1``, which the docs list as unsupported.
        Injecting ``strict`` would therefore replace a 400 that fires only on
        FORCED requests with one that fires on EVERY tool-bearing request.

        This test is the lock on that measurement: if someone later makes the
        schemas strict-clean and wants the flag back, they have to delete this
        test deliberately rather than regress it by accident.
        """
        kwargs = _kwargs(model, tools=TOOLS, tool_choice="get_weather")
        assert kwargs["tools"], "tools must survive the downgrade"
        for tool in kwargs["tools"]:
            assert "strict" not in tool, tool["name"]

    @pytest.mark.parametrize("model", NO_FORCED_TOOLS)
    def test_downgrade_preserves_the_tool_payload_verbatim(self, model: str) -> None:
        """The only thing the guard changes is ``tool_choice``."""
        forced = _kwargs(model, tools=TOOLS, tool_choice="get_weather")
        auto = _kwargs(model, tools=TOOLS, tool_choice="auto")
        assert forced["tools"] == auto["tools"]

    @pytest.mark.parametrize("model", NO_FORCED_TOOLS)
    def test_auto_is_untouched(self, model: str) -> None:
        kwargs = _kwargs(model, tools=TOOLS, tool_choice="auto")
        assert kwargs["tool_choice"] == {"type": "auto"}

    @pytest.mark.parametrize("model", NO_FORCED_TOOLS)
    def test_none_still_drops_the_tools(self, model: str) -> None:
        """``none`` is supported upstream; the adapter's existing translation
        (omit tools entirely) must not be caught by the downgrade."""
        kwargs = _kwargs(model, tools=TOOLS, tool_choice="none")
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    @pytest.mark.parametrize("model", FORCED_TOOLS_OK)
    def test_other_models_keep_forced_tool_choice(self, model: str) -> None:
        """The guard is scoped.  Opus 5 and earlier still force tools, and a
        blanket downgrade would be a silent capability regression there."""
        assert _kwargs(model, tools=TOOLS, tool_choice="required")[
            "tool_choice"
        ] == {"type": "any"}
        assert _kwargs(model, tools=TOOLS, tool_choice="get_weather")[
            "tool_choice"
        ] == {"type": "tool", "name": "get_weather"}

    @pytest.mark.parametrize("model", FORCED_TOOLS_OK)
    def test_other_models_gain_no_strict(self, model: str) -> None:
        kwargs = _kwargs(model, tools=TOOLS, tool_choice="get_weather")
        assert all("strict" not in t for t in kwargs["tools"])

    def test_downgrade_logs_once_per_process(self, caplog) -> None:
        """Loud enough to explain a behavior change in a log, quiet enough not
        to flood an agent loop that forces a tool every turn."""
        import agent.anthropic_adapter as aa

        aa._forced_tool_choice_downgrade_logged.clear()
        with caplog.at_level("INFO", logger=aa.logger.name):
            for _ in range(5):
                _kwargs("claude-opus-5-5", tools=TOOLS, tool_choice="required")
        hits = [r for r in caplog.records if "forced tool_choice" in r.getMessage()]
        assert len(hits) == 1, [r.getMessage() for r in hits]
        assert "claude-opus-5-5" in hits[0].getMessage()


class TestThinkingDisableIsWithheld:
    """5.5 rejects the disable the same way claude-fable always has."""

    @pytest.mark.parametrize("model", NO_FORCED_TOOLS)
    def test_disable_is_not_sent(self, model: str) -> None:
        kwargs = _kwargs(model, reasoning_config={"enabled": False})
        assert "thinking" not in kwargs

    @pytest.mark.parametrize("model", NO_FORCED_TOOLS)
    def test_enable_path_is_unchanged(self, model: str) -> None:
        """Withholding the disable must not disturb thinking-ON."""
        kwargs = _kwargs(model, reasoning_config={"enabled": True, "effort": "high"})
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kwargs["output_config"] == {"effort": "high"}

    def test_opus_5_still_accepts_the_disable(self) -> None:
        """The control: Opus 5 is disableable, 5.5 is not."""
        kwargs = _kwargs("anthropic/claude-opus-5", reasoning_config={"enabled": False})
        assert kwargs["thinking"] == {"type": "disabled"}


class TestFamilyVerdictHelpers:
    """One family set, read by both guards — not two per-id special cases."""

    @pytest.mark.parametrize("model", NO_FORCED_TOOLS)
    def test_family_rejects_both_shapes(self, model: str) -> None:
        assert _accepts_forced_tool_choice(model) is False, model
        assert _accepts_thinking_disable(model) is False, model

    @pytest.mark.parametrize("model", FORCED_TOOLS_OK)
    def test_control_models_accept_forced_tools(self, model: str) -> None:
        assert _accepts_forced_tool_choice(model) is True, model

    def test_non_claude_models_are_left_alone(self) -> None:
        """Third-party Anthropic-Messages endpoints (minimax, qwen3, GLM) have
        their own contract; Claude's restriction is not evidence about them."""
        for model in ("minimax-m2.7", "qwen3-max", "kimi-k2.5", "glm-4.6"):
            assert _accepts_forced_tool_choice(model) is True, model

    def test_unknown_future_claude_defaults_to_permissive(self) -> None:
        """Asymmetric failure, opposite direction from the thinking disable: a
        spurious downgrade silently removes a capability from every model that
        still has it, while a missing one 400s loudly and names the field.  So
        this list is opt-IN, and an unrecognized release keeps forced tools."""
        assert _accepts_forced_tool_choice("claude-opus-6") is True
        assert _accepts_forced_tool_choice("anthropic/claude-sonnet-6") is True
