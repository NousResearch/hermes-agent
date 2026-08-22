"""Tests for the Anthropic OAuth system-prompt budget (GH-65564).

Anthropic's subscription/OAuth billing classifier inspects the system prompt.
A large block of Hermes-specific instructions (skill_manage, session_search,
Computer Use guidance, mid-turn steering, ...) is fingerprinted as raw API
usage, so the request is billed against the pay-per-token "extra usage" pool
instead of the subscription's included quota.  Once that pool is empty the API
answers:

    HTTP 400 "You're out of extra usage. Add more at claude.ai/settings/usage"
    HTTP 429 "monthly spend limit"

...even though the Claude Pro/Max subscription still has plenty of headroom.
This is the same class of problem as the ``mcp_`` -> ``mcp__`` tool-name
normalization (GH-25255, see test_anthropic_mcp_prefix_strip.py): the payload
*shape*, not the actual usage, decides the billing lane.

The mitigation trims the Hermes-specific portion of the system prompt on the
OAuth wire to a budget configured in ``config.yaml``:

    providers:
      anthropic:
        oauth_system_budget_chars: 3000   # 0 disables

Invariants covered here:

  1. The Claude Code identity block (index 0) is never trimmed.
  2. The Hermes portion is capped at the configured budget.
  3. Budget 0 disables the behavior entirely (upstream semantics).
  4. Short prompts are passed through untouched.
  5. No empty text block is ever left on the wire — Anthropic 400s on those,
     and an empty block carrying ``cache_control`` is *always* rejected
     ("cache_control cannot be set for empty text blocks").
  6. Prompt caching survives: a dropped block's ``cache_control`` marker is
     migrated to the last surviving text block.
  7. API-key (non-OAuth) requests are never trimmed — they bill per token.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


IDENTITY_MARKER = "Claude Code"
DEFAULT_BUDGET = 3000


def _config(budget) -> dict:
    """Build a config.yaml-shaped dict carrying the budget setting."""
    if budget is None:
        return {}
    return {"providers": {"anthropic": {"oauth_system_budget_chars": budget}}}


def _build(system_text: str, budget=None, *, is_oauth: bool = True):
    """Build Anthropic kwargs with a patched config.yaml value."""
    from agent.anthropic_adapter import build_anthropic_kwargs

    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value=_config(budget),
    ):
        return build_anthropic_kwargs(
            model="claude-sonnet-4-5",
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": "hello"},
            ],
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            is_oauth=is_oauth,
        )


def _hermes_len(system) -> int:
    """Length of the system text excluding the Claude Code identity block."""
    if isinstance(system, str):
        return len(system)
    texts = [
        b.get("text", "")
        for b in system
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    return sum(len(t) for t in texts[1:])


class TestOAuthSystemBudgetResolution:
    """The budget comes from config.yaml, not from an environment variable."""

    def test_reads_budget_from_config_yaml(self):
        from agent.anthropic_adapter import _resolve_oauth_system_budget

        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value=_config(1234),
        ):
            assert _resolve_oauth_system_budget() == 1234

    def test_defaults_when_unset(self):
        from agent.anthropic_adapter import _resolve_oauth_system_budget

        with patch("hermes_cli.config.load_config_readonly", return_value={}):
            assert _resolve_oauth_system_budget() == DEFAULT_BUDGET

    def test_zero_disables(self):
        from agent.anthropic_adapter import _resolve_oauth_system_budget

        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value=_config(0),
        ):
            assert _resolve_oauth_system_budget() == 0

    def test_malformed_value_falls_back_to_default(self):
        """A typo in config.yaml must not break request building."""
        from agent.anthropic_adapter import _resolve_oauth_system_budget

        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value=_config("not-a-number"),
        ):
            assert _resolve_oauth_system_budget() == DEFAULT_BUDGET

    def test_unreadable_config_falls_back_to_default(self):
        from agent.anthropic_adapter import _resolve_oauth_system_budget

        with patch(
            "hermes_cli.config.load_config_readonly",
            side_effect=OSError("config.yaml unreadable"),
        ):
            assert _resolve_oauth_system_budget() == DEFAULT_BUDGET


class TestOAuthSystemBudgetTrim:
    """The Hermes portion of the system prompt is capped on the OAuth wire."""

    def test_long_system_prompt_is_trimmed_to_budget(self):
        kwargs = _build("Hermes instructions. " * 800, budget=DEFAULT_BUDGET)

        assert isinstance(kwargs["system"], list)
        assert _hermes_len(kwargs["system"]) <= DEFAULT_BUDGET

    def test_claude_code_identity_block_is_preserved(self):
        """Block 0 carries the Claude Code identity and must survive intact."""
        kwargs = _build("Hermes instructions. " * 800, budget=DEFAULT_BUDGET)
        system = kwargs["system"]

        assert system, "system prompt unexpectedly empty"
        assert IDENTITY_MARKER in system[0].get("text", "")

    def test_budget_zero_disables_trimming(self):
        """Opt-out restores the previous (upstream) behavior."""
        text = "Hermes instructions. " * 800
        trimmed = _build(text, budget=DEFAULT_BUDGET)
        untouched = _build(text, budget=0)

        assert _hermes_len(untouched["system"]) > _hermes_len(trimmed["system"])
        # Sanitization may rewrite product names, so allow a small delta.
        assert _hermes_len(untouched["system"]) >= len(text) - 1

    def test_short_system_prompt_is_untouched(self):
        """Prompts below the budget must not be modified at all."""
        text = "Be concise."
        kwargs = _build(text, budget=DEFAULT_BUDGET)

        assert _hermes_len(kwargs["system"]) == len(text)

    def test_custom_budget_is_respected(self):
        kwargs = _build("Hermes instructions. " * 800, budget=500)

        assert _hermes_len(kwargs["system"]) <= 500

    def test_non_oauth_requests_are_never_trimmed(self):
        """API-key requests bill per token and must keep the full prompt."""
        text = "Hermes instructions. " * 800
        kwargs = _build(text, budget=DEFAULT_BUDGET, is_oauth=False)

        assert _hermes_len(kwargs["system"]) >= len(text) - 1


class TestOAuthSystemBudgetWireSafety:
    """Trimming must never produce a payload Anthropic rejects."""

    def test_no_empty_text_block_on_the_wire(self):
        """Anthropic 400s on empty text blocks — none may survive trimming."""
        kwargs = _build("Hermes instructions. " * 800, budget=DEFAULT_BUDGET)

        empty = [
            b
            for b in kwargs["system"]
            if isinstance(b, dict) and b.get("type") == "text" and not b.get("text")
        ]
        assert empty == [], "empty text block would trigger HTTP 400"

    def test_no_empty_block_retains_cache_control(self):
        """``cache_control`` on an empty text block is always rejected."""
        kwargs = _build("Hermes instructions. " * 800, budget=DEFAULT_BUDGET)

        offenders = [
            b
            for b in kwargs["system"]
            if isinstance(b, dict)
            and b.get("type") == "text"
            and not b.get("text")
            and b.get("cache_control")
        ]
        assert offenders == [], (
            "empty text block with cache_control -> "
            "'cache_control cannot be set for empty text blocks'"
        )

    def test_trim_prefers_a_line_boundary(self):
        """Instructions are cut at a newline, not mid-sentence, when possible."""
        # Many short lines guarantee a newline exists just below the budget.
        text = "\n".join(f"Instruction line {i}." for i in range(600))
        kwargs = _build(text, budget=DEFAULT_BUDGET)

        blocks = [
            b
            for b in kwargs["system"]
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
        ]
        trimmed_text = blocks[-1]["text"]
        assert trimmed_text.endswith("."), "cut landed mid-sentence"
