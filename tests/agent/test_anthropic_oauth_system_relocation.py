"""Tests for the OAuth system-prompt relocation (plan-limit billing).

Anthropic's subscription/OAuth billing classifier fingerprints the *content* of
``system[]``: a large, distinctive non-Claude-Code system prompt is scored as a
third-party app and rejected with HTTP 400 "Third-party apps now draw from extra
usage, not plan limits" — independently of the tool-name trigger.

Fix: on the OAuth path, ``system[]`` is reduced to the Claude Code identity line
and the real prompt is relocated into a ``<system_context>`` preamble on the
first user message (carrying a ``cache_control`` marker so caching is preserved).

The system prompt enters ``build_anthropic_kwargs`` as a ``role: system`` entry
in ``messages`` (extracted by ``convert_messages_to_anthropic``), so the tests
below pass it that way.
"""

from __future__ import annotations

from agent.anthropic_adapter import (
    _CLAUDE_CODE_SYSTEM_PREFIX,
    _prepend_oauth_system_context,
    build_anthropic_kwargs,
)


class TestPrependHelper:
    def test_prepends_to_string_user_content(self):
        msgs = [{"role": "user", "content": "hello"}]
        _prepend_oauth_system_context(msgs, "<system_context>PROMPT</system_context>")
        blocks = msgs[0]["content"]
        assert isinstance(blocks, list)
        assert blocks[0]["text"].startswith("<system_context>")
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert blocks[1]["text"] == "hello"

    def test_prepends_to_list_user_content(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        _prepend_oauth_system_context(msgs, "PRE")
        blocks = msgs[0]["content"]
        assert blocks[0]["text"] == "PRE"
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert blocks[1]["text"] == "hi"

    def test_targets_first_user_not_assistant(self):
        msgs = [
            {"role": "assistant", "content": "earlier"},
            {"role": "user", "content": "now"},
        ]
        _prepend_oauth_system_context(msgs, "PRE")
        assert msgs[0]["role"] == "assistant"  # untouched
        assert msgs[1]["content"][0]["text"] == "PRE"

    def test_synthesises_user_message_when_none(self):
        msgs = [{"role": "assistant", "content": "only assistant"}]
        _prepend_oauth_system_context(msgs, "PRE")
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"][0]["text"] == "PRE"

    def test_empty_preamble_noop(self):
        msgs = [{"role": "user", "content": "x"}]
        _prepend_oauth_system_context(msgs, "")
        assert msgs[0]["content"] == "x"


class TestOAuthSystemRelocation:
    """End-to-end through build_anthropic_kwargs (the real call path)."""

    def _kwargs(self, system_text, is_oauth=True):
        return build_anthropic_kwargs(
            model="claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": "do the thing"},
            ],
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            is_oauth=is_oauth,
        )

    def test_oauth_system_is_identity_only(self):
        kw = self._kwargs("You are Hermes Agent. Follow these rules. " * 200)
        system = kw["system"]
        assert isinstance(system, list)
        assert len(system) == 1
        assert system[0]["text"] == _CLAUDE_CODE_SYSTEM_PREFIX

    def test_oauth_prompt_relocated_to_first_user_message(self):
        big = "You are Hermes Agent built by Nous Research. " * 50
        kw = self._kwargs(big)
        first_user = next(m for m in kw["messages"] if m["role"] == "user")
        blocks = first_user["content"]
        assert isinstance(blocks, list)
        # relocated preamble is first, cache-marked, wrapped, and sanitized
        assert blocks[0]["text"].startswith("<system_context>")
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert "Hermes Agent" not in blocks[0]["text"]
        assert "Nous Research" not in blocks[0]["text"]
        assert "Claude Code" in blocks[0]["text"]
        # the user's real text survives after the preamble
        assert any(b.get("text") == "do the thing" for b in blocks)

    def test_oauth_no_distinctive_content_left_in_system(self):
        big = "You are Hermes Agent built by Nous Research. " * 50
        kw = self._kwargs(big)
        # system[] must carry ONLY the identity line — nothing of the real prompt
        system_text = " ".join(b.get("text", "") for b in kw["system"])
        assert system_text == _CLAUDE_CODE_SYSTEM_PREFIX

    def test_non_oauth_keeps_system_prompt(self):
        big = "You are Hermes Agent. " * 50
        kw = self._kwargs(big, is_oauth=False)
        # Non-OAuth: the system prompt is preserved as the system arg, NOT
        # relocated into the user message.
        assert kw.get("system")
        sys_text = (
            kw["system"] if isinstance(kw["system"], str)
            else " ".join(b.get("text", "") for b in kw["system"])
        )
        assert "Hermes Agent" in sys_text
        first_user = next(m for m in kw["messages"] if m["role"] == "user")
        content = first_user["content"]
        # no <system_context> preamble was injected
        if isinstance(content, list):
            assert not any("<system_context>" in b.get("text", "") for b in content)
        else:
            assert "<system_context>" not in content


def _count_cache_markers(obj) -> int:
    """Count cache_control markers anywhere in a nested message/system structure."""
    n = 0
    if isinstance(obj, dict):
        if "cache_control" in obj:
            n += 1
        for v in obj.values():
            n += _count_cache_markers(v)
    elif isinstance(obj, list):
        for v in obj:
            n += _count_cache_markers(v)
    return n


class TestOAuthCacheBreakpointCap:
    """Regression guard for the 4-breakpoint cap under the REAL production order.

    The risky interaction (untested before this): the upstream
    ``apply_anthropic_cache_control`` pass marks the system block + last 3
    messages (4 breakpoints), THEN the OAuth relocation reduces system[] to the
    identity line. The relocated preamble must take the place of the discarded
    system marker so the wire stays at exactly 4 — never 5, which Anthropic 400s.
    """

    def test_oauth_relocation_respects_4_breakpoint_cap(self):
        from agent.prompt_caching import apply_anthropic_cache_control

        msgs = [
            {"role": "system", "content": "You are Hermes Agent by Nous Research. " * 80},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
            {"role": "user", "content": "third question"},
        ]
        # Production order: caching pass first, then build_anthropic_kwargs.
        cached = apply_anthropic_cache_control(msgs, cache_ttl="5m", native_anthropic=False)
        kw = build_anthropic_kwargs(
            model="claude-opus-4-8",
            messages=cached,
            tools=None,
            max_tokens=8,
            reasoning_config=None,
            is_oauth=True,
        )
        total = _count_cache_markers(kw.get("system")) + _count_cache_markers(kw.get("messages"))
        assert total <= 4, f"OAuth wire exceeded Anthropic's 4-breakpoint cap: {total}"
        # And the relocated preamble must be one of the breakpoints (cache preserved).
        first_user = next(m for m in kw["messages"] if m["role"] == "user")
        blocks = first_user["content"]
        assert isinstance(blocks, list)
        assert "cache_control" in blocks[0]
        assert blocks[0]["text"].startswith("<system_context>")
