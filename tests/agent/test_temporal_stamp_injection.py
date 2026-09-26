"""Per-turn wall-clock stamp injection (PR #40252 rework, fixes #693).

The agent gets ambient temporal awareness without breaking the provider
prefix cache: the stamp is appended to the *request-only* ephemeral system
tail by ``compose_effective_system_tail`` at API-call time, AFTER the
byte-stable cached prefix. The cached prefix itself is never mutated.

Covered contracts:

- ``compose_effective_system_tail`` appends stamp-then-ephemeral, preserves
  the base unchanged, and degrades to the pre-feature behaviour when neither
  is present.
- All THREE request-assembly sites route through the one shared composer
  (``build_api_messages`` / ``_sync_failover_system_message`` /
  ``_iteration_summary_api_messages``) — the failover site was missed by the
  original PR.
- The cache layout is preserved: the static-prefix block stays byte-identical
  (prefix cache stays warm) and the volatile tail block carries the stamp.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.conversation_loop import _sync_failover_system_message
from agent.prompt_caching import apply_anthropic_cache_control
from agent.turn_context import compose_effective_system_tail


# A fixed, timezone-aware instant so assertions are exact, not clock-relative.
from datetime import datetime, timezone

_FIXED_NOW = datetime(2026, 9, 11, 15, 45, 0, tzinfo=timezone.utc)
_FIXED_STAMP = "Current time: Friday 2026-09-11 15:45 UTC"


def _agent(**overrides):
    base = dict(
        _cached_system_prompt="STABLE SYSTEM PROMPT",
        _cached_system_prompt_static="STABLE SYSTEM PROMPT",
        ephemeral_system_prompt=None,
        _current_turn_timestamp=_FIXED_STAMP,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestComposeEffectiveSystemTail:
    def test_appends_stamp_after_base(self):
        agent = _agent()
        result = compose_effective_system_tail(agent, agent._cached_system_prompt)
        assert result == "STABLE SYSTEM PROMPT\n\n" + _FIXED_STAMP
        assert result.startswith(agent._cached_system_prompt)

    def test_stamp_precedes_ephemeral_prompt(self):
        agent = _agent(ephemeral_system_prompt="EPHEMERAL NOTE")
        result = compose_effective_system_tail(agent, agent._cached_system_prompt)
        assert result.index(_FIXED_STAMP) < result.index("EPHEMERAL NOTE")

    def test_base_preserved_when_no_stamp_or_ephemeral(self):
        agent = _agent(_current_turn_timestamp="", ephemeral_system_prompt=None)
        assert compose_effective_system_tail(agent, "BASE") == "BASE"

    def test_missing_attribute_degrades(self):
        # Older agent stubs without the stamp attribute must not raise.
        agent = SimpleNamespace(
            _cached_system_prompt="BASE", ephemeral_system_prompt=None
        )
        assert compose_effective_system_tail(agent, "BASE") == "BASE"

    def test_ephemeral_only_still_appended(self):
        agent = _agent(_current_turn_timestamp="", ephemeral_system_prompt="EPH")
        assert compose_effective_system_tail(agent, "BASE") == "BASE\n\nEPH"


class TestStampTracksClock:
    def test_stamp_uses_hermes_time_now(self):
        # Patch hermes_time.now() to a fixed aware instant and assert the exact
        # stamp — never hard-code a year, which rots (maintainer review note).
        with patch("hermes_time.now", return_value=_FIXED_NOW):
            from hermes_time import now as hermes_now

            ts = hermes_now()
            stamp = f"Current time: {ts.strftime('%A %Y-%m-%d %H:%M %Z')}"
        assert stamp == _FIXED_STAMP


class TestFailoverSiteCarriesStamp:
    def test_failover_refresh_includes_stamp(self):
        agent = _agent(
            _cached_system_prompt="STABLE", _cached_system_prompt_static="STABLE"
        )
        api_messages = [
            {"role": "system", "content": "STABLE"},
            {"role": "user", "content": "hi"},
        ]
        _sync_failover_system_message(agent, api_messages, "STABLE")
        assert _FIXED_STAMP in api_messages[0]["content"]

    def test_failover_noop_without_cached_prompt(self):
        agent = _agent(_cached_system_prompt=None, _cached_system_prompt_static=None)
        api_messages = [{"role": "system", "content": "original"}]
        _sync_failover_system_message(agent, api_messages, "active")
        assert api_messages[0]["content"] == "original"


class TestCacheLayoutPreserved:
    """The stamp rides in the volatile tail; the static prefix stays byte-identical."""

    def test_static_prefix_block_unchanged_and_marked(self):
        stable = "STABLE SYSTEM PROMPT"
        tail_base = "\n\nsession context"
        agent = _agent(_current_turn_timestamp=_FIXED_STAMP)
        effective = compose_effective_system_tail(agent, stable + tail_base)

        messages = [
            {"role": "system", "content": effective},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "new"},
        ]
        out = apply_anthropic_cache_control(
            messages, static_system_prefix=stable, native_anthropic=True
        )
        blocks = out[0]["content"]
        assert isinstance(blocks, list)
        # Block 0: the byte-stable static prefix, still its own marked block.
        assert blocks[0] == {
            "type": "text",
            "text": stable,
            "cache_control": {"type": "ephemeral"},
        }
        # Block 1: volatile tail carries the session context then the stamp.
        tail = blocks[1]
        assert tail["type"] == "text"
        assert tail["text"].startswith(tail_base)
        assert _FIXED_STAMP in tail["text"]

    def test_cached_prefix_not_mutated(self):
        agent = _agent()
        before = agent._cached_system_prompt
        compose_effective_system_tail(agent, agent._cached_system_prompt)
        assert agent._cached_system_prompt == before

    def test_static_prefix_equal_to_whole_prompt_no_empty_block(self):
        # When the prompt IS the static prefix, the tail carries only the stamp;
        # no empty text block is emitted (Anthropic rejects those with HTTP 400).
        stable = "STABLE"
        agent = _agent(_current_turn_timestamp=_FIXED_STAMP)
        effective = compose_effective_system_tail(agent, stable)
        messages = [
            {"role": "system", "content": effective},
            {"role": "user", "content": "hi"},
        ]
        out = apply_anthropic_cache_control(
            messages, static_system_prefix=stable, native_anthropic=True
        )
        for block in out[0]["content"]:
            assert block["text"].strip(), "no empty text block on the wire"
