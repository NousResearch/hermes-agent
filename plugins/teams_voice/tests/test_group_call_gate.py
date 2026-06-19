"""Tests for the group-call "speak only when addressed" gate."""

from __future__ import annotations

from plugins.teams_voice.group_call_gate import (
    GroupCallGateConfig,
    is_addressed,
    resolve_group_call_gate_config,
    should_respond_to_group_turn,
)


def test_is_addressed_word_boundary():
    phrases = ("assistant", "aria")
    assert is_addressed("Hey assistant, what's up?", phrases)
    assert is_addressed("ARIA can you help", phrases)
    # boundary: "assistantship" must NOT match "assistant"
    assert not is_addressed("the assistantship program", phrases)
    assert not is_addressed("no wake word here", phrases)


def test_is_addressed_unicode_arabic():
    # Unicode-aware boundaries: Arabic wake phrase matches.
    assert is_addressed("مرحبا مساعد كيف حالك", ("مساعد",))


def test_one_to_one_always_responds():
    cfg = GroupCallGateConfig()
    d = should_respond_to_group_turn(
        transcript="anything", is_group=False, config=cfg,
        last_addressed_at_ms=None, now_ms=1000,
    )
    assert d.respond and not d.gated


def test_group_gate_suppresses_unaddressed():
    cfg = GroupCallGateConfig(wake_phrases=("assistant",))
    d = should_respond_to_group_turn(
        transcript="just chatting with colleagues", is_group=True, config=cfg,
        last_addressed_at_ms=None, now_ms=1000,
    )
    assert not d.respond and d.gated


def test_group_follow_up_window():
    cfg = GroupCallGateConfig(wake_phrases=("assistant",), follow_up_window_ms=12_000)
    # addressed at t=0
    addressed = should_respond_to_group_turn(
        transcript="assistant, hello", is_group=True, config=cfg,
        last_addressed_at_ms=None, now_ms=0,
    )
    assert addressed.respond and addressed.addressed
    # follow-up 5s later without the name still responds
    follow = should_respond_to_group_turn(
        transcript="and what about tomorrow", is_group=True, config=cfg,
        last_addressed_at_ms=0, now_ms=5_000,
    )
    assert follow.respond and not follow.addressed
    # 20s later (past window) goes quiet
    late = should_respond_to_group_turn(
        transcript="and what about tomorrow", is_group=True, config=cfg,
        last_addressed_at_ms=0, now_ms=20_000,
    )
    assert not late.respond


def test_resolve_config_merges_camelcase():
    cfg = resolve_group_call_gate_config(
        {"requireAddress": True, "wakePhrases": ["Assistant", "Hermes"], "followUpWindowMs": 8000}
    )
    assert cfg.require_address is True
    assert cfg.wake_phrases == ("assistant", "hermes")
    assert cfg.follow_up_window_ms == 8000
