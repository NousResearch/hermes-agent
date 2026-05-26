"""Tests for proactive skill-content decay (_decay_skill_content).

Covers the cache-stability and transcript-safety guarantees the design
depends on: decay operates on an ephemeral per-request content copy and
never touches the persistent history, and a decayed message renders
identically on every subsequent turn.
"""

from run_agent import AIAgent


def _agent(distance: int = 6) -> AIAgent:
    """Bare AIAgent with only the decay knob installed (matches the
    object.__new__ stub pattern used across the run_agent test suite)."""
    agent = object.__new__(AIAgent)
    agent.skill_decay_distance = distance
    return agent


SKILL_BLOCK = '<hermes-skill name="brainstorm">\nlots of skill body text here\n</hermes-skill>'


def test_decays_block_past_threshold():
    agent = _agent(distance=6)
    out = agent._decay_skill_content(SKILL_BLOCK, distance=10)
    assert out is not None
    assert "<hermes-skill" not in out
    assert 'skill_view("brainstorm")' in out
    assert "decayed after 10 messages" in out


def test_keeps_recent_block_within_threshold():
    agent = _agent(distance=6)
    # distance == threshold is still "recent" (only strictly greater decays).
    assert agent._decay_skill_content(SKILL_BLOCK, distance=6) is None
    assert agent._decay_skill_content(SKILL_BLOCK, distance=1) is None


def test_no_skill_block_returns_none():
    agent = _agent(distance=6)
    assert agent._decay_skill_content("just a normal tool result", distance=99) is None


def test_non_string_content_returns_none():
    agent = _agent(distance=6)
    assert agent._decay_skill_content([{"type": "text", "text": "x"}], distance=99) is None
    assert agent._decay_skill_content(None, distance=99) is None


def test_zero_distance_disables_decay():
    agent = _agent(distance=0)
    assert agent._decay_skill_content(SKILL_BLOCK, distance=1000) is None
    agent_neg = _agent(distance=-1)
    assert agent_neg._decay_skill_content(SKILL_BLOCK, distance=1000) is None


def test_preserves_trailing_steer_text():
    """Text appended after </hermes-skill> (e.g. a steer injection) must
    survive decay so mid-run steering is not silently dropped."""
    agent = _agent(distance=6)
    content = SKILL_BLOCK + "\n\n[STEER] focus on the auth bug"
    out = agent._decay_skill_content(content, distance=20)
    assert out is not None
    assert "[STEER] focus on the auth bug" in out
    assert "<hermes-skill" not in out


def test_decay_is_idempotent_and_stable_across_turns():
    """Cache-stability guarantee: once decayed, the same message decays to
    byte-identical output every turn, so the cached request prefix holds."""
    agent = _agent(distance=6)
    first = agent._decay_skill_content(SKILL_BLOCK, distance=10)
    # Re-running on the already-decayed content is a no-op (no skill block
    # left to match) — the ephemeral copy is regenerated from the pristine
    # persistent message each turn, and produces the same bytes.
    again = agent._decay_skill_content(SKILL_BLOCK, distance=10)
    assert first == again
    assert agent._decay_skill_content(first, distance=10) is None


def test_persistent_message_not_mutated_by_caller_pattern():
    """The decay helper is pure: it returns a new string and never edits
    its input, so applying it to an ephemeral api_msg copy cannot corrupt
    the persistent transcript."""
    agent = _agent(distance=6)
    original = {"role": "tool", "content": SKILL_BLOCK}
    api_copy = original.copy()
    decayed = agent._decay_skill_content(api_copy.get("content"), distance=10)
    if decayed is not None:
        api_copy["content"] = decayed
    # Persistent message still holds the full skill payload.
    assert original["content"] == SKILL_BLOCK
    assert "<hermes-skill" in original["content"]
    # Ephemeral copy is decayed.
    assert "<hermes-skill" not in api_copy["content"]
