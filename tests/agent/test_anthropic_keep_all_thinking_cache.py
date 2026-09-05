"""Prior-turn thinking blocks and the prompt cache.

Keep-all Claude models (Opus >= 4.5, Sonnet >= 4.6, Fable, Mythos) hold every earlier turn's
thinking in context and in the cache when it is passed back unchanged. Deleting those blocks
client-side changes the cached prefix from the first deleted block onward on every request,
which turned ~80% of uncached input in a 1,393-agent run into cache writes. Last-turn-only
models strip older blocks server-side, so the client strip there is harmless and stays.
"""
import copy

from agent.anthropic_endpoints import _model_keeps_all_thinking
from agent.anthropic_message_convert import convert_messages_to_anthropic

PORTAL = "https://inference-api.nousresearch.com/v1"


def _thinking_of(msg):
    return [b for b in msg["content"] if isinstance(b, dict) and b.get("type") == "thinking"]


def _tool_loop(turns: int):
    """user -> (assistant[thinking+tool_use] -> tool) x turns, as the agent stores it."""
    msgs = [{"role": "user", "content": "do the thing"}]
    for i in range(turns):
        msgs.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"t{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}],
            "reasoning_details": [{"type": "thinking", "thinking": f"step {i}", "signature": f"sig-{i}"}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "name": "terminal", "content": f"out {i}"})
    return msgs


def test_keep_all_models_replay_every_prior_turn_thinking_unchanged():
    for model in ("anthropic/claude-fable-5.1", "claude-opus-4-8", "claude-sonnet-4-6"):
        _s, out = convert_messages_to_anthropic(_tool_loop(3), base_url=PORTAL, model=model)
        assistants = [m for m in out if m["role"] == "assistant"]
        assert [len(_thinking_of(m)) for m in assistants] == [1, 1, 1], model
        assert [t[0]["signature"] for t in map(_thinking_of, assistants)] == ["sig-0", "sig-1", "sig-2"]


def test_keep_all_prefix_is_byte_stable_as_the_loop_grows():
    """The invariant behind the cache: request N's messages must be a prefix of request N+1's."""
    prev = None
    for turns in (1, 2, 3, 4):
        _s, out = convert_messages_to_anthropic(_tool_loop(turns), base_url=PORTAL, model="anthropic/claude-fable-5.1")
        if prev is not None:
            assert out[: len(prev)] == prev, f"prefix mutated between {turns - 1} and {turns} turns"
        prev = copy.deepcopy(out)


def test_last_turn_only_models_still_strip_older_thinking():
    _s, out = convert_messages_to_anthropic(_tool_loop(3), base_url=PORTAL, model="claude-sonnet-4-5")
    assistants = [m for m in out if m["role"] == "assistant"]
    assert [len(_thinking_of(m)) for m in assistants] == [0, 0, 1]


def test_third_party_gateways_strip_regardless_of_model():
    _s, out = convert_messages_to_anthropic(_tool_loop(2), base_url="https://api.minimax.io/anthropic", model="claude-fable-5.1")
    assert all(not _thinking_of(m) for m in out if m["role"] == "assistant")


def test_model_predicate_classifies_generations():
    assert _model_keeps_all_thinking("anthropic/claude-fable-5.1")
    assert _model_keeps_all_thinking("claude-opus-4-5")
    assert _model_keeps_all_thinking("claude-sonnet-4-6")
    assert _model_keeps_all_thinking("some-future-claude")  # unknown => keep-all (the cheap direction)
    assert not _model_keeps_all_thinking("claude-sonnet-4-5")
    assert not _model_keeps_all_thinking("claude-opus-4-1")
    assert not _model_keeps_all_thinking("claude-haiku-4-5")
    assert not _model_keeps_all_thinking("claude-3-5-sonnet")


def test_estimator_counts_retained_thinking_on_the_anthropic_wire():
    """Independent-review witness: with thinking retained, the wire carried ~285K tokens while
    preflight estimated ~9K (its stale-thinking predicate only knew the reasoning-echo families), so
    compression never fired. The single shared predicate must agree with the converter."""
    from agent.message_sanitization import stale_thinking_reaches_wire
    assert stale_thinking_reaches_wire("anthropic_messages", "nous", "anthropic/claude-fable-5.1", "") is True
    assert stale_thinking_reaches_wire("anthropic_messages", "anthropic", "claude-opus-4-8", "") is True
    # last-turn-only generations: the API drops older blocks, so they never reach the wire
    assert stale_thinking_reaches_wire("anthropic_messages", "anthropic", "claude-haiku-4-5", "") is False
    # codex sidecar and non-Anthropic chat wires are unchanged
    assert stale_thinking_reaches_wire("codex_responses", "openai-codex", "gpt-5.6", "") is False
    assert stale_thinking_reaches_wire("chat_completions", "openrouter", "anthropic/claude-fable-5.1", "") is False
