"""Stale-call watchdog estimator prices images at the learned per-image cost, not their base64
length (#63871 / #76411; salvage of #76471 by @crdesign8). A single native screenshot read as
~100K+ tokens and selected the 600-1200s giant-conversation watchdog tiers while the provider's
real prompt was a fraction of that."""

from agent.chat_completion_helpers import estimate_request_context_tokens, openai_codex_stale_timeout_floor
from agent.image_token_cost import DEFAULT_IMAGE_TOKEN_COST, image_cost_context

_B64 = "data:image/png;base64," + "A" * 400_000


def _chat(*parts):
    return {"messages": [{"role": "user", "content": list(parts)}], "tools": [{"type": "function", "function": {"name": "t"}}]}


def test_image_parts_cost_the_learned_price_on_both_wire_shapes():
    chat = _chat({"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": _B64}})
    responses = {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": _B64}]}], "instructions": "i" * 400}
    default = estimate_request_context_tokens(chat)
    assert DEFAULT_IMAGE_TOKEN_COST <= default < DEFAULT_IMAGE_TOKEN_COST + 200
    assert DEFAULT_IMAGE_TOKEN_COST <= estimate_request_context_tokens(responses) < DEFAULT_IMAGE_TOKEN_COST + 200
    with image_cost_context(4_000):
        assert 4_000 <= estimate_request_context_tokens(chat) < 4_200
    # The watchdog tier follows: one screenshot no longer buys the 100K+ (1200s) floor.
    assert openai_codex_stale_timeout_floor(default) == 0.0


def test_text_payloads_and_tool_schemas_keep_the_legacy_estimate():
    """A base64-looking STRING is text; tool schemas mentioning ``image`` are not images."""
    payload = {"messages": [{"role": "user", "content": _B64}],
               "tools": [{"type": "function", "function": {"name": "vision", "parameters": {"type": "image"}}}]}
    legacy = (sum(len(str(m)) for m in payload["messages"]) + len(str(payload["tools"]))) // 4
    assert abs(estimate_request_context_tokens(payload) - legacy) < legacy * 0.02


def test_schema_nodes_with_structured_type_values_keep_the_legacy_estimate():
    """A JSON-Schema ``properties`` dict whose ``type`` KEY holds a sub-schema dict, or a multi-type
    list like ``["string", "null"]``, is payload data — not an image part. The membership test must
    not raise ``TypeError: unhashable type`` before the request reaches the provider (#104793)."""
    def _tools(param_name):
        return [{"type": "function", "function": {"name": "memory_search", "parameters": {"type": "object", "properties": {
            param_name: {"type": "string", "enum": ["episode", "semantic"]},
            "maybe": {"type": ["string", "null"]},
        }}}}]
    messages = [{"role": "user", "content": "hi"}]
    payload = {"messages": messages, "tools": _tools("type")}
    renamed = {"messages": messages, "tools": _tools("kind")}  # equal-length key: identical walk
    assert estimate_request_context_tokens(payload) == estimate_request_context_tokens(renamed)
    # A real image part in the same request is still priced at the learned cost.
    chat = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": _B64}},
    ]}], "tools": _tools("type")}
    assert DEFAULT_IMAGE_TOKEN_COST <= estimate_request_context_tokens(chat) < DEFAULT_IMAGE_TOKEN_COST + 400
