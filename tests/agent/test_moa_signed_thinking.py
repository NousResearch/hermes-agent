"""An Anthropic MoA aggregator gets its own signed thinking back, valid, and never as plain text.

The aggregator runs through the OpenAI-shaped aux adapter and its reply lives in the agent's
chat-completions history. Each hop used to keep only the readable thinking summary, and conversion
then demoted it to assistant text on every later request: a running log of the model's own
reasoning, which Anthropic's classifier refuses as ``reasoning_extraction``. And Anthropic binds a
signed block to everything sent before it, so the trailing reference guidance MoA attaches must stay
where it was sent for the block to replay.
"""
import hashlib
import json
from types import SimpleNamespace

import httpx

THOUGHTS = ("I should read the note first.", "The note says hello.", "Same note, still hello.", "A fresh start.")
ANTHROPIC_RUNTIME = {"provider": "anthropic", "model": "claude-fable-5-2", "api_mode": "anthropic_messages",
                     "base_url": "https://api.anthropic.com", "api_key": "k"}


def _signature_400():
    import anthropic

    body = {"type": "error", "error": {"type": "invalid_request_error", "message": (
        "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation.")}}
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    return anthropic.BadRequestError(f"Error code: 400 - {body}", response=response, body=body)


def _prefix_signature(request, messages, earlier_blocks):
    """Digest of what precedes a block: system, tools, every earlier message, the turn's earlier blocks.
    Thinking itself and cache markers are not part of it, as on the real API."""
    def blocks(content):
        content = [{"type": "text", "text": content}] if isinstance(content, str) else content
        return [{k: v for k, v in b.items() if k != "cache_control"} for b in content
                if b.get("type") not in ("thinking", "redacted_thinking")]

    prefix = {"system": blocks(request.get("system") or []), "tools": request.get("tools"),
              "messages": [[m["role"], blocks(m["content"])] for m in messages], "turn": blocks(earlier_blocks)}
    return "sig-" + hashlib.sha256(json.dumps(prefix, sort_keys=True, default=str).encode()).hexdigest()[:24]


class _PreservedThinkingAPI:
    """Anthropic Messages with preserved thinking: each signed block is bound to its prefix, and one replayed
    behind any other prefix is answered with the 400 the API gives on the models that enforce the check."""

    def __init__(self, replies):
        self.replies, self.requests, self.replayed, self.rejected = list(replies), [], [], 0

    def create(self, **request):
        self.requests.append(request)
        replayed = 0
        for i, message in enumerate(request["messages"]):
            content = message["content"] if message["role"] == "assistant" and isinstance(message["content"], list) else []
            for j, block in enumerate(content):
                if block.get("type") == "thinking":
                    if block.get("signature") != _prefix_signature(request, request["messages"][:i], content[:j]):
                        self.rejected += 1
                        raise _signature_400()
                    replayed += 1
        self.replayed.append(replayed)
        thought, rest = self.replies.pop(0)
        signature = _prefix_signature(request, request["messages"], [])
        content = [SimpleNamespace(type="thinking", thinking=thought, signature=signature),
                   *(SimpleNamespace(**block) for block in rest)]
        stop = "tool_use" if any(block["type"] == "tool_use" for block in rest) else "end_turn"
        return SimpleNamespace(content=content, stop_reason=stop, model="claude-fable-5-2",
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def _moa_agent(monkeypatch, tmp_path, api):
    from agent.auxiliary_client import _AnthropicCompletionsAdapter
    from run_agent import AIAgent

    adapter = _AnthropicCompletionsAdapter(SimpleNamespace(messages=api), "claude-fable-5-2",
                                           base_url="https://api.anthropic.com")

    def fake_call_llm(**kw):
        if kw["task"] == "moa_reference":
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=f"advice #{len(api.requests)}: read the note", tool_calls=[]), finish_reason="stop")],
                usage=None, model="gpt-6-astra")
        return adapter.create(model=kw["model"], messages=kw["messages"], tools=kw.get("tools"),
                              max_tokens=kw.get("max_tokens"), _reasoning_config=kw.get("reasoning_config"))

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    monkeypatch.setattr("agent.moa_loop._slot_runtime", lambda slot: (
        dict(ANTHROPIC_RUNTIME) if slot.get("provider") == "anthropic"
        else {"provider": slot.get("provider"), "model": slot.get("model"), "api_mode": "codex_responses"}))
    monkeypatch.setattr("agent.auxiliary_client._aux_progress_active", lambda: False)
    agent = AIAgent(api_key="moa-virtual-provider", base_url="http://127.0.0.1/v1", model="review", provider="moa",
                    quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["file"],
                    max_iterations=4)
    agent._disable_streaming = True
    return agent


def test_the_aggregator_replays_valid_signed_thinking_across_tool_steps_and_turns(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("""
moa:
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openai-codex
          model: gpt-6-astra
      aggregator:
        provider: anthropic
        model: claude-fable-5-2
        reasoning_effort: high
""".strip(), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    note = tmp_path / "note.txt"
    note.write_text("hello\n", encoding="utf-8")
    api = _PreservedThinkingAPI([
        (THOUGHTS[0], [{"type": "text", "text": "Reading it."},
                       {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": str(note)}}]),
        (THOUGHTS[1], [{"type": "text", "text": "hello"}]),
        (THOUGHTS[2], [{"type": "text", "text": "still hello"}]),
        (THOUGHTS[3], [{"type": "text", "text": "hello again"}]),
    ])

    agent = _moa_agent(monkeypatch, tmp_path, api)
    first = agent.run_conversation(f"What does {note} say?")
    second = agent.run_conversation("And now?", conversation_history=first["messages"])
    # A new facade (the gateway evicted the agent, or restarted) cannot rebuild what the old one sent.
    third = _moa_agent(monkeypatch, tmp_path, api).run_conversation("Once more?", conversation_history=second["messages"])

    assert [first["final_response"], second["final_response"], third["final_response"]] == ["hello", "still hello",
                                                                                           "hello again"]
    assert api.rejected == 0  # every replayed block sat behind the prefix it was signed for
    assert api.replayed[1] >= 1 and api.replayed[2] >= 1  # the tool step and the next turn got thinking back
    assert api.replayed[3] == 0  # the fresh facade dropped what it could not vouch for
    for request in api.requests:  # and never as assistant text
        assert not any(t in json.dumps(m["content"]) for m in request["messages"] if m["role"] == "assistant"
                       for t in THOUGHTS if not any(b.get("type") == "thinking" for b in m["content"]
                                                    if isinstance(m["content"], list)))


def _assistant_turn():
    signed = {"type": "thinking", "thinking": THOUGHTS[0], "signature": "sig-1"}
    return {"role": "assistant", "content": "Reading it.", "reasoning": THOUGHTS[0], "reasoning_content": THOUGHTS[0],
            "reasoning_details": [signed], "anthropic_content_blocks": [signed, {"type": "text", "text": "Reading it."}]}


def test_a_rejected_signature_is_dropped_for_the_session_and_retried_once():
    from agent.moa_aggregator_wire import AggregatorWire

    wire, sent = AggregatorWire(), []

    def send(*, messages):
        sent.append(messages)
        if len(sent) == 1:
            raise _signature_400()
        return "ok"

    assert wire.send_with_recovery(send, "anthropic:claude", messages=[_assistant_turn()]) == "ok"
    retried = sent[1][0]
    assert not any(b.get("type") == "thinking" for key in ("reasoning_details", "anthropic_content_blocks")
                   for b in retried.get(key) or ())
    assert retried["anthropic_content_blocks"] == [{"type": "text", "text": "Reading it."}]  # the rest of the turn stays
    assert len(wire.rejected) == 1


def test_other_aggregators_keep_the_shape_they_had():
    from agent.moa_aggregator_wire import aggregator_reasoning_view, replays_signed_thinking

    runtime = {"provider": "openrouter", "api_mode": "chat_completions"}
    shaped = aggregator_reasoning_view([_assistant_turn()], runtime)[0]
    assert shaped["reasoning_content"] == THOUGHTS[0] and shaped["reasoning"] == THOUGHTS[0]
    assert "reasoning_details" not in shaped and "anthropic_content_blocks" not in shaped
    assert not replays_signed_thinking(runtime)  # so its guidance still moves to the end of each request
