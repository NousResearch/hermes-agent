"""Responses output must survive a stored assistant turn and the next request (#157124)."""
from types import SimpleNamespace as Item

import pytest

from agent.transports.codex import ResponsesApiTransport
from persistence.conversation.messages import NewMessage
from persistence_assembly import open_persistence, profile_scope
from run_agent import AIAgent


def _message(*texts, phase="final_answer"):
    return Item(type="message", id="msg_fixture", role="assistant", status="completed", phase=phase,
                content=[Item(type="output_text", text=text) for text in texts])


def _reasoning(blob):
    return Item(type="reasoning", id=f"rs_{blob}", encrypted_content=blob, summary=[])


def _call(call_id):
    return Item(type="function_call", call_id=call_id, name="fixture_read", arguments="{}", status="completed")


@pytest.fixture
def agent(tmp_path):
    store = open_persistence(profile_scope(home=tmp_path))
    agent = AIAgent(model="gpt-6.2-sol", provider="openai-codex", api_mode="codex_responses",
                    base_url="https://chatgpt.com/backend-api/codex", api_key="unused",
                    enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True, persistence=store)
    agent._ensure_db_session()
    try:
        yield agent
    finally:
        agent.close()
        store.close()


@pytest.fixture
def roundtrip(agent):
    transport = ResponsesApiTransport()

    def request(messages):
        kwargs = transport.build_kwargs(model=agent.model, provider=agent.provider, base_url=agent.base_url,
                                        is_codex_backend=True, messages=messages, instructions="Stay concise.")
        return transport.preflight_kwargs(kwargs, sanitize_harmony_tokens=True)["input"]

    def replay(response):
        user = {"role": "user", "content": "Read the fixture."}
        request([user])
        normalized = transport.normalize_response(response)
        row = agent._build_assistant_message(normalized, normalized.finish_reason)
        outputs = [{"role": "tool", "tool_call_id": tc.id, "content": "ok"} for tc in normalized.tool_calls or []]
        transcript = agent.persistence.messages
        for msg in [user, row, *outputs, {"role": "user", "content": "Continue."}]:
            transcript.append(agent.session_id, NewMessage.from_live(msg))
        history = [m.as_openai_message() for m in transcript.read_conversation(agent.session_id)]
        return normalized.content, request(history)[1:-1]

    return replay


def test_mixed_output_replays_in_provider_order_with_safe_ids(roundtrip):
    output = [_reasoning("one"), _message("Checking.", phase="commentary"), _call("call_a"),
              _reasoning("two"), _call("call_b")]
    _, wire = roundtrip(Item(status="completed", output=output))
    assert [item["type"] for item in wire] == [i.type for i in output] + ["function_call_output"] * 2
    assert [i["encrypted_content"] for i in wire if i["type"] == "reasoning"] == ["one", "two"]
    assert wire[1]["phase"] == "commentary"
    assert not any("id" in i for i in wire)
    assert [i["call_id"] for i in wire if "call_id" in i] == ["call_a", "call_b", "call_a", "call_b"]


@pytest.mark.parametrize("parts", [["  indented", "", " line\n"], [" \t\n"], ["ordinary text"]])
def test_exact_message_parts_replay_independently_of_display_text(roundtrip, parts):
    display, wire = roundtrip(Item(status="completed", output=[_message(*parts)]))
    assert (display or "") == "".join(parts).strip()
    assert wire == [{"type": "message", "role": "assistant", "status": "completed", "phase": "final_answer",
                     "id": "msg_fixture", "content": [{"type": "output_text", "text": text} for text in parts]}]


@pytest.mark.parametrize("parts", [[""], ["", ""]])
def test_all_empty_message_item_is_not_replayed_before_tool_call(roundtrip, parts):
    _, wire = roundtrip(Item(status="completed", output=[_reasoning("one"), _message(*parts), _call("call_a")]))
    assert [item["type"] for item in wire] == ["reasoning", "function_call", "function_call_output"]


def test_all_empty_message_item_without_call_uses_nonempty_reasoning_follower(roundtrip):
    _, wire = roundtrip(Item(status="completed", output=[_reasoning("one"), _message("")]))
    assert wire == [
        {"type": "reasoning", "encrypted_content": "one", "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": " "}]},
    ]


def test_checkpoint_kept_from_prior_partial_precedes_refreshed_items(agent, monkeypatch):
    from agent.codex_responses_adapter import _chat_messages_to_responses_input

    checkpoint = Item(type="compaction", encrypted_content="checkpoint")
    partial = _message("Partial", phase=None)
    responses = iter([Item(status="in_progress", output=[partial, checkpoint]),
                      Item(status="in_progress", output=[_reasoning("new"), partial]),
                      Item(status="completed", output=[_message("Done")])])
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda kwargs: next(responses))
    result = agent.run_conversation("Read the fixture.")
    history = [m for m in result["messages"] if m.get("finish_reason") == "incomplete"]
    assert len(history) == 1
    wire = _chat_messages_to_responses_input(history, native_compaction_eligible=True)
    assert [i["type"] for i in wire[:3]] == ["compaction", "reasoning", "message"]
    assert wire[1]["encrypted_content"] == "new"
