"""Streamed ``reasoning_details`` survive to the assembled response (replay continuity).

The non-streaming path always kept OpenRouter's ``reasoning_details`` (signatures,
encrypted blocks a provider needs back on the next turn); the streaming chunk loop
dropped them. Consecutive text/summary fragments merge into one logical entry,
encrypted entries stay discrete.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.reasoning_summaries import append_streamed_reasoning_detail
from agent.transports import get_transport


def _make_chunk(content=None, finish_reason=None, model=None, reasoning_details=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None, reasoning=None)
    if reasoning_details is not None:
        delta.reasoning_details = reasoning_details
    return SimpleNamespace(choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
                           model=model, usage=usage)


def test_fragments_merge_per_block_and_backfill_signature():
    acc = []
    append_streamed_reasoning_detail(acc, {"type": "reasoning.text", "text": "The user "})
    append_streamed_reasoning_detail(acc, SimpleNamespace(type="reasoning.text", text="wants X.", signature="sig1"))
    append_streamed_reasoning_detail(acc, {"type": "reasoning.encrypted", "data": "AAAA"})
    append_streamed_reasoning_detail(acc, {"type": "reasoning.encrypted", "data": "BBBB"})
    append_streamed_reasoning_detail(acc, {"type": "reasoning.summary", "summary": "s1 "})
    append_streamed_reasoning_detail(acc, {"type": "reasoning.summary", "summary": "s2"})
    assert [d["type"] for d in acc] == [
        "reasoning.text", "reasoning.encrypted", "reasoning.encrypted", "reasoning.summary"]
    assert acc[0] == {"type": "reasoning.text", "text": "The user wants X.", "signature": "sig1"}
    assert acc[3]["summary"] == "s1 s2"


@pytest.mark.parametrize("dtype,text_key", [("reasoning.text", "text"), ("reasoning.summary", "summary")])
@pytest.mark.parametrize("identity,first,second", [("index", 0, 1), ("id", "first", "second")])
@pytest.mark.parametrize("shared_identity", [False, True])
def test_fragments_preserve_distinct_block_identities(dtype, text_key, identity, first, second, shared_identity):
    acc = []
    common = ({"id": "shared"} if identity == "index" else {"index": 0}) if shared_identity else {}
    for block, text in [(first, "First "), (first, "block"), (second, "Second "), (second, "block")]:
        append_streamed_reasoning_detail(acc, {"type": dtype, text_key: text, identity: block, **common})

    assert acc == [
        {"type": dtype, text_key: "First block", identity: first, **common},
        {"type": dtype, text_key: "Second block", identity: second, **common},
    ]


def test_fragment_identity_backfill_retains_zero_index_and_signature():
    acc = []
    append_streamed_reasoning_detail(acc, {"type": "reasoning.text", "text": "First "})
    append_streamed_reasoning_detail(acc, {
        "type": "reasoning.text", "text": "block", "index": 0, "id": "first", "signature": "sig1",
    })
    append_streamed_reasoning_detail(acc, {"type": "reasoning.text", "text": "."})
    append_streamed_reasoning_detail(acc, {
        "type": "reasoning.text", "text": "Second", "index": 1, "id": "second", "signature": "sig2",
    })

    assert acc == [
        {"type": "reasoning.text", "text": "First block.", "index": 0, "id": "first", "signature": "sig1"},
        {"type": "reasoning.text", "text": "Second", "index": 1, "id": "second", "signature": "sig2"},
    ]


def _agent():
    from run_agent import AIAgent
    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_streamed_details_keep_block_identity_through_replay(_mock_close, mock_create):
    chunks = [
        _make_chunk(reasoning_details=[{"type": "reasoning.text", "text": "I should ", "index": 0}]),
        _make_chunk(reasoning_details=[{"type": "reasoning.text", "text": "answer.", "signature": "sigZ", "index": 0, "id": "first"}]),
        _make_chunk(reasoning_details=[{"type": "reasoning.text", "text": "Next thought.", "signature": "sigY", "index": 1, "id": "second"}]),
        _make_chunk(content="Hello!", finish_reason="stop", model="test-model"),
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter(chunks)
    mock_create.return_value = mock_client

    agent = _agent()
    delivered = []
    agent.reasoning_callback = delivered.append
    agent.stream_delta_callback = lambda text: None

    def streamed_chunks():
        yield chunks[0]
        assert delivered == ["I should "]
        yield chunks[1]
        assert delivered == ["I should ", "answer."]
        yield chunks[2]
        yield chunks[3]

    mock_client.chat.completions.create.return_value = streamed_chunks()
    response = agent._interruptible_streaming_api_call({})
    assert "".join(delivered) == "I should answer.Next thought."
    msg = response.choices[0].message
    assert msg.content == "Hello!"
    expected_details = [
        {"type": "reasoning.text", "text": "I should answer.", "signature": "sigZ", "index": 0, "id": "first"},
        {"type": "reasoning.text", "text": "Next thought.", "signature": "sigY", "index": 1, "id": "second"},
    ]
    assert msg.reasoning_details == expected_details
    assistant_message = agent._build_assistant_message(msg, "stop")
    history = [{"role": "user", "content": "Hello"}, assistant_message, {"role": "user", "content": "Continue"}]
    transport = get_transport("chat_completions")
    for base_url in ("https://openrouter.ai/api/v1", "https://inference-api.nousresearch.com/v1"):
        kwargs = transport.build_kwargs("test/model", history, base_url=base_url)
        assert kwargs["messages"][1]["reasoning_details"] == expected_details
    assert assistant_message["reasoning_details"] == expected_details


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_live_details_deglue_summary_part_boundaries(_mock_close, mock_create):
    """The live display gets the same boundary repair as the persisted reasoning.

    A reasoning-summary model streams one delta per completed summary part, each
    a bare bold heading. Without the ``separate_glued_reasoning_blocks`` repair
    on the detail path, the live box glues parts head-to-tail while
    ``reasoning_content`` stays de-glued — display and history disagree.
    """
    agent = _agent()
    client = MagicMock()
    mock_create.return_value = client

    def summary_chunk(summary, **kw):
        chunk = _make_chunk(**kw)
        chunk.choices[0].delta.reasoning = summary  # provider mirrors both fields
        chunk.choices[0].delta.model_extra = {
            "reasoning_details": [{"type": "reasoning.summary", "summary": summary}]}
        return chunk

    client.chat.completions.create.return_value = iter([
        summary_chunk("**One**"),
        summary_chunk("**Two**"),
        _make_chunk(content="Answer", finish_reason="stop", model="test-model"),
    ])
    delivered = []
    agent.reasoning_callback = delivered.append
    response = agent._interruptible_streaming_api_call({})

    assert "".join(delivered) == "**One**\n\n**Two**"
    assert response.choices[0].message.reasoning_content == "**One**\n\n**Two**"

    # Details-only variant: the delta carries no plain ``reasoning`` field at all.
    client.chat.completions.create.return_value = iter([
        _make_chunk(reasoning_details=[{"type": "reasoning.summary", "summary": "**One**"}]),
        _make_chunk(reasoning_details=[{"type": "reasoning.summary", "summary": "**Two**"}]),
        _make_chunk(content="Answer", finish_reason="stop", model="test-model"),
    ])
    delivered = []
    agent.reasoning_callback = delivered.append
    response = agent._interruptible_streaming_api_call({})

    assert "".join(delivered) == "**One**\n\n**Two**"


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_no_details_leaves_attribute_absent(_mock_close, mock_create):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter([
        _make_chunk(content="plain", finish_reason="stop", model="test-model")])
    mock_create.return_value = mock_client
    response = _agent()._interruptible_streaming_api_call({})
    assert not hasattr(response.choices[0].message, "reasoning_details")


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_live_details_keep_plain_fallback_and_opaque_replay(_mock_close, mock_create):
    agent = _agent()
    client = MagicMock()
    mock_create.return_value = client
    cases = [
        ([{"type": "reasoning.text", "text": "Complete thought"}], "C", "Complete thought"),
        ([{"type": "reasoning.text", "text": "Same"}], "Same", "Same"),
        ([SimpleNamespace(type="reasoning.summary", summary="Summary")], None, "Summary"),
        ([{"type": "reasoning.encrypted", "data": "secret", "text": "not readable"}], "Plain", "Plain"),
        ([{"type": "unknown", "text": "not readable"}], "Plain", "Plain"),
        ([{"type": "reasoning.text", "text": ""}], "Plain", "Plain"),
        ([], "Plain", "Plain"),
        ([{"type": "reasoning.encrypted", "data": "secret"}], None, ""),
    ]
    for details, plain, expected in cases:
        chunk = _make_chunk(content="Answer", finish_reason="stop")
        chunk.choices[0].delta.reasoning = plain
        chunk.choices[0].delta.model_extra = {"reasoning_details": details}
        client.chat.completions.create.return_value = iter([chunk])
        delivered = []
        agent.reasoning_callback = delivered.append
        response = agent._interruptible_streaming_api_call({})
        assert "".join(delivered) == expected
        assert response.choices[0].message.content == "Answer"
        assert response.choices[0].message.reasoning_content == plain
        preserved = []
        for detail in details:
            append_streamed_reasoning_detail(preserved, detail)
        assert getattr(response.choices[0].message, "reasoning_details", []) == preserved

    def broken_callback(text):
        raise RuntimeError("consumer failed")

    for callback in (None, broken_callback):
        agent.reasoning_callback = callback
        client.chat.completions.create.return_value = iter([
            _make_chunk(content="Answer", finish_reason="stop", reasoning_details=[
                {"type": "reasoning.text", "text": "Thought"}])])
        assert agent._interruptible_streaming_api_call({}).choices[0].message.content == "Answer"
