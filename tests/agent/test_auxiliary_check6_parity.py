"""Check-6 parity: the Codex auxiliary adapter must reject leaked tool-call
text and preserve reasoning state, mirroring the main transport's
``_normalize_codex_response`` behavior.

The MoA auxiliary path (``_CodexCompletionsAdapter``) previously joined
message text raw: a model that emitted ``assistant to=functions.foo {...}``
as plain text (instead of a structured ``function_call`` item) produced a
confident-looking summary with no tools executed and no audit trail. The
main transport detects this (``_TOOL_CALL_LEAK_PATTERN``), clears the text,
and returns ``finish_reason="incomplete"`` so the continuation path can
re-elicit a proper tool call. The auxiliary adapter must do the same, and
must also preserve encrypted reasoning items for multi-turn continuity.
"""

from types import SimpleNamespace

from agent.auxiliary_client import _CodexCompletionsAdapter


def _adapter_for_items(items):
    events = [
        SimpleNamespace(type="response.created"),
        *[
            SimpleNamespace(type="response.output_item.done", item=item)
            for item in items
        ],
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                status="completed",
                id="resp_test",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
            ),
        ),
    ]

    class _FakeCreateStream:
        def __iter__(self):
            return iter(events)

        def close(self):
            pass

    def _create(**kwargs):
        return _FakeCreateStream()

    client = SimpleNamespace(
        base_url="https://chatgpt.com/backend-api/codex",
        responses=SimpleNamespace(create=_create),
    )
    return _CodexCompletionsAdapter(client, "gpt-5.6-sol")


def test_aux_adapter_rejects_leaked_tool_call_text_and_preserves_reasoning():
    reasoning = SimpleNamespace(
        type="reasoning",
        id="rs_1",
        status="completed",
        encrypted_content="sealed-reasoning",
        summary=[SimpleNamespace(type="summary_text", text="Need a tool.")],
    )
    leaked = SimpleNamespace(
        type="message",
        role="assistant",
        status="completed",
        content=[
            SimpleNamespace(
                type="output_text",
                text='assistant to=functions.exec_command {"command":"pwd"}',
            )
        ],
    )
    response = _adapter_for_items([reasoning, leaked]).create(
        messages=[{"role": "user", "content": "Run pwd."}],
        tools=[{"type": "function", "function": {"name": "terminal", "parameters": {}}}],
    )
    msg = response.choices[0].message
    assert response.choices[0].finish_reason == "incomplete"
    assert msg.content == ""
    assert msg.tool_calls == []
    assert msg.codex_reasoning_items[0]["encrypted_content"] == "sealed-reasoning"


def test_aux_adapter_preserves_clean_text_and_tool_calls():
    function_call = SimpleNamespace(
        type="function_call",
        id="fc_1",
        call_id="call_1",
        name="terminal",
        arguments='{"command":"pwd"}',
        status="completed",
    )
    response = _adapter_for_items([function_call]).create(
        messages=[{"role": "user", "content": "Run pwd."}],
        tools=[{"type": "function", "function": {"name": "terminal", "parameters": {}}}],
    )
    call = response.choices[0].message.tool_calls[0]
    assert response.choices[0].finish_reason == "tool_calls"
    assert call.function.name == "terminal"
    assert call.function.arguments == '{"command":"pwd"}'


def test_aux_adapter_neutralizes_harmony_tokens_in_text():
    clean = SimpleNamespace(
        type="message",
        role="assistant",
        status="completed",
        content=[
            SimpleNamespace(
                type="output_text",
                text="Summary with <|channel|> reserved token",
            )
        ],
    )
    response = _adapter_for_items([clean]).create(
        messages=[{"role": "user", "content": "Summarize."}],
    )
    msg = response.choices[0].message
    assert response.choices[0].finish_reason == "stop"
    assert "<|channel|>" not in msg.content
    assert "｜" in msg.content
