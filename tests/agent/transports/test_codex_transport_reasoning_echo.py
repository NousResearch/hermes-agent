"""DeepSeek Responses replay contract regression tests."""

from agent.transports.codex import ResponsesApiTransport
from agent.codex_responses_adapter import _preflight_codex_input_items


def test_deepseek_responses_replay_pads_reasoning_text_only_for_required_route():
    history = [{
        "role": "assistant", "content": "", "codex_reasoning_items": [
            {"type": "reasoning", "encrypted_content": "opaque", "summary": []},
        ],
    }]
    transport = ResponsesApiTransport()

    deepseek = transport.convert_messages(history, base_url="https://api.deepseek.com/responses")
    other = transport.convert_messages(history, base_url="https://api.openai.com/v1/responses")
    deepseek_wire = _preflight_codex_input_items(deepseek)
    other_wire = _preflight_codex_input_items(other)

    assert next(item for item in deepseek_wire if item["type"] == "reasoning")["content"] == [
        {"type": "reasoning_text", "text": " "},
    ]
    assert "content" not in next(item for item in other_wire if item["type"] == "reasoning")
