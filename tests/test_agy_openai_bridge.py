from pathlib import Path

from scripts.agy_openai_bridge import (
    agy_prompt_argument,
    build_agy_prompt,
    build_completion,
    extract_final_response,
    extract_tool_calls,
    stream_completion,
)


def test_prompt_contains_conversation_and_tool_contract():
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }

    prompt = build_agy_prompt(payload)

    assert "User:\nhello" in prompt
    assert '"name": "lookup"' in prompt
    assert "<tool_call>" in prompt


def test_extract_tool_calls_removes_protocol_block():
    text = (
        'before <tool_call>{"id":"call_1","type":"function","function":'
        '{"name":"lookup","arguments":"{\\"q\\":\\"x\\"}"}}</tool_call> after'
    )

    calls, cleaned = extract_tool_calls(text)

    assert calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
        }
    ]
    assert cleaned == "before\nafter"


def test_extract_final_response_drops_progress_output():
    text = (
        "I will inspect the request.\n"
        "<hermes_final>The actual answer.</hermes_final>\n"
    )

    assert extract_final_response(text) == "The actual answer."


def test_completion_and_stream_use_openai_shapes():
    completion = build_completion(
        {"model": "gemini-3.5-flash-high"},
        "bridge response",
    )

    assert completion["model"] == "gemini-3.5-flash-high"
    assert completion["choices"][0]["message"]["content"] == "bridge response"
    assert completion["choices"][0]["finish_reason"] == "stop"

    chunks = list(stream_completion(completion))
    assert '"object": "chat.completion.chunk"' in chunks[0]
    assert '"content": "bridge response"' in chunks[0]
    assert chunks[-1] == "data: [DONE]\n\n"


def test_large_prompt_uses_private_temporary_file(tmp_path: Path):
    prompt = "x" * 100_001

    with agy_prompt_argument(prompt, tmp_path) as argument:
        prompt_path = Path(argument.split("@", 1)[1].split(". Follow", 1)[0])
        assert prompt_path.read_text(encoding="utf-8") == prompt
        assert prompt_path.stat().st_mode & 0o777 == 0o600

    assert not prompt_path.exists()
