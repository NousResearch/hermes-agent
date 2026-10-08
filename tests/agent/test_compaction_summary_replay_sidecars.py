"""Compaction summary must reach the wire when the merged tail row carried a replay sidecar.

``_merge_summary_into_tail_row`` rewrites the carried tail row's ``content``. The Anthropic,
Bedrock and Codex Responses converters replay a provider-native sidecar
(``anthropic_content_blocks`` / ``bedrock_content_blocks`` / ``codex_message_items``) verbatim and
ignore ``content``, so a stale sidecar silently drops the summary from the request.
"""

import copy
import json
from unittest.mock import MagicMock, patch

import pytest

from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.bedrock_adapter import convert_messages_to_converse
from agent.codex_responses_adapter import _chat_messages_to_responses_input
from agent.context_compressor import ContextCompressor

_MARKER = "SUMMARY_MARKER_7f3a"


def _tool_call(i):
    return {"id": f"call_{i:02d}", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"n": i})}}


def _sidecar(wire, i):
    if wire == "anthropic":
        return {"anthropic_content_blocks": [
            {"type": "thinking", "thinking": f"thought {i}", "signature": f"sig{i:02d}"},
            {"type": "text", "text": f"Reading slice {i}"},
            {"type": "tool_use", "id": f"call_{i:02d}", "name": "read_file", "input": {"n": i}},
        ]}
    if wire == "bedrock":
        return {"bedrock_content_blocks": [
            {"text": f"Reading slice {i}"},
            {"toolUse": {"toolUseId": f"call_{i:02d}", "name": "read_file", "input": {"n": i}}},
        ]}
    return {"codex_message_items": [{
        "type": "message", "role": "assistant", "id": f"msg_{i:02d}", "status": "completed", "phase": "commentary",
        "content": [{"type": "output_text", "text": f"Reading slice {i}"}],
    }]}


def _history(wire, with_sidecar):
    def assistant_tool_call(i):
        row = {"role": "assistant", "content": f"Reading slice {i}", "tool_calls": [_tool_call(i)]}
        return {**row, **_sidecar(wire, i)} if with_sidecar else row

    def tool_result(i):
        return {"role": "tool", "tool_call_id": f"call_{i:02d}", "content": f"slice {i} " + "x" * 200}

    history = [{"role": "system", "content": "system prompt"}]
    for i in range(1, 4):
        history += [{"role": "user", "content": f"task {i}"}, assistant_tool_call(i), tool_result(i),
                    {"role": "assistant", "content": f"done {i}"}]
    return history + [{"role": "user", "content": "task four"}, assistant_tool_call(4), tool_result(4)]


def _wire_text(wire, messages):
    messages = copy.deepcopy(messages)
    if wire == "anthropic":
        return json.dumps(convert_messages_to_anthropic(messages, base_url="https://api.anthropic.com", model="claude-sonnet-5-5"))
    if wire == "bedrock":
        return json.dumps(convert_messages_to_converse(messages), default=str)
    return json.dumps(_chat_messages_to_responses_input(messages, current_issuer_kind="openai", current_issuer_model="gpt-5"))


def _compress(history):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = f"{_MARKER}: three tasks done; nothing pending."
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test", quiet_mode=True, protect_first_n=3, protect_last_n=6)
    with patch("agent.context_compressor.call_llm", return_value=response):
        return compressor.compress(history)


@pytest.mark.parametrize("wire", ["anthropic", "bedrock", "codex"])
@pytest.mark.parametrize("with_sidecar", [False, True], ids=["no-sidecar", "replay-sidecar"])
def test_merged_summary_reaches_the_wire(wire, with_sidecar):
    compressed = _compress(_history(wire, with_sidecar))
    assert _MARKER in json.dumps(compressed)
    assert _MARKER in _wire_text(wire, compressed)
