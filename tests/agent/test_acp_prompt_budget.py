"""Copilot's text-shaped budget refusal must enter Hermes' overflow recovery."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import cast

import pytest

from agent.acp_client import ACPClient
from agent.error_classifier import FailoverReason, classify_api_error
from agent.model_metadata import get_cached_context_length
from agent.turn_overflow import _Recovery, _adopt_provider_context_limit


def _client(tmp_path, text):
    # Exercise the real ACP handshake, notification collection and completion facade.
    script = tmp_path / "agent.py"
    script.write_text(
        "import json, sys\n"
        f"text = {text!r}\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    method = request['method']\n"
        "    result = {}\n"
        "    if method == 'initialize':\n"
        "        result = {'protocolVersion': 1}\n"
        "    elif method == 'session/new':\n"
        "        result = {'sessionId': 'budget-test'}\n"
        "    elif method == 'session/prompt':\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'method': 'session/update',\n"
        "            'params': {'update': {'sessionUpdate': 'agent_message_chunk',\n"
        "                'content': {'type': 'text', 'text': text}}}}), flush=True)\n"
        "        result = {'stopReason': 'end_turn'}\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}), flush=True)\n",
        encoding="utf-8",
    )
    return ACPClient(command=sys.executable, args=[str(script)], acp_cwd=str(tmp_path))


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("input_tokens,limit", [(273582, 200000), (274889, 272000)])
def test_text_refusal_enters_overflow_and_adopts_channel_budget(tmp_path, stream, input_tokens, limit):
    text = (
        "Error: Execution failed: Request exceeds the model's prompt budget after reserving output "
        f"({input_tokens} tokens; limit {limit}). Reduce context or compact before continuing."
    )
    client = _client(tmp_path, text)
    with pytest.raises(RuntimeError, match="prompt budget") as raised:
        client.chat.completions.create(messages=[{"role": "user", "content": "hi"}], stream=stream, timeout=15)
    error = raised.value
    classified = classify_api_error(error, provider="copilot-acp", model="budget-test")
    assert classified.reason == FailoverReason.context_overflow
    assert classified.should_compress

    compressor = SimpleNamespace(context_length=1_000_000)

    def update_model(**kwargs):
        compressor.context_length = kwargs["context_length"]

    compressor.update_model = update_model
    agent = SimpleNamespace(
        model="budget-test", base_url="acp://copilot", provider="copilot-acp", api_mode="chat_completions",
        context_compressor=compressor, _buffer_vprint=lambda *args: None,
    )
    state = cast(_Recovery, SimpleNamespace(agent=agent))
    assert _adopt_provider_context_limit(state, str(error), 1_000_000) == limit
    assert compressor.context_length == limit
    # Real persistence in the runner's isolated HERMES_HOME, not a mocked save call.
    assert get_cached_context_length(agent.model, agent.base_url) == limit


def test_answer_quoting_budget_error_is_still_a_normal_completion(tmp_path):
    text = "The CLI returned: Error: Execution failed: Request exceeds the model's prompt budget after reserving output."
    client = _client(tmp_path, text)
    response = client.chat.completions.create(messages=[{"role": "user", "content": "explain"}], timeout=15)
    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content == text
