"""Compression output budgets reach an OpenAI-compatible endpoint via the real aux route."""

import json
from unittest.mock import patch

import httpx
from openai import OpenAI

from agent.context_compressor import ContextCompressor


def test_compression_checkpoint_limit_reaches_endpoint(tmp_path, monkeypatch):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if payload.get("stream"):
            chunk = {
                "id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1,
                "model": "checkpoint-test", "choices": [{"index": 0, "finish_reason": None,
                    "delta": {"role": "assistant", "content": "## Active Task\nContinue the task"}}],
            }
            end = {**chunk, "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}]}
            body = f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(end)}\n\ndata: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
            "model": "checkpoint-test",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "## Active Task\nContinue the task",
            }}],
        })

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    base_url = "https://checkpoint.example.test/v1"
    (home / "config.yaml").write_text(
        "auxiliary:\n  compression:\n    provider: custom\n"
        f"    model: checkpoint-test\n    base_url: {base_url}\n"
        "    max_tokens: 12000\n"
    )
    compressor = ContextCompressor(
        model="checkpoint-test", provider="custom", base_url=base_url,
        api_key="local-test-key", config_context_length=128_000, quiet_mode=True,
    )
    transport = httpx.MockTransport(respond)
    def client_factory(**kwargs):
        return OpenAI(
            api_key=kwargs["api_key"], base_url=kwargs["base_url"],
            http_client=httpx.Client(transport=transport),
        )
    with patch("agent.auxiliary_client._create_openai_client", side_effect=client_factory):
        result = compressor._generate_summary([{"role": "user", "content": "Continue the task"}])
    assert result and "Continue the task" in result
    assert len(requests) == 1
    assert requests[0]["max_tokens"] == 12_000
