"""Gemini requests use defaults for sampling and documented thinking levels."""
import asyncio
import json
import runpy
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from agent.auxiliary_client import _build_call_kwargs
from agent.gemini_native_adapter import AsyncGeminiNativeClient, GeminiNativeClient
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile


@pytest.mark.parametrize("provider", ["gemini", "openrouter", "nous"])
def test_main_and_aux_serialized_gemini_requests_omit_sampling(provider):
    profile = get_provider_profile(provider)
    model = "google/gemini-3-flash-preview" if provider != "gemini" else "gemini-3-flash-preview"
    base = "https://generativelanguage.googleapis.com/v1beta/openai/" if provider == "gemini" else profile.base_url
    captured = []

    def receive(request):
        captured.append(json.loads(request.content))
        if captured[-1].get("stream"):
            event = {"id": "test", "object": "chat.completion.chunk", "created": 1,
                "model": model, "choices": [{"index": 0, "finish_reason": "stop", "delta": {"content": "ok"}}]}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
            "model": model, "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"}}]})

    with OpenAI(api_key="test", base_url=base, http_client=httpx.Client(transport=httpx.MockTransport(receive))) as client:
        main = ChatCompletionsTransport().build_kwargs(
            model=model, messages=[{"role": "user", "content": "hi"}],
            provider_profile=profile, base_url=base, temperature=0.7,
            supports_reasoning=True, reasoning_config={"enabled": False},
        )
        aux = _build_call_kwargs(provider, model, [{"role": "user", "content": "hi"}],
            temperature=0.1, base_url=base, reasoning_config={"enabled": False})
        for kwargs in (main, aux):
            client.chat.completions.create(**kwargs)
        list(client.chat.completions.create(**main, stream=True))
        from trajectory_compressor import CompressionConfig, TrajectoryCompressor
        compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
        compressor.config = CompressionConfig(summarization_model=model)
        compressor._use_call_llm = False
        _, summary = compressor._summary_request("Summarize the conversation")
        client.chat.completions.create(**summary)
        if provider == "openrouter":
            scripts = Path(__file__).resolve().parents[2] / "optional-skills/security/godmode/scripts"
            runpy.run_path(str(scripts / "godmode_race.py"))["_query_model"](client, model, main["messages"])
            runpy.run_path(str(scripts / "auto_jailbreak.py"))["_test_query"](client, model, main["messages"])

    async def send_async():
        async with AsyncOpenAI(api_key="test", base_url=base,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(receive))) as client:
            await client.chat.completions.create(**aux)
            stream = await client.chat.completions.create(**aux, stream=True)
            async for _ in stream:
                pass

    asyncio.run(send_async())
    assert len(captured) == (8 if provider == "openrouter" else 6)
    for body in captured:
        assert not {"temperature", "top_p", "top_k"}.intersection(body)
        assert "thinkingBudget" not in json.dumps(body)
        assert "thinking_budget" not in json.dumps(body)
        assert '"max_tokens"' not in json.dumps(body.get("reasoning", {}))
    assert _build_call_kwargs("openrouter", "openai/gpt-4o", [], temperature=0.2)["temperature"] == 0.2
    for obsolete in ({"temperature": 0.1}, {"generationConfig": {"topK": 20}},
            {"reasoning": {"max_tokens": 100}}, {"thinking_config": {"thinkingBudget": 0}}):
        with pytest.raises(ValueError, match="Gemini"):
            _build_call_kwargs(provider, model, [], extra_body=obsolete, base_url=base)
        with pytest.raises(ValueError, match="Gemini"):
            ChatCompletionsTransport().build_kwargs(model=model, messages=[],
                provider_profile=profile, base_url=base, request_overrides={"extra_body": obsolete})


def test_native_gemini_rejects_legacy_body_and_uses_levels(monkeypatch, tmp_path):
    captured = []

    def receive(request):
        captured.append(json.loads(request.content))
        payload = {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}
        if ":streamGenerateContent" in str(request.url):
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(payload)}\n\n")
        return httpx.Response(200, json=payload)

    client = GeminiNativeClient(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(receive)))
    try:
        client.chat.completions.create(model="gemini-3-flash-preview", messages=[{"role": "user", "content": "hi"}],
            extra_body={"thinking_config": {"thinkingLevel": "minimal"}})
        kwargs = {"model": "gemini-3-flash-preview", "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"thinking_config": {"thinkingLevel": "minimal"}}}
        list(client.chat.completions.create(**kwargs, stream=True))

        async def send_async():
            adapter = AsyncGeminiNativeClient(client)
            await adapter.chat.completions.create(**kwargs)
            stream = await adapter.chat.completions.create(**kwargs, stream=True)
            async for _ in stream:
                pass

        asyncio.run(send_async())
        client.chat.completions.create(model="gemma-4-31b-it", messages=[{"role": "user", "content": "hi"}],
            temperature=0.2, top_p=0.8)
        for obsolete in ({"thinking_config": {"thinking_budget": 0}}, {"top_k": 20}):
            with pytest.raises(ValueError, match="Gemini"):
                client.chat.completions.create(model="gemini-3-flash-preview", messages=[], extra_body=obsolete)
        with pytest.raises(ValueError, match="not verified"):
            client.chat.completions.create(model="gemini-2.5-flash", messages=[],
                extra_body={"thinking_config": {"thinkingLevel": "minimal"}})
        monkeypatch.setattr("agent.auxiliary_client._get_auxiliary_task_config", lambda task: {"temperature": 0.1})
        with pytest.raises(ValueError, match="temperature"):
            _build_call_kwargs("gemini", "gemini-3-flash-preview", [], task="vision", temperature=0.1)
        monkeypatch.setattr("agent.auxiliary_client._get_auxiliary_task_config", lambda task: {})
        with pytest.raises(ValueError, match="MoA"):
            _build_call_kwargs("gemini", "gemini-3-flash-preview", [], task="moa_reference", temperature=0.6)
    finally:
        client.close()
    from trajectory_compressor import CompressionConfig
    config_file = tmp_path / "compression.yaml"
    config_file.write_text("summarization:\n  model: google/gemini-3-flash-preview\n  temperature: 0.3\n")
    with pytest.raises(ValueError, match="temperature"):
        CompressionConfig.from_yaml(str(config_file))
    assert len(captured) == 5
    assert captured[-1]["generationConfig"]["temperature"] == 0.2
    assert captured[-1]["generationConfig"]["topP"] == 0.8
    for body in captured[:-1]:
        config = body["generationConfig"]
        assert config["thinkingConfig"]["thinkingLevel"] == "minimal"
        assert not {"temperature", "topP", "topK"}.intersection(config)
        assert "thinkingBudget" not in config["thinkingConfig"]
