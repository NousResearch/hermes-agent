"""LongCat's declared route must work through discovery and tool-call replay."""

import copy
import json

import httpx
from openai import OpenAI


def _seed_catalog(tmp_path, model):
    entry = {
        "id": model,
        "name": model,
        "tool_call": True,
        "reasoning": True,
        "modalities": {"input": ["text"], "output": ["text"]},
        "limit": {"context": 777_777, "output": 12_345},
        "cost": {"input": 0.42, "output": 1.23, "cache_read": 0.01},
    }
    (tmp_path / "models_dev_cache.json").write_text(
        json.dumps({"longcat": {"id": "longcat", "models": {model: entry}}}),
        encoding="utf-8",
    )
    return entry


def _agent(profile, model, key):
    from run_agent import AIAgent

    return AIAgent(
        model=model, provider=profile.name, base_url=profile.base_url, api_key=key,
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        enabled_toolsets=[], save_trajectories=False,
    )


def test_discovered_longcat_route_reaches_chat_and_catalog_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from providers import get_provider_profile

    profile = get_provider_profile("longcat")
    assert profile is not None, "The bundled LongCat plugin must be discoverable"
    model = profile.fallback_models[0]
    entry = _seed_catalog(tmp_path, model)

    from hermes_cli.models import parse_model_input, provider_model_ids
    from hermes_cli.provider_catalog import provider_catalog_by_slug
    from hermes_cli.runtime_provider import resolve_runtime_provider

    # A fresh offline install can select a model before entering a key.
    assert provider_model_ids(profile.name) == list(profile.fallback_models)
    assert parse_model_input(f"{profile.name}:{model}", "custom") == (profile.name, model)
    descriptor = provider_catalog_by_slug()[profile.name]
    assert descriptor.api_key_env_vars == profile.env_vars
    assert descriptor.signup_url == profile.signup_url

    (tmp_path / ".env").write_text(f"{profile.env_vars[0]}=test-longcat-key\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        f"model:\n  provider: {profile.name}\n  default: {model}\n", encoding="utf-8",
    )
    resolved = resolve_runtime_provider(requested=profile.name)
    assert resolved["base_url"] == profile.base_url
    assert resolved["api_mode"] == profile.api_mode

    from agent import models_dev
    from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT, get_model_context_length
    from agent.auxiliary_client import _build_call_kwargs

    # Use the actual disk-cache path and preserve changing catalog values.
    monkeypatch.setattr(models_dev, "_models_dev_cache", {})
    monkeypatch.setattr(models_dev, "_models_dev_cache_time", 0)
    info = models_dev.get_model_info(profile.name, model)
    assert info is not None
    assert info.context_window == entry["limit"]["context"]
    assert info.cost_input == entry["cost"]["input"]
    assert info.cost_cache_read == entry["cost"]["cache_read"]
    assert info.supports_vision() is False
    assert get_model_context_length(model, profile.base_url, provider=profile.name) == info.context_window

    agent = _agent(profile, model, resolved["api_key"])
    received = []

    def handle(request):
        received.append(json.loads(request.content))
        assert str(request.url) == profile.base_url + "/chat/completions"
        assert request.headers["Authorization"] == "Bearer " + resolved["api_key"]
        return httpx.Response(200, json={
            "id": "chat-test", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Ready"}, "finish_reason": "stop"}],
        })

    with OpenAI(api_key=resolved["api_key"], base_url=resolved["base_url"],
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        for config, expected in ((None, "enabled"), ({"enabled": False}, "disabled"),
                                 ({"effort": "none"}, "disabled"), ({"effort": "high"}, "enabled")):
            agent.reasoning_config = config
            kwargs = agent._build_api_kwargs([{"role": "user", "content": "Hello"}], tools_for_api=[])
            response = client.chat.completions.create(**kwargs)
            assert response.choices[0].message.content == "Ready"
            assert received[-1]["thinking"] == {"type": expected}
            assert "reasoning_effort" not in received[-1]
            assert "reasoning" not in received[-1]
            aux = _build_call_kwargs(profile.name, model, [{"role": "user", "content": "Summarize"}],
                                     reasoning_config=config, base_url=profile.base_url)
            assert aux["extra_body"]["thinking"] == received[-1]["thinking"]
            assert "reasoning_effort" not in aux
            assert "reasoning" not in aux["extra_body"]

    # A fresh offline catalog must still avoid the generic unknown-model window.
    (tmp_path / "models_dev_cache.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(models_dev, "_models_dev_cache", {})
    monkeypatch.setattr(models_dev, "_models_dev_cache_time", 0)
    assert get_model_context_length(model, profile.base_url, provider=profile.name) != DEFAULT_FALLBACK_CONTEXT


def test_longcat_tool_round_preserves_reasoning_and_strict_fallback_strips_it(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from providers import get_provider_profile

    profile = get_provider_profile("longcat")
    assert profile is not None, "The bundled LongCat plugin must be discoverable"
    model = profile.fallback_models[0]
    _seed_catalog(tmp_path, model)
    agent = _agent(profile, model, "test-longcat-key")
    reasoning = "I need the tool result before answering."
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    tools = [{"type": "function", "function": {
        "name": "calculate", "description": "Calculate a sum", "parameters": {
            "type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"],
        },
    }}]
    received = []

    def handle(request):
        body = json.loads(request.content)
        received.append(body)
        assert body["stream"] is True
        assert body["stream_options"]["include_usage"] is True
        if len(received) == 1:
            deltas = [
                {"role": "assistant", "content": "", "reasoning_content": ""},
                {"reasoning_content": reasoning},
                {"tool_calls": [{"index": 0, "id": "call_sum", "type": "function", "function": {
                    "name": "calculate", "arguments": '{"expression":',
                }}]},
                {"tool_calls": [{"index": 0, "id": None, "function": {"name": None, "arguments": '"2+2"}'}}]},
            ]
            finish = "tool_calls"
        else:
            replay = body["messages"][1]
            assert replay["reasoning_content"] == reasoning
            assert replay["content"] == ""
            assert replay["tool_calls"][0]["id"] == body["messages"][2]["tool_call_id"]
            assert replay["tool_calls"][0]["function"] == {"name": "calculate", "arguments": '{"expression":"2+2"}'}
            deltas, finish = [{"role": "assistant", "content": "4"}], "stop"
        chunks = [{
            "id": "chat-tool-test", "object": "chat.completion.chunk", "created": 0, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        } for delta in deltas]
        chunks.append({**chunks[0], "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        chunks.append({**chunks[0], "choices": [], "usage": {
            "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 50},
        }})
        sse = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=sse)

    with OpenAI(api_key=agent.api_key, base_url=agent.base_url,
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kwargs: client)
        monkeypatch.setattr(agent, "_close_request_openai_client", lambda *args, **kwargs: None)
        first = agent._interruptible_streaming_api_call(agent._build_api_kwargs(messages, tools_for_api=tools))
        assistant = agent._build_assistant_message(first.choices[0].message, first.choices[0].finish_reason)
        assert first.usage.prompt_tokens_details.cached_tokens == 50
        # Resumed sessions can contain null from non-streaming tool responses.
        assistant["content"] = None
        original = copy.deepcopy(assistant)
        replay = {k: v for k, v in assistant.items() if k != "reasoning_content"}
        agent._copy_reasoning_content_for_api(assistant, replay)
        messages += [replay, {"role": "tool", "tool_call_id": "call_sum", "content": "4"}]
        second = agent._interruptible_streaming_api_call(agent._build_api_kwargs(messages, tools_for_api=tools))
        assert second.choices[0].message.content == "4"
        assert assistant == original, "Wire normalization must preserve cached conversation history"

    from agent.message_sanitization import stale_thinking_reaches_wire

    assert stale_thinking_reaches_wire(agent.api_mode, agent.provider, agent.model, agent.base_url)
    agent.provider, agent.base_url = "custom", "https://strict.example/v1"
    assert agent._reapply_reasoning_echo_for_provider(messages) == 1
    assert "reasoning_content" not in messages[1]
    assert not stale_thinking_reaches_wire(agent.api_mode, agent.provider, agent.model, agent.base_url)

    # The same replay capability works for another plugin, without name checks.
    import providers
    from providers.base import ProviderProfile

    other = ProviderProfile(name="test-echo", requires_reasoning_echo=True)
    monkeypatch.setitem(providers._REGISTRY, other.name, other)
    agent.provider = other.name
    agent._copy_reasoning_content_for_api(assistant, messages[1])
    assert messages[1]["reasoning_content"] == reasoning
    assert stale_thinking_reaches_wire(agent.api_mode, agent.provider, agent.model, agent.base_url)
