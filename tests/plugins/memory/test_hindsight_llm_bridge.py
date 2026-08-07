"""Tests for the Hermes-owned LLM bridge used by Hindsight Embedded."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import MagicMock

import pytest
from openai import OpenAI

import plugins.memory.hindsight as hindsight
from plugins.memory.hindsight import HindsightMemoryProvider
from plugins.memory.hindsight.hermes_llm_bridge import HermesLlmBridge


class _FakeLlm:
    def __init__(self) -> None:
        self.complete_calls: list[tuple[list[dict], dict]] = []
        self.structured_calls: list[dict] = []

    def complete(self, messages, **kwargs):
        self.complete_calls.append((messages, kwargs))
        return SimpleNamespace(
            text="plain response",
            provider="openai-codex",
            model="gpt-5.6-luna",
            usage=SimpleNamespace(input_tokens=7, output_tokens=3, total_tokens=10),
        )

    def complete_structured(self, **kwargs):
        self.structured_calls.append(kwargs)
        return SimpleNamespace(
            text=json.dumps({"facts": []}),
            parsed={"facts": []},
            provider="openai-codex",
            model="gpt-5.6-luna",
            usage=SimpleNamespace(input_tokens=11, output_tokens=5, total_tokens=16),
        )


def _post(url: str, api_key: str, payload: dict, *, authorization: str | None = None):
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": authorization or f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(request, timeout=5) as response:  # noqa: S310
        return response.status, json.loads(response.read())


def _post_raw(url: str, api_key: str, body: bytes):
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(request, timeout=5) as response:  # noqa: S310
        return response.status, json.loads(response.read())


def test_bridge_routes_plain_completion_to_host_facade():
    llm = _FakeLlm()
    bridge = HermesLlmBridge(llm)
    bridge.start()
    try:
        status, response = _post(
            f"{bridge.base_url}/chat/completions",
            bridge.api_key,
            {
                "model": "a-model-hindsight-requested",
                "messages": [
                    {"role": "system", "content": "System instruction"},
                    {"role": "user", "content": "Remember this"},
                ],
            },
        )
    finally:
        bridge.close()

    assert status == 200
    assert response["choices"][0]["message"]["content"] == "plain response"
    assert response["model"] == "gpt-5.6-luna"
    assert response["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }
    assert llm.complete_calls[0][1]["purpose"] == "hindsight"
    assert llm.complete_calls[0][0][1]["content"] == "Remember this"


def test_bridge_forwards_native_tools_and_returns_openai_tool_calls():
    class ToolCallingLlm(_FakeLlm):
        def complete(self, messages, **kwargs):
            self.complete_calls.append((messages, kwargs))
            return SimpleNamespace(
                text="",
                tool_calls=[
                    {
                        "id": "call-reflect-1",
                        "type": "function",
                        "function": {
                            "name": "recall",
                            "arguments": '{"query":"timezone"}',
                        },
                    }
                ],
                provider="openai-codex",
                model="gpt-5.6-sol",
                usage=SimpleNamespace(input_tokens=9, output_tokens=2, total_tokens=11),
            )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": "Search memory",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    llm = ToolCallingLlm()
    bridge = HermesLlmBridge(llm)
    bridge.start()
    try:
        status, response = _post(
            f"{bridge.base_url}/chat/completions",
            bridge.api_key,
            {
                "model": "hermes-inherited",
                "messages": [{"role": "user", "content": "Use memory tools"}],
                "tools": tools,
                "tool_choice": "auto",
            },
        )
    finally:
        bridge.close()

    assert status == 200
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"] == {
        "name": "recall",
        "arguments": '{"query":"timezone"}',
    }
    call_kwargs = llm.complete_calls[0][1]
    assert call_kwargs["tools"] == tools
    assert call_kwargs["tool_choice"] == "auto"


def test_bridge_preserves_tool_call_transcript_for_next_agent_iteration():
    llm = _FakeLlm()
    bridge = HermesLlmBridge(llm)
    bridge.start()
    messages = [
        {"role": "user", "content": "Use memory"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-reflect-1",
                    "type": "function",
                    "function": {
                        "name": "recall",
                        "arguments": '{"query":"timezone"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-reflect-1",
            "content": '{"results":["WIB"]}',
        },
    ]
    try:
        status, _ = _post(
            f"{bridge.base_url}/chat/completions",
            bridge.api_key,
            {
                "model": "hermes-inherited",
                "messages": messages,
                "tools": [],
            },
        )
    finally:
        bridge.close()

    assert status == 200
    forwarded = llm.complete_calls[0][0]
    assert forwarded[1]["tool_calls"] == messages[1]["tool_calls"]
    assert forwarded[1]["content"] is None
    assert forwarded[2]["role"] == "tool"
    assert forwarded[2]["tool_call_id"] == "call-reflect-1"
    assert forwarded[2]["content"] == '{"results":["WIB"]}'


def test_bridge_is_compatible_with_openai_sdk_client():
    llm = _FakeLlm()
    bridge = HermesLlmBridge(llm)
    bridge.start()
    try:
        client = OpenAI(base_url=bridge.base_url, api_key=bridge.api_key)
        response = client.chat.completions.create(
            model="ignored-by-inherit-mode",
            messages=[{"role": "user", "content": "Use the OpenAI wire format"}],
        )
    finally:
        bridge.close()

    assert response.choices[0].message.content == "plain response"
    assert response.usage is not None
    assert response.usage.total_tokens == 10


def test_bridge_routes_json_schema_to_structured_host_facade():
    llm = _FakeLlm()
    bridge = HermesLlmBridge(llm)
    bridge.start()
    schema = {"type": "object", "properties": {"facts": {"type": "array"}}}
    try:
        status, response = _post(
            f"{bridge.base_url}/chat/completions",
            bridge.api_key,
            {
                "model": "ignored-by-inherit-mode",
                "messages": [
                    {"role": "system", "content": "Return facts as JSON"},
                    {"role": "user", "content": "User prefers WIB"},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "facts", "schema": schema, "strict": True},
                },
                "max_completion_tokens": 256,
            },
        )
    finally:
        bridge.close()

    assert status == 200
    assert response["choices"][0]["message"]["content"] == '{"facts": []}'
    call = llm.structured_calls[0]
    assert call["json_schema"] == schema
    assert call["schema_name"] == "facts"
    assert call["max_tokens"] == 256
    assert call["purpose"] == "hindsight"
    assert "User prefers WIB" in call["input"][0].text
    assert call["system_prompt"] == "Return facts as JSON"


def test_bridge_rejects_wrong_token_and_streaming():
    llm = _FakeLlm()
    bridge = HermesLlmBridge(llm)
    bridge.start()
    try:
        with pytest.raises(HTTPError) as unauthorized:
            _post(
                f"{bridge.base_url}/chat/completions",
                bridge.api_key,
                {"messages": [{"role": "user", "content": "x"}]},
                authorization="Bearer wrong-token",
            )
        assert unauthorized.value.code == 401

        with pytest.raises(HTTPError) as streaming:
            _post(
                f"{bridge.base_url}/chat/completions",
                bridge.api_key,
                {
                    "stream": True,
                    "messages": [{"role": "user", "content": "x"}],
                },
            )
        assert streaming.value.code == 400
    finally:
        bridge.close()


def test_bridge_rejects_malformed_json_as_bad_request():
    bridge = HermesLlmBridge(_FakeLlm())
    bridge.start()
    try:
        with pytest.raises(HTTPError) as malformed:
            _post_raw(
                f"{bridge.base_url}/chat/completions",
                bridge.api_key,
                b"{not-json",
            )
        assert malformed.value.code == 400
        body = json.loads(malformed.value.read())
        assert body["error"]["type"] == "invalid_request_error"
    finally:
        bridge.close()


def test_hindsight_provider_wires_hermes_mode_to_loopback_bridge(monkeypatch):
    class FakeEmbedded:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    hindsight_module = ModuleType("hindsight")
    setattr(hindsight_module, "HindsightEmbedded", FakeEmbedded)
    monkeypatch.setitem(sys.modules, "hindsight", hindsight_module)
    monkeypatch.setattr(hindsight, "_check_local_runtime", lambda: (True, None))
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        hindsight,
        "get_secret",
        lambda *args, **kwargs: pytest.fail("hermes inheritance must not read a direct Hindsight secret"),
    )

    host_llm = _FakeLlm()
    context = SimpleNamespace(llm=host_llm)
    provider = HindsightMemoryProvider(context)
    provider._mode = "local_embedded"
    provider._config = {
        "mode": "local_embedded",
        "llm_provider": "hermes",
        "llm_api_key": "must-not-be-used",
        "llm_model": "another-model",
        "profile": "hermes",
        "idle_timeout": 0,
    }

    client = provider._get_client()
    try:
        bridge = provider._llm_bridge
        assert bridge is not None
        assert client.kwargs["llm_provider"] == "openai"
        assert client.kwargs["llm_model"] == "hermes-inherited"
        assert client.kwargs["llm_base_url"].startswith("http://127.0.0.1:")
        assert client.kwargs["llm_api_key"] == bridge.api_key
        assert client.kwargs["llm_api_key"] != "must-not-be-used"
        env_overrides = provider._embedded_llm_env_overrides()
        profile_env = hindsight._build_embedded_profile_env(
            provider._config,
            **env_overrides,
        )
        assert profile_env["HINDSIGHT_API_LLM_PROVIDER"] == "openai"
        assert profile_env["HINDSIGHT_API_LLM_MODEL"] == "hermes-inherited"
        assert profile_env["HINDSIGHT_API_LLM_BASE_URL"] == client.kwargs["llm_base_url"]
        assert profile_env["HINDSIGHT_API_LLM_API_KEY"] == bridge.api_key
    finally:
        provider.shutdown()


def test_hermes_provider_instances_use_isolated_embedded_profiles(monkeypatch):
    class FakeEmbedded:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "hindsight", SimpleNamespace(HindsightEmbedded=FakeEmbedded))
    monkeypatch.setattr(hindsight, "_check_local_runtime", lambda: (True, None))
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *args, **kwargs: None)

    config = {
        "mode": "local_embedded",
        "llm_provider": "hermes",
        "profile": "hermes",
    }
    first = HindsightMemoryProvider(SimpleNamespace(llm=_FakeLlm()))
    second = HindsightMemoryProvider(SimpleNamespace(llm=_FakeLlm()))
    first._mode = second._mode = "local_embedded"
    first._config = dict(config)
    second._config = dict(config)

    try:
        first_client = first._get_client()
        second_client = second._get_client()
        assert first_client.kwargs["profile"] != second_client.kwargs["profile"]
        assert first_client.kwargs["profile"].startswith("hermes-hermes-")
        assert first_client.kwargs["llm_base_url"] != second_client.kwargs["llm_base_url"]
    finally:
        first.shutdown()
        second.shutdown()


def test_register_passes_plugin_context_to_provider():
    context = SimpleNamespace(llm=MagicMock(), register_memory_provider=MagicMock())
    hindsight.register(context)
    provider = context.register_memory_provider.call_args.args[0]
    assert isinstance(provider, HindsightMemoryProvider)
    assert provider._host_context is context


def test_memory_loader_provides_host_owned_llm_to_hindsight(monkeypatch):
    import plugins.memory as memory_loader
    from agent.plugin_llm import PluginLlm

    provider_dir = Path(hindsight.__file__).parent
    monkeypatch.setattr(memory_loader, "find_provider_dir", lambda name: provider_dir)
    host_llm = MagicMock()

    provider = memory_loader.load_memory_provider(
        "hindsight",
        host_context=SimpleNamespace(llm=host_llm),
    )

    assert isinstance(provider, HindsightMemoryProvider)
    assert provider is not None
    assert provider._host_context is not None
    assert getattr(provider._host_context, "llm") is host_llm

    default_provider = memory_loader.load_memory_provider("hindsight")
    assert isinstance(default_provider, HindsightMemoryProvider)
    assert default_provider is not None
    assert default_provider._host_context is not None
    assert isinstance(getattr(default_provider._host_context, "llm"), PluginLlm)


def test_profile_env_materializes_loopback_bridge_without_provider_secret(monkeypatch, tmp_path):
    host_llm = _FakeLlm()
    provider = HindsightMemoryProvider(SimpleNamespace(llm=host_llm))
    provider._config = {"profile": "hermes", "llm_provider": "hermes"}
    bridge = HermesLlmBridge(host_llm)
    bridge.start()
    provider._llm_bridge = bridge
    monkeypatch.setattr(
        hindsight,
        "_embedded_profile_env_path",
        lambda config: tmp_path / "profile.env",
    )
    try:
        path = hindsight._materialize_embedded_profile_env(
            provider._config,
            **provider._embedded_llm_env_overrides(),
        )
        values = hindsight._load_simple_env(path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert values["HINDSIGHT_API_LLM_PROVIDER"] == "openai"
        assert values["HINDSIGHT_API_LLM_BASE_URL"] == bridge.base_url
        assert values["HINDSIGHT_API_LLM_API_KEY"] == bridge.api_key
    finally:
        bridge.close()


def test_hermes_mode_availability_requires_context_and_embedded_runtime(monkeypatch):
    monkeypatch.setattr(
        hindsight,
        "_load_config",
        lambda: {"mode": "local_embedded", "llm_provider": "hermes"},
    )
    monkeypatch.setattr(hindsight, "_check_local_runtime", lambda: (True, None))

    assert HindsightMemoryProvider(SimpleNamespace(llm=MagicMock())).is_available() is True
    assert HindsightMemoryProvider().is_available() is False
