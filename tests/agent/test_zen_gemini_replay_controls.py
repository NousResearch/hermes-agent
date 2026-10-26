"""Zen native replay and sibling wire controls at the HTTP boundary."""

import json

import httpx
import pytest

from agent import auxiliary_client as aux
from hermes_cli.runtime_provider import resolve_runtime_provider
from run_agent import AIAgent


@pytest.mark.parametrize("provider,root", [("opencode-zen", "/zen"), ("opencode-go", "/zen/go")])
@pytest.mark.parametrize("model,wire", [("glm-5.3", "chat"), ("gpt-5.4", "responses"), ("claude-sonnet-4-6", "messages")])
@pytest.mark.parametrize("surface", ["main", "auto_aux"])
def test_non_gemini_models_keep_their_http_wire(monkeypatch, provider, root, model, wire, surface):
    requests = []
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "test-key")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key")

    def send(_client, request, **_kwargs):
        requests.append(request)
        if wire == "responses":
            payload = {"id": "resp-contract", "object": "response", "created_at": 0, "model": model,
                       "status": "completed", "output": [], "parallel_tool_calls": True,
                       "tool_choice": "auto", "tools": []}
        elif wire == "messages":
            payload = {"id": "msg-contract", "type": "message", "role": "assistant", "model": model,
                       "content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn",
                       "usage": {"input_tokens": 1, "output_tokens": 1}}
        else:
            payload = {"id": "chat-contract", "object": "chat.completion", "model": model,
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                                    "finish_reason": "stop"}]}
        return httpx.Response(200, request=request, json=payload)

    monkeypatch.setattr(httpx.Client, "send", send)
    agent = None
    if surface == "main":
        rt = resolve_runtime_provider(requested=provider, target_model=model)
        agent = AIAgent(**{k: rt[k] for k in ("provider", "api_mode", "api_key", "base_url")}, model=model,
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
        client = agent._anthropic_client if wire == "messages" else agent.client
    else:
        client, resolved = aux.resolve_provider_client(
            "auto", model=model, raw_codex=True,
            main_runtime={"provider": provider, "model": "gemini-4-argon", "api_key": "test-key",
                          "base_url": f"https://opencode.ai{root}/v1", "api_mode": "chat_completions"},
        )
        assert resolved == model
        if wire in {"messages", "responses"}:
            client = client._real_client
    try:
        if wire == "responses":
            client.responses.create(model=model, input="OK")
        elif wire == "messages":
            client.messages.create(model=model, messages=[{"role": "user", "content": "OK"}], max_tokens=64)
        else:
            client.chat.completions.create(model=model, messages=[{"role": "user", "content": "OK"}])
        expected = {"responses": "responses", "messages": "messages", "chat": "chat/completions"}[wire]
        assert requests[-1].url.path == f"{root}/v1/{expected}"
        assert requests[-1].headers.get("authorization") == "Bearer test-key" or requests[-1].headers.get("x-api-key") == "test-key"
        assert "x-goog-api-key" not in requests[-1].headers
        assert json.loads(requests[-1].content)["model"] == model
    finally:
        if agent:
            agent.close()
        else:
            client.close()


def test_native_tool_call_signature_replays_through_main_storage(monkeypatch):
    requests = []
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "test-key")

    def send(_client, request, **_kwargs):
        requests.append(request)
        parts = ([{"thought": True, "text": "Need a tool."},
                  {"functionCall": {"id": "native-call", "name": "terminal", "args": {"command": "pwd"}},
                   "thoughtSignature": "opaque-signed-tool"}]
                 if len(requests) == 1 else [{"text": "OK"}])
        return httpx.Response(200, request=request,
                              json={"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}]})

    monkeypatch.setattr(httpx.Client, "send", send)
    rt = resolve_runtime_provider(requested="opencode-zen", target_model="gemini-4-argon")
    agent = AIAgent(**{k: rt[k] for k in ("provider", "api_mode", "api_key", "base_url")}, model="gemini-4-argon",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    history = [{"role": "user", "content": "Run pwd"}]
    tools = [{"type": "function", "function": {"name": "terminal", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
    try:
        response = agent.client.chat.completions.create(**agent._build_api_kwargs(history, tools))
        choice = response.choices[0]
        stored = agent._build_assistant_message(choice.message, choice.finish_reason)
        assert stored["reasoning"] == "Need a tool."
        history.extend([stored, {"role": "tool", "tool_call_id": stored["tool_calls"][0]["id"], "content": "C:/project"}])
        result = agent.client.chat.completions.create(**agent._build_api_kwargs(history, tools))
        assert result.choices[0].message.content == "OK"
        contents = json.loads(requests[-1].content)["contents"]
        replayed = next(p for c in contents for p in c["parts"] if "functionCall" in p)
        assert replayed["thoughtSignature"] == "opaque-signed-tool"
        assert replayed["functionCall"]["args"] == {"command": "pwd"}
        output = next(p for c in contents for p in c["parts"] if "functionResponse" in p)
        assert output["functionResponse"]["name"] == "terminal"
    finally:
        agent.close()
