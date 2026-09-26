"""Private Anthropic reasoning must follow the selected auxiliary transport."""
import asyncio
import json

import httpx
import pytest

from agent import auxiliary_client as aux


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("base_url", ["https://api.minimax.io/v1", "https://relay.example/v1",
                                      "https://relay.example/anthropic"])
def test_public_minimax_chat_route_accepts_reasoning(monkeypatch, base_url, async_mode):
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {
        "provider": "minimax", "api_mode": "chat_completions"})
    requests = []

    def response(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, request=request, json={
            "id": "fixture", "object": "chat.completion", "created": 0,
            "model": "MiniMax-M3", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "Fixture title"},
                "finish_reason": "stop"}],
        })

    monkeypatch.setattr(httpx.Client, "send", lambda self, request, **kw: response(request))

    async def async_send(self, request, **kw):
        return response(request)

    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    options = dict(task="title_generation", provider="minimax", model="MiniMax-M3",
                   base_url=base_url, api_key="fixture-only",
                   messages=[{"role": "user", "content": "Make a title"}],
                   reasoning_config={"enabled": False})
    error = None
    try:
        result = asyncio.run(aux.async_call_llm(**options)) if async_mode else aux.call_llm(**options)
    except TypeError as exc:
        error = exc
    assert error is None, f"Plain OpenAI SDK rejected auxiliary request: {error}"
    assert result.choices[0].message.content == "Fixture title"
    assert len(requests) == 1
    assert "_reasoning_config" not in requests[0]


def _messages_client(async_mode=False):
    from types import SimpleNamespace

    captured = []

    class Messages:
        def create(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Fixture title")],
                stop_reason="end_turn", usage=None,
            )

    client = aux.AnthropicAuxiliaryClient(
        SimpleNamespace(messages=Messages()), "claude-sonnet-4-6", "fixture-only",
        "https://relay.example/v1",
    )
    return (aux.AsyncAnthropicAuxiliaryClient(client) if async_mode else client), captured


@pytest.mark.parametrize("async_mode", [False, True])
def test_messages_adapter_preserves_explicit_reasoning_on_unmarked_url(monkeypatch, async_mode):
    client, captured = _messages_client(async_mode)
    monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **kw: (client, "claude-sonnet-4-6"))
    options = dict(task="title_generation", provider="custom", model="claude-sonnet-4-6",
                   base_url=client.base_url, api_key="fixture-only",
                   messages=[{"role": "user", "content": "Title"}],
                   reasoning_config={"enabled": False},
                   extra_body={"reasoning": {"enabled": True, "effort": "high"}})
    result = asyncio.run(aux.async_call_llm(**options)) if async_mode else aux.call_llm(**options)
    assert result.choices[0].message.content == "Fixture title"
    assert captured[0]["thinking"] == {"type": "disabled"}
    assert options["reasoning_config"] == {"enabled": False}
    assert options["extra_body"]["reasoning"]["enabled"] is True


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("reasoning", [None, {}, {"enabled": False}])
def test_builder_private_control_owned_by_client(async_mode, reasoning):
    client, _ = _messages_client(async_mode)
    for destination in [client, object(), None]:
        kwargs = aux._build_call_kwargs(
            "minimax", "MiniMax-M3", [{"role": "user", "content": "Title"}],
            reasoning_config=reasoning, base_url="https://api.minimax.io/anthropic",
            client=destination,
        )
        assert ("_reasoning_config" in kwargs) == (destination is client and bool(reasoning))


@pytest.mark.parametrize("async_mode", [False, True])
def test_retry_and_fallback_rebuild_use_new_client(monkeypatch, async_mode):
    from types import SimpleNamespace

    messages_client, _ = _messages_client(async_mode)
    plain_client = SimpleNamespace(base_url="https://api.minimax.io/v1")
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {})
    request = dict(messages=[{"role": "user", "content": "Title"}], tools=None,
                   temperature=None, max_tokens=32, effective_extra_body={},
                   reasoning_config={"enabled": False})
    for initial, rebuilt in [(messages_client, plain_client), (plain_client, messages_client)]:
        _, first, rebuild = aux._plan_fallback_candidate(
            initial, "MiniMax-M3", "minimax", task="title_generation",
            effective_timeout=12, apply_fast_lane=False, **request,
        )
        _, second = rebuild("minimax", rebuilt, "MiniMax-M3")
        assert ("_reasoning_config" in first) == (initial is messages_client)
        assert ("_reasoning_config" in second) == (rebuilt is messages_client)
        assert second["timeout"] == 12
        monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **kw: (rebuilt, "MiniMax-M3"))
        actual, retry = aux._prepare_same_provider_retry(
            task="title_generation", resolved_provider="minimax", resolved_model="MiniMax-M3",
            resolved_base_url=rebuilt.base_url, resolved_api_key="fixture-only",
            resolved_api_mode="chat_completions", main_runtime={}, final_model="MiniMax-M3",
            effective_timeout=12, async_mode=async_mode, **request,
        )
        assert actual is rebuilt
        assert ("_reasoning_config" in retry) == (rebuilt is messages_client)


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
def test_local_configured_route_reaches_http_after_fallback(tmp_path, monkeypatch, mode):
    """Real config + resolver + SDK + fallback transport, with only local HTTP."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            if payload["model"] == "unavailable":
                status = 402
                body = json.dumps({"error": {"message": "insufficient credits"}}).encode()
                content_type = "application/json"
            else:
                status = 200
                reply = {"id": "fixture", "created": 0, "model": payload["model"],
                         "object": "chat.completion", "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": "Local title"},
                         "finish_reason": "stop"}]}
                if payload.get("stream"):
                    reply["object"] = "chat.completion.chunk"
                    reply["choices"][0]["delta"] = reply["choices"][0].pop("message")
                    body = f"data: {json.dumps(reply)}\n\ndata: [DONE]\n\n".encode()
                    content_type = "text/event-stream"
                else:
                    body, content_type = json.dumps(reply).encode(), "application/json"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    config = {"auxiliary": {"title_generation": {
        "provider": "minimax", "model": "MiniMax-M3" if mode == "stream" else "unavailable",
        "base_url": base, "api_key": "fixture-only", "api_mode": "chat_completions",
        "fallback_chain": [{"provider": "minimax", "model": "MiniMax-M3", "base_url": base,
                            "api_key": "fixture-only", "api_mode": "chat_completions"}],
    }}}
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    (tmp_path / "config.yaml").write_text(json.dumps(config), encoding="utf8")
    aux.shutdown_cached_clients()
    aux._reset_aux_unhealthy_cache()
    try:
        options = dict(task="title_generation", messages=[{"role": "user", "content": "Title"}],
                       reasoning_config={"enabled": False})
        if mode == "async":
            result = asyncio.run(aux.async_call_llm(**options))
        elif mode == "stream":
            chunks = list(aux.call_llm(**options, stream=True))
            assert chunks[0].choices[0].delta.content == "Local title"
        else:
            result = aux.call_llm(**options)
        if mode != "stream":
            assert result.choices[0].message.content == "Local title"
            assert [p["model"] for _, p in requests] == ["unavailable", "MiniMax-M3"]
        assert all(path == "/v1/chat/completions" for path, _ in requests)
        assert all("_reasoning_config" not in payload for _, payload in requests)
    finally:
        aux.shutdown_cached_clients()
        aux._reset_aux_unhealthy_cache()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
