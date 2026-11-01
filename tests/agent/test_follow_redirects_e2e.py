"""Real config -> agent -> SDK/adapter -> redirect -> parsed response tests.

OpenAI and Anthropic use loopback HTTP servers. Gemini keeps the genuine Google
HTTPS URL and replaces only the wire transport, not routing or client builders.
"""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
from threading import Thread

import httpx
import pytest

from run_agent import AIAgent


MODEL = "redirect-test-model"
KEY = "redirect-test-secret"
MESSAGES = [{"role": "user", "content": "hello redirect"}]


def _payload(protocol):
    if protocol == "anthropic":
        return {
            "id": "msg_redirect", "type": "message", "role": "assistant",
            "model": MODEL, "content": [{"type": "text", "text": "redirect works"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
    if protocol == "gemini":
        return {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "redirect works"}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
        }
    return {
        "id": "chatcmpl_redirect", "object": "chat.completion", "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "redirect works"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    # Fail closed if initialization or a redirect ever tries the public network.
    connect = socket.socket.connect

    def local_connect(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1"):
            raise AssertionError(f"External network forbidden in redirect E2E: {address}")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_connect)
    try:
        yield tmp_path
    finally:
        from agent.process_bootstrap import close_shared_transports
        close_shared_transports()


def _config(home, providers):
    # JSON is YAML; use stdlib serialization without depending on a test-only parser.
    (home / "config.yaml").write_text(
        json.dumps({"providers": providers, "model": {"context_length": 128000}}), encoding="utf-8",
    )


@pytest.fixture
def origins():
    servers = []

    def start(protocol="openai", status=307):
        records, received = [], []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append((self.command, self.path))
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                records.append({"method": self.command, "path": self.path, "body": body,
                                "headers": dict(self.headers)})
                canonical = self.path == "/canonical"
                data = json.dumps(_payload(protocol) if canonical else {"error": "redirect"}).encode()
                self.send_response(200 if canonical else status)
                if not canonical:
                    self.send_header("Location", self.server.location)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                received.append((self.command, self.path))
                # Real init-time metadata discovery is unrelated to inference
                # redirects; answer it locally without redirecting or replaying it.
                data = b'{"data": []}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        server.records = records
        server.received = received
        server.url = f"http://127.0.0.1:{server.server_port}"
        server.location = server.url + "/canonical"
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        servers.append((server, thread))
        thread.start()
        return server

    try:
        yield start
    finally:
        for server, thread in servers:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            assert not thread.is_alive()


@contextmanager
def _agent(server, protocol="openai", *, provider=None, requested_provider=None, key=KEY):
    from hermes_cli.config import load_config_readonly
    config = load_config_readonly()
    config["model"] = {"default": MODEL, "provider": provider or protocol,
                       "base_url": server.url + "/v1", "context_length": 128000}
    server_home = Path(os.environ["HERMES_HOME"])
    (server_home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    agent = AIAgent(
        api_key=key, base_url=server.url + "/v1", provider=provider or protocol,
        requested_provider=requested_provider, model=MODEL,
        api_mode="anthropic_messages" if protocol == "anthropic" else "chat_completions",
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        enabled_toolsets=[], disabled_toolsets=["all"],
    )
    try:
        yield agent
    finally:
        native = getattr(agent, "_anthropic_client", None)
        try:
            if native is not None:
                native.close()
        finally:
            agent.close()


def _send(agent, protocol):
    if protocol == "anthropic":
        client = agent._anthropic_client
        client.max_retries = 0
        client.timeout = 2.0
        result = client.messages.create(model=MODEL, messages=MESSAGES, max_tokens=16)
        assert result.content[0].text == "redirect works"
        assert result.usage.output_tokens == 2
    else:
        agent.client.max_retries = 0
        agent.client.timeout = 2.0
        result = agent.client.chat.completions.create(model=MODEL, messages=MESSAGES, max_tokens=16, timeout=2.0)
        assert result.choices[0].message.content == "redirect works"
        assert result.usage.completion_tokens == 2


def _assert_replay(records, protocol, *, bearer=False):
    assert len(records) == 2
    first, canonical = records
    assert first["method"] == canonical["method"] == "POST"
    assert canonical["path"] == "/canonical"
    assert first["body"] == canonical["body"]
    body = json.loads(first["body"])
    if protocol == "gemini":
        assert body["contents"][0]["parts"][0]["text"] == "hello redirect"
    else:
        assert body["model"] == MODEL
        assert body["messages"] == MESSAGES
    headers = [{k.lower(): v for k, v in record["headers"].items()} for record in records]
    credential = "x-goog-api-key" if protocol == "gemini" else (
        "x-api-key" if protocol == "anthropic" and not bearer else "authorization")
    expected = KEY if credential != "authorization" else f"Bearer {KEY}"
    assert headers[0][credential] == headers[1][credential] == expected
    assert headers[0]["content-type"] == headers[1]["content-type"] == "application/json"


@pytest.mark.parametrize("status", [307, 308])
@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_real_sdk_same_origin_redirect(isolated_home, origins, protocol, status):
    _config(isolated_home, {protocol: {"follow_redirects": True}})
    server = origins(protocol, status)
    with _agent(server, protocol) as agent:
        _send(agent, protocol)
    _assert_replay(server.records, protocol)
    assert server.records[0]["path"] == ("/v1/messages" if protocol == "anthropic" else "/v1/chat/completions")


@pytest.mark.parametrize("protocol,policy", [("openai", None), ("anthropic", False)])
def test_default_openai_and_explicit_false_anthropic_do_not_follow(isolated_home, origins, protocol, policy):
    entry = {} if policy is None else {"follow_redirects": policy}
    _config(isolated_home, {protocol: entry, "unrelated": {"follow_redirects": True}})
    server = origins(protocol)
    with _agent(server, protocol) as agent:
        sdk = __import__(protocol)
        with pytest.raises(sdk.APIStatusError) as error:
            _send(agent, protocol)
        assert error.value.status_code == 307
    assert len(server.records) == 1
    assert server.records[0]["method"] == "POST"


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("enabled", [False, True])
def test_model_override_controls_actual_request(isolated_home, origins, protocol, enabled):
    _config(isolated_home, {protocol: {"follow_redirects": not enabled,
                                       "models": {MODEL: {"follow_redirects": enabled}}}})
    server = origins(protocol, 308)
    with _agent(server, protocol) as agent:
        if enabled:
            _send(agent, protocol)
        else:
            with pytest.raises(__import__(protocol).APIStatusError) as error:
                _send(agent, protocol)
            assert error.value.status_code == 308
    if enabled:
        _assert_replay(server.records, protocol)
    else:
        assert len(server.records) == 1


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_named_custom_identity_reaches_real_client(isolated_home, origins, protocol):
    server = origins(protocol)
    _config(isolated_home, {"private-proxy": {"name": "Private Proxy", "base_url": server.url + "/v1",
                                             "follow_redirects": True}})
    with _agent(server, protocol, provider="custom", requested_provider="custom:private-proxy") as agent:
        _send(agent, protocol)
    _assert_replay(server.records, protocol)


@pytest.mark.parametrize("status", [307, 308])
@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("change", ["host", "port"])
def test_cross_origin_never_receives_request(isolated_home, origins, protocol, status, change):
    _config(isolated_home, {protocol: {"follow_redirects": True}})
    source, destination = origins(protocol, status), origins(protocol)
    if change == "port":
        source.location = destination.url + "/canonical"
    else:
        # Same port, different hostname: an unguarded request would reach source again.
        source.location = source.url.replace("127.0.0.1", "localhost") + "/canonical"
    calls = []

    def token():
        calls.append(True)
        return KEY

    # Exercise the real per-request bearer hook, which can reattach credentials
    # AFTER httpx strips Authorization on a cross-origin redirect.
    key = token if protocol == "anthropic" else KEY
    with _agent(source, protocol, key=key) as agent:
        with pytest.raises((__import__(protocol).APIConnectionError, httpx.HTTPError)) as error:
            _send(agent, protocol)
        cause = error.value.__cause__ or error.value
        assert isinstance(cause, httpx.LocalProtocolError)
        assert "Cross-origin" in str(cause)
    assert len(source.records) == 1
    assert destination.records == destination.received == []
    assert ("GET", "/canonical") not in source.received
    assert ("POST", "/canonical") not in source.received
    headers = {k.lower(): v for k, v in source.records[0]["headers"].items()}
    assert headers["authorization"] == f"Bearer {KEY}"
    if protocol == "anthropic":
        assert calls
        assert "x-api-key" not in headers


@pytest.mark.parametrize("rebuild", [False, True])
def test_anthropic_bearer_hook_and_rebuild_preserve_policy(isolated_home, origins, rebuild):
    _config(isolated_home, {"anthropic": {"follow_redirects": True}})
    server = origins("anthropic", 308)
    with _agent(server, "anthropic", key=lambda: KEY) as agent:
        if rebuild:
            agent._anthropic_client.close()
            agent._rebuild_anthropic_client()
        _send(agent, "anthropic")
    _assert_replay(server.records, "anthropic", bearer=True)


@pytest.mark.parametrize("status", [307, 308])
@pytest.mark.parametrize("policy", [None, False, True])
@pytest.mark.parametrize("destination", ["same", "host", "port"])
def test_native_gemini_real_creation_and_request(isolated_home, monkeypatch, status, policy, destination):
    from agent.gemini_native_adapter import GeminiAPIError, GeminiNativeClient

    records = []
    google = "https://generativelanguage.googleapis.com"
    target = {"same": google, "host": "https://other.googleapis.com",
              "port": google + ":444"}[destination]

    def wire(transport, request):
        records.append({"method": request.method, "path": request.url.path,
                        "body": request.read(), "headers": dict(request.headers)})
        assert str(request.url).startswith(google + "/"), "Cross-origin request reached Gemini transport"
        if request.url.path == "/canonical":
            return httpx.Response(200, json=_payload("gemini"), request=request)
        return httpx.Response(status, headers={"Location": target + "/canonical"},
                              json={"error": {"message": "redirect"}}, request=request)

    # Replace only I/O on the actual keepalive transport: config resolution,
    # native detection, httpx redirect logic, payload conversion and parsing run.
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", wire)
    _config(isolated_home, {"gemini": {} if policy is None else {"follow_redirects": policy}})
    endpoint = type("Endpoint", (), {"url": google})()
    with _agent(endpoint, provider="gemini") as agent:
        assert isinstance(agent.client, GeminiNativeClient)
        if policy is True and destination == "same":
            _send(agent, "gemini")
        elif policy is True:
            with pytest.raises(httpx.LocalProtocolError):
                _send(agent, "gemini")
        else:
            with pytest.raises(GeminiAPIError) as error:
                _send(agent, "gemini")
            assert error.value.status_code == status
    if policy is True and destination == "same":
        _assert_replay(records, "gemini")
    else:
        assert len(records) == 1
    assert records[0]["path"].endswith(f"/models/{MODEL}:generateContent")
