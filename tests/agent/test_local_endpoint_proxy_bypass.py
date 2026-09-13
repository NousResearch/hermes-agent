"""#101803 — a local/LAN Ollama endpoint must not be routed through an env proxy.

Reported shape: every chat completion to a local or LAN Ollama failed *instantly* with
``[Errno 65] No route to host`` while ``curl`` to the same endpoint succeeded at the same
moment. The request was being sent to the proxy host instead of the Ollama box, so the
failure was a connect error to an (often unreachable) proxy, not an Ollama error — hence
instant, while curl (which understands the CIDR NO_PROXY entries the reporter had) went
direct.

These tests drive the real client factory (``build_keepalive_http_client``) against a mock
Ollama plus a trap proxy, and assert both directions:
  * local endpoint + NO_PROXY CIDR  -> the mock Ollama is hit, the proxy is never used
  * remote endpoint + the same env  -> the proxy IS used (unchanged behaviour)
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent import process_bootstrap
from agent.process_bootstrap import build_keepalive_http_client

PROXY_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
              "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy")
HITS = {"ollama": [], "proxy": []}


class _Recorder(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: pooled connections persist across calls
    bucket = ""

    def log_message(self, *_args):
        pass

    def _reply(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        HITS[self.bucket].append(self.path)
        body = json.dumps({"via": self.bucket, "choices": [{"message": {"content": "ok"}}],
                           "usage": {}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _reply


def _recorder(bucket):
    handler = type(f"_{bucket}Recorder", (_Recorder,), {"bucket": bucket})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def servers():
    ollama, proxy = _recorder("ollama"), _recorder("proxy")
    HITS["ollama"].clear()
    HITS["proxy"].clear()
    try:
        yield f"http://127.0.0.1:{ollama.server_address[1]}", f"http://127.0.0.1:{proxy.server_address[1]}"
    finally:
        for server in (ollama, proxy):
            server.shutdown()
            server.server_close()
        HITS["ollama"].clear()
        HITS["proxy"].clear()


@pytest.fixture(autouse=True)
def clean_proxy_env(monkeypatch):
    for name in PROXY_KEYS:
        monkeypatch.delenv(name, raising=False)
    process_bootstrap.close_shared_transports()
    yield
    process_bootstrap.close_shared_transports()


def _post(base_url, timeout=5.0):
    client = build_keepalive_http_client(base_url)
    try:
        response = client.post(f"{base_url}/v1/chat/completions",
                               json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
                               timeout=timeout)
        return response.json()
    finally:
        client.close()


def test_local_ollama_bypasses_env_proxy_with_cidr_no_proxy(servers, monkeypatch):
    ollama, proxy = servers
    monkeypatch.setenv("HTTP_PROXY", proxy)
    monkeypatch.setenv("NO_PROXY", "127.0.0.0/8")

    assert _post(ollama)["via"] == "ollama"
    assert HITS["ollama"], "local Ollama endpoint was not reached directly"
    assert not HITS["proxy"], f"local request was routed through the proxy: {HITS['proxy']}"


def test_lan_ollama_bypasses_env_proxy_with_cidr_no_proxy(servers, monkeypatch):
    """The LAN half of #101803: decide by policy, the fake box need not be reachable."""
    _, proxy = servers
    monkeypatch.setenv("HTTP_PROXY", proxy)
    monkeypatch.setenv("NO_PROXY", "192.168.0.0/16")

    assert process_bootstrap._get_proxy_for_base_url("http://192.168.1.50:11434/v1") is None


def test_remote_endpoint_still_uses_env_proxy(servers, monkeypatch):
    _, proxy = servers
    monkeypatch.setenv("HTTP_PROXY", proxy)
    monkeypatch.setenv("NO_PROXY", "127.0.0.0/8")

    # Unreachable directly — a 200 here can only have come from the proxy.
    assert _post("http://remote-provider.example:8080")["via"] == "proxy"
    assert HITS["proxy"], "remote request no longer goes through the env proxy"


def test_local_ollama_without_no_proxy_still_uses_env_proxy(servers, monkeypatch):
    """Unchanged, curl-compatible behaviour: no NO_PROXY entry means the proxy IS used."""
    ollama, proxy = servers
    monkeypatch.setenv("HTTP_PROXY", proxy)

    assert _post(ollama)["via"] == "proxy"
    assert HITS["proxy"] and not HITS["ollama"]
