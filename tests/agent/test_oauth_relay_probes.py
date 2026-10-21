"""No metadata probe reaches an Anthropic OAuth relay in a shape it cannot answer.

A provider entry declaring ``capabilities.anthropic_oauth_proxy`` is an Anthropic Messages relay
behind its own key (a pooled-subscription relay, say). Its models live at ``<base>/v1/models`` and answer
only an authenticated request carrying ``anthropic-version``. Every other probe the metadata layer
knows (OpenAI ``<base>/models``, Ollama ``/api/show`` and ``/api/tags``, the llama.cpp / vLLM /
LM Studio waterfall) fails there — as a 401 when the caller had no key (the auxiliary pricing path
passes none), as a 404 or 400 otherwise — on every new process, because nothing caches a failure.

The gym below runs every probe entry point against one relay stub, on a remote host (reached as an
HTTP proxy, so no local-server branch runs) and on a loopback host (where the local probes run),
with and without the key, and checks every request the relay saw.
"""

import json
import threading
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

import hermes_yaml as yaml
from agent import model_metadata as mm

KEY = "relay-test-key"
MODEL = "relay-only-model"  # in no catalog: a window for it can only come from the relay
RELAY_WINDOW = 777_777
REMOTE = "http://relay.example.test:3443"


class _Relay(BaseHTTPRequestHandler):
    """Answers like a real relay: 401 without the key, /v1/models needs anthropic-version, else 404."""

    protocol_version = "HTTP/1.1"
    seen: list

    def log_message(self, format, *args):
        pass

    def _reply(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle(self):
        if n := int(self.headers.get("Content-Length") or 0):
            self.rfile.read(n)
        bearer, x_key = self.headers.get("Authorization", ""), self.headers.get("x-api-key", "")
        keyed = bearer == f"Bearer {KEY}" or x_key == KEY
        path = urlsplit(self.path).path
        self.seen.append({"method": self.command, "path": path, "keyed": keyed,
                          "version": self.headers.get("anthropic-version", "")})
        if not keyed:
            return self._reply(401, {"type": "error", "error": {"type": "authentication_error"}})
        if self.command == "GET" and path == "/v1/models":
            if not self.headers.get("anthropic-version"):
                return self._reply(400, {"type": "error", "error": {"type": "invalid_request_error"}})
            return self._reply(200, {"data": [{"type": "model", "id": MODEL, "display_name": MODEL,
                                               "max_input_tokens": RELAY_WINDOW, "max_tokens": 64_000}],
                                     "has_more": False})
        return self._reply(404, {"type": "error", "error": {"type": "not_found_error"}})

    do_GET = do_POST = _handle


@pytest.fixture
def relay_server():
    seen: list = []
    handler = type("Relay", (_Relay,), {"seen": seen})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server, seen
    finally:
        server.shutdown()
        server.server_close()


def _clear_probe_memos():
    for cache in (mm._endpoint_model_metadata_cache, mm._endpoint_model_metadata_cache_time,
                  mm._endpoint_probe_path_cache, mm._endpoint_blackhole_cache, mm._LOCAL_CTX_PROBE_CACHE):
        cache.clear()
    from hermes_cli import models
    models._probe_neg_cache.clear()


def _route(tmp_path, monkeypatch, relay_server, where, *, oauth_proxy=True):
    """Configure the relay as provider ``relay`` and return its base URL."""
    server, _seen = relay_server
    port = server.server_address[1]
    if where == "remote":
        base_url = REMOTE
        proxy = f"http://127.0.0.1:{port}"
        for name in ("HTTP_PROXY", "http_proxy"):
            monkeypatch.setenv(name, proxy)
        for name in ("NO_PROXY", "no_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.delenv(name, raising=False)
    else:
        base_url = f"http://127.0.0.1:{port}"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TEST_RELAY_KEY", KEY)
    entry = {"api": base_url, "key_env": "TEST_RELAY_KEY", "transport": "anthropic_messages",
             "models": {MODEL: {}}}
    if oauth_proxy:
        entry["capabilities"] = {"anthropic_oauth_proxy": True}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "relay", "default": MODEL, "base_url": base_url},
        "providers": {"relay": entry},
    }), encoding="utf-8")
    _clear_probe_memos()
    return base_url


def _pricing(base_url, key):
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
    # The auxiliary accounting call: no api_key, whatever the route holds (aux_accounting).
    return estimate_usage_cost(MODEL, CanonicalUsage(input_tokens=10, output_tokens=5),
                               provider="custom", base_url=base_url)


ENTRY_POINTS = {
    "context/named-provider": lambda url, key: mm.get_model_context_length(MODEL, url, api_key=key, provider="relay"),
    "context/custom": lambda url, key: mm.get_model_context_length(MODEL, url, api_key=key, provider="custom"),
    "context/no-provider": lambda url, key: mm.get_model_context_length(MODEL, url, api_key=key),
    "endpoint-metadata": lambda url, key: mm.fetch_endpoint_model_metadata(url, api_key=key),
    "aux-pricing": _pricing,
    "ollama-api-show": lambda url, key: mm._query_ollama_api_show(MODEL, url, api_key=key),
    "ollama-num-ctx": lambda url, key: mm.query_ollama_num_ctx(MODEL, url, api_key=key),
    "ollama-vision": lambda url, key: mm.query_ollama_supports_vision(MODEL, url, api_key=key),
    "server-type": lambda url, key: mm.detect_local_server_type(url, api_key=key),
    "local-context": lambda url, key: mm._query_local_context_length(MODEL, url, api_key=key),
}


@pytest.mark.parametrize("key", [KEY, ""], ids=["keyed", "keyless"])
@pytest.mark.parametrize("where", ["remote", "loopback"])
@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
def test_relay_sees_only_authenticated_anthropic_model_listing(entry_point, where, key, tmp_path, monkeypatch, relay_server):
    base_url = _route(tmp_path, monkeypatch, relay_server, where)
    ENTRY_POINTS[entry_point](base_url, key)
    seen = relay_server[1]
    wrong = [r for r in seen if (r["method"], r["path"]) != ("GET", "/v1/models") or not r["keyed"] or not r["version"]]
    assert wrong == [], f"{entry_point} sent the relay requests it cannot answer: {wrong}"


@pytest.mark.parametrize("where", ["remote", "loopback"])
def test_relay_window_comes_from_its_anthropic_model_listing(where, tmp_path, monkeypatch, relay_server):
    base_url = _route(tmp_path, monkeypatch, relay_server, where)
    assert mm.get_model_context_length(MODEL, base_url, api_key=KEY, provider="relay") == RELAY_WINDOW
    assert [(r["path"], r["keyed"]) for r in relay_server[1]] == [("/v1/models", True)]


def test_keyless_relay_lookup_sends_nothing_and_falls_back(tmp_path, monkeypatch, relay_server):
    base_url = _route(tmp_path, monkeypatch, relay_server, "remote")
    assert mm.get_model_context_length(MODEL, base_url, api_key="", provider="relay") == mm.DEFAULT_FALLBACK_CONTEXT
    assert relay_server[1] == []


@pytest.mark.parametrize("where", ["remote", "loopback"])
def test_picker_lists_relay_models_without_a_root_probe(where, tmp_path, monkeypatch, relay_server):
    from hermes_cli.models import probe_api_models
    base_url = _route(tmp_path, monkeypatch, relay_server, where)
    result = probe_api_models(KEY, base_url, api_mode="anthropic_messages")
    assert result["models"] == [MODEL]
    assert [r["path"] for r in relay_server[1]] == ["/v1/models"]


def test_endpoint_without_the_capability_keeps_its_probes(tmp_path, monkeypatch, relay_server):
    """The relay rule is opt-in: an undeclared endpoint (an Ollama or OpenAI-compatible server
    behind the same URL shape) still gets the OpenAI and Ollama probes."""
    base_url = _route(tmp_path, monkeypatch, relay_server, "remote", oauth_proxy=False)
    mm.get_model_context_length(MODEL, base_url, api_key=KEY, provider="relay")
    paths = {r["path"] for r in relay_server[1]}
    assert {"/models", "/api/show"} <= paths


def test_relay_declared_on_one_model_counts_for_the_endpoint(tmp_path, monkeypatch, relay_server):
    base_url = _route(tmp_path, monkeypatch, relay_server, "remote", oauth_proxy=False)
    config = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    config["providers"]["relay"]["models"] = {MODEL: {"anthropic_oauth_proxy": True}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    mm.get_model_context_length("another-model", base_url, api_key=KEY, provider="relay")
    assert [r["path"] for r in relay_server[1]] == ["/v1/models"]


def test_relay_pricing_still_resolves_without_metadata(tmp_path, monkeypatch, relay_server):
    base_url = _route(tmp_path, monkeypatch, relay_server, "remote")
    cost = _pricing(base_url, KEY)
    assert cost.amount_usd is None or isinstance(cost.amount_usd, Decimal)
