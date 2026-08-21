"""Production-path regressions for the bundled Freemaxxing router."""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

os.environ.setdefault("FREEMAXXING_PORT", "0")
import providers as provider_registry  # noqa: E402
from providers import get_provider_profile  # noqa: E402

for module_name in list(sys.modules):
    if module_name.startswith("plugins.model_providers.freemaxxing"):
        sys.modules.pop(module_name, None)
provider_registry._REGISTRY.clear()
provider_registry._ALIASES.clear()
provider_registry._PROVIDER_LIST_CACHE = None
provider_registry._discovered = False
provider_registry._user_plugins_dir = lambda: None

PROFILE = get_provider_profile("freemaxxing")
PLUGIN = importlib.import_module("plugins.model_providers.freemaxxing")
PROXY = importlib.import_module("plugins.model_providers.freemaxxing.proxy")
UPSTREAM = importlib.import_module("plugins.model_providers.freemaxxing.upstream")
HANDLER = importlib.import_module("plugins.model_providers.freemaxxing.handler")
Backend, pool = PROXY.Backend, PROXY.pool
TOKEN = "test-freemaxxing-capability"


class Mock:
    def __init__(
        self,
        *,
        models=None,
        status=200,
        raw=None,
        retry_after=None,
        stream=None,
        close=False,
    ):
        self.models = models or ["test-model"]
        self.status = status
        self.raw = raw
        self.retry_after = retry_after
        self.stream = stream
        self.close = close
        self.count = 0
        self.body = None
        self.auth = None
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                return

            def do_GET(self):  # noqa: N802
                outer.auth = self.headers.get("Authorization")
                data = json.dumps(
                    {"data": [{"id": model} for model in outer.models]}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):  # noqa: N802
                outer.count += 1
                outer.auth = self.headers.get("Authorization")
                length = int(self.headers.get("Content-Length", "0"))
                raw_request = self.rfile.read(length)
                outer.body = json.loads(raw_request) if raw_request else None
                if outer.close:
                    self.close_connection = True
                    return
                if outer.stream is not None:
                    self.send_response(200)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for chunk in outer.stream:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    self.close_connection = True
                    return
                self.send_response(outer.status)
                if outer.status == 429 and outer.retry_after is not None:
                    self.send_header("Retry-After", outer.retry_after)
                data = outer.raw
                if data is None:
                    data = json.dumps(
                        {
                            "id": "ok",
                            "choices": [],
                            "object": "chat.completion",
                        }
                    ).encode()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@pytest.fixture(autouse=True)
def clean():
    PROXY.stop_proxy()
    PLUGIN._listener = None
    pool.clear()
    yield
    PROXY.stop_proxy()
    PLUGIN._listener = None
    pool.clear()


def start(initializer=None):
    server = PROXY.spawn_proxy(
        port=0,
        token=TOKEN,
        pool_initializer=initializer,
    )
    return int(server.server_address[1])


def request(port, path, *, token=TOKEN, method="GET", body=None):
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    return urllib.request.urlopen(req, timeout=10)


def post(port, *, model="test-model", stream=False, extra=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if stream:
        body["stream"] = True
    if extra:
        body.update(extra)
    return request(port, "/v1/chat/completions", method="POST", body=body)


def add(mock, *, name="backend", tier=0, key="key"):
    backend = Backend(name, mock.url, api_key=key, tier=tier)
    pool.add(backend)
    return backend


def test_canonical_module_identity():
    assert PROFILE is not None and PROFILE.name == "freemaxxing"
    assert PLUGIN.pool is PROXY.pool
    assert PROXY is importlib.import_module(
        "plugins.model_providers.freemaxxing.proxy"
    )


def test_listener_capability_is_required_and_immutable():
    with pytest.raises(ValueError):
        PROXY.spawn_proxy(port=0, token="")
    first = PROXY.spawn_proxy(port=0, token=TOKEN)
    assert PROXY.spawn_proxy(port=0, token=TOKEN) is first
    with pytest.raises(RuntimeError):
        PROXY.spawn_proxy(port=0, token="different")


def test_public_liveness_is_minimal_but_runtime_surfaces_require_auth():
    port = start()
    with request(port, "/v1/healthz", token=None) as response:
        assert json.loads(response.read()) == {
            "service": "freemaxxing",
            "status": "ok",
        }
    for path in ("/healthz", "/v1/models"):
        with pytest.raises(urllib.error.HTTPError) as caught:
            request(port, path, token=None)
        assert caught.value.code == 401


def test_initializer_failure_is_fail_closed():
    port = start(lambda: (_ for _ in ()).throw(RuntimeError("multiplex")))
    with pytest.raises(urllib.error.HTTPError) as caught:
        request(port, "/v1/models")
    assert caught.value.code == 409
    assert pool.count() == 0


def test_empty_pool_is_503():
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(start())
    assert caught.value.code == 503


@pytest.mark.parametrize(
    "status,error_class",
    [(429, "rate_limit"), (503, "transient")],
)
def test_retryable_http_errors_fail_over(status, error_class):
    first = Mock(status=status, retry_after="1")
    second = Mock()
    backend = add(first, name="first")
    add(second, name="second")
    try:
        with post(start()) as response:
            assert json.loads(response.read())["id"] == "ok"
        assert first.count == second.count == 1
        assert backend.last_error_class == error_class
    finally:
        first.stop()
        second.stop()


@pytest.mark.parametrize("raw", [b"{bad", b"\xff\xfe", b"[]"])
def test_invalid_success_bodies_fail_over(raw):
    first, second = Mock(raw=raw), Mock()
    backend = add(first, name="first")
    add(second, name="second")
    try:
        with post(start()) as response:
            assert json.loads(response.read())["id"] == "ok"
        assert backend.last_error_class == "transient"
    finally:
        first.stop()
        second.stop()


def test_interrupted_success_body_fails_over():
    first, second = Mock(close=True), Mock()
    backend = add(first, name="first")
    add(second, name="second")
    try:
        with post(start()) as response:
            response.read()
        assert backend.last_error_class == "transient"
        assert second.count == 1
    finally:
        first.stop()
        second.stop()


def test_oversized_success_body_fails_over(monkeypatch):
    monkeypatch.setattr(UPSTREAM, "_MAX_RESPONSE_BODY_BYTES", 16)
    first = Mock(raw=b'{"x":"' + b"a" * 100 + b'"}')
    second = Mock(raw=b'{"id":"ok"}')
    backend = add(first, name="first")
    add(second, name="second")
    try:
        with post(start()) as response:
            response.read()
        assert backend.last_error_class == "transient"
    finally:
        first.stop()
        second.stop()


def test_non_model_400_does_not_retry_elsewhere():
    first, second = Mock(status=400), Mock()
    add(first, name="first")
    add(second, name="second")
    try:
        with pytest.raises(urllib.error.HTTPError) as caught:
            post(start())
        assert caught.value.code == 400 and second.count == 0
    finally:
        first.stop()
        second.stop()


def test_auth_refresh_is_serialized_and_retries_same_backend():
    upstream = Mock(status=401)
    calls = 0

    def refresh():
        nonlocal calls
        calls += 1
        upstream.status = 200
        return upstream.url, "new-key"

    pool.add(Backend("rotating", upstream.url, api_key="old", refresh=refresh))
    try:
        with post(start()) as response:
            response.read()
        assert calls == 1 and upstream.count == 2
        assert upstream.auth == "Bearer new-key"
    finally:
        upstream.stop()


def test_tier_precedence_and_exclusion():
    backends = [
        Backend("a", "http://127.0.0.1", api_key="x", tier=0),
        Backend("b", "http://127.0.0.1", api_key="x", tier=1),
        Backend("c", "http://127.0.0.1", api_key="x", tier=0),
    ]
    for backend in backends:
        pool.add(backend)
    first = pool.next("freemaxxing")
    second = pool.next("freemaxxing", exclude={first.name})
    third = pool.next("freemaxxing", exclude={"a", "c"})
    assert first.tier == second.tier == 0 and first is not second
    assert third.tier == 1


def test_cost_policy_is_backend_specific_and_fail_closed():
    openrouter = Backend("openrouter", "http://127.0.0.1", api_key="x")
    nous = Backend("nous-portal", "http://127.0.0.1", api_key="x")
    assert PROXY._accept_catalog_id(openrouter, "qwen/qwen3:free")
    assert not PROXY._accept_catalog_id(openrouter, "qwen/qwen3")
    assert PROXY._accept_catalog_id(nous, "deepseek/deepseek-v4-flash-0731")
    assert not PROXY._accept_catalog_id(nous, "x-ai/grok-4.6")


def test_router_alias_substitutes_only_admitted_free_model():
    upstream = Mock(models=["paid/model", "qwen/qwen3:free"])
    add(upstream, name="openrouter", tier=1, key="or-key")
    try:
        with post(start(), model="freemaxxing") as response:
            response.read()
        assert upstream.body["model"] == "qwen/qwen3:free"
    finally:
        upstream.stop()


def test_request_bound_applies_before_pool_use(monkeypatch):
    monkeypatch.setattr(HANDLER, "_MAX_REQUEST_BODY_BYTES", 16)
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(start(), extra={"padding": "x" * 100})
    assert caught.value.code == 413 and pool.count() == 0


def test_streaming_passthrough_and_precommit_failover():
    first = Mock(status=503)
    second = Mock(stream=[b"data: ok\n\n", b"data: [DONE]\n\n"])
    backend = add(first, name="first")
    add(second, name="second")
    try:
        with post(start(), stream=True) as response:
            raw = response.read()
        assert b"ok" in raw and backend.last_error_class == "transient"
    finally:
        first.stop()
        second.stop()


@pytest.mark.parametrize(
    "chunks",
    [[], [b"data: " + b"x" * 100 + b"\n\n"]],
)
def test_invalid_first_stream_line_fails_over(monkeypatch, chunks):
    monkeypatch.setattr(HANDLER, "_MAX_SSE_LINE_BYTES", 16)
    first = Mock(stream=chunks)
    second = Mock(stream=[b"data: ok\n\n", b"data: [DONE]\n\n"])
    backend = add(first, name="first")
    add(second, name="second")
    try:
        with post(start(), stream=True) as response:
            assert b"ok" in response.read()
        assert backend.last_error_class == "stream_interrupted"
    finally:
        first.stop()
        second.stop()


def test_downstream_cancellation_does_not_penalize_upstream(monkeypatch):
    backend = Backend("healthy", "http://127.0.0.1", api_key="x")
    pool.add(backend)

    class Stream:
        chunks = [b"data: hello\n\n", b"data: [DONE]\n\n"]

        def readline(self, _limit):
            return self.chunks.pop(0) if self.chunks else b""

        def close(self):
            return

    class Writer:
        def write(self, _chunk):
            raise BrokenPipeError

        def flush(self):
            return

    monkeypatch.setattr(HANDLER, "_open_stream", lambda *_args: Stream())
    handler = object.__new__(PROXY.ChatCompletionsHandler)
    handler.wfile = Writer()
    handler.send_response = handler.send_header = lambda *_args: None
    handler.end_headers = lambda: None
    handler.close_connection = False
    handler._handle_streaming({"model": "test-model"}, "test-model")
    assert backend.last_error_class is None and backend.is_available()


@pytest.mark.parametrize(
    "raw,expected",
    [("nan", 30.0), ("inf", 30.0), ("-10", 0.0), ("9999", 300.0)],
)
def test_retry_after_is_finite_and_bounded(raw, expected):
    assert PROXY._parse_retry_after({"Retry-After": raw}) == expected
