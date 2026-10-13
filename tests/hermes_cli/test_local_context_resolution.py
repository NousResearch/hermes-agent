"""Context-length resolution for the managed llama.cpp router.

The incident: the statusbar showed 131K for a local model the server had
granted 262144 tokens. The router reports ``meta: null`` on /v1/models
for a model that is not currently LOADED (models autoload on first chat,
so at session start the model is routinely unloaded), and /v1/models/{id}
404s — every metadata probe missed, resolution fell through to the
name-pattern defaults, and the "qwen" family catch-all (131072) shipped
as the compressor's budget and the statusbar's denominator.

Contract: a loaded router child answers with its RUNTIME window from
``/props?model=``. An unloaded one is never asked: the router loads
whatever model a request names, so ``/props?model=`` would load it. Its
``/models`` entry lists the ``--ctx-size`` it will launch with, which
answers instead.
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest

import agent.model_metadata as mm


GRANTED = 262144
RUNTIME = 131072  # a loaded child's /props value, distinct from the launch flag
MODEL = "Qwen-Test-UD-Q4_K_M"


@pytest.fixture
def router(request):
    """Stub of llama-server: a router whose child has ``request.param`` status, or
    (None) a single-model server. The router lists the child's status and launch
    args on ``/models`` and ``/v1/models`` (meta=null while unloaded) and answers
    the bare ``/props`` for itself (n_ctx 0); ``/props?model=`` is answered by the
    child, which the real router loads first."""
    status = request.param
    asked: list[str] = []

    class _Router(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            asked.append(self.path)
            if self.path.startswith("/props?model="):
                body = {"default_generation_settings": {"n_ctx": RUNTIME}}
            elif self.path == "/props":  # a single-model server is its own model; a router answers for itself
                body = {"default_generation_settings": {"n_ctx": RUNTIME if status is None else 0}}
            elif self.path in ("/models", "/v1/models"):
                entry = {"id": MODEL, "owned_by": "llamacpp", "meta": None}
                if status is not None:  # router entries carry status; single-model servers list no status
                    entry["status"] = {"value": status, "args": ["llama-server", "--ctx-size", str(GRANTED)]}
                body = {"data": [entry]}
            else:  # /v1/models/{id} -> 404, as the real router answers
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Router)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", asked
    server.shutdown()


@pytest.fixture(autouse=True)
def _llamacpp(monkeypatch):
    monkeypatch.setattr(mm, "detect_local_server_type", lambda *a, **k: "llamacpp")
    monkeypatch.setattr(mm, "_endpoint_blackholed", lambda *a, **k: False)


@pytest.mark.parametrize("router", ["unloaded"], indirect=True)
def test_unloaded_llamacpp_model_resolves_granted_window(router):
    url, asked = router
    ctx = mm._query_local_context_length_uncached(MODEL, url)
    assert ctx == GRANTED, (
        f"resolved {ctx}; an unloaded model must resolve its launch window, "
        "not fall through to name-pattern catch-alls")
    assert not any(p.startswith("/props?model=") for p in asked), "asking /props?model= loads the model"


@pytest.mark.parametrize("router", ["loaded"], indirect=True)
def test_loaded_model_answers_with_its_runtime_window(router):
    url, asked = router
    assert mm._query_local_context_length_uncached(MODEL, url) == RUNTIME
    assert any(p.startswith(f"/props?model={MODEL}") and "autoload=false" in p for p in asked)


def test_non_llamacpp_servers_skip_props(monkeypatch):
    """Ollama/LM Studio/vLLM keep their existing probe order — /props is
    llama.cpp-shaped and must not be consulted for other server types."""
    calls = []

    class _FakeResp:
        status_code = 404

        def json(self):
            return {}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            calls.append(url)
            return _FakeResp()

        def post(self, url, **k):
            calls.append(url)
            return _FakeResp()

    import httpx
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    monkeypatch.setattr(mm, "detect_local_server_type", lambda *a, **k: "vllm")
    monkeypatch.setattr(mm, "_endpoint_blackholed", lambda *a, **k: False)
    mm._query_local_context_length_uncached("m", "http://127.0.0.1:9999/v1")
    assert not any("/props" in u for u in calls)


@pytest.mark.parametrize("router", [None], indirect=True)
def test_single_model_server_answers_bare_props(router):
    url, _asked = router
    assert mm._query_local_context_length_uncached(MODEL, url) == RUNTIME
