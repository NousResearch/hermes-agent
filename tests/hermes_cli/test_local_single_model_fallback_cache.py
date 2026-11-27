"""A single-model llama.cpp server must not poison the context cache.

The incident: a desktop tab pinned to ``Qwen3.8-27B-UD-Q4_K_M`` was repointed at
a single-model fork server (127.0.0.1:18435) serving Ternary weights under a GGUF
*path* id. No requested id ever equals that path, so both id-blind probes — bare
``/props`` and the sole-entry ``/v1/models`` fallback — answered with the
serving engine's 262144 window no matter which id was asked for, and
``_probe_local_context_length`` persisted it under the *requested* id. The disk
ended up with ``Qwen...@18435: 262144`` next to ``Qwen...@18434: 65536``: the
same name pinning two different windows depending on port.

Contract: the live value is still returned (the engine really does serve that
window for any id), but it is persisted only when the live listing actually
names the requested id — exact, publisher slug, or weight-file stem. Foreign
ids resolve live every time (short-TTL memo only) and never touch disk.
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest

import hermes_yaml as yaml

import agent.model_metadata as mm

TERNARY = "Ternary-Bonsai-2-27B-PQ2_0"
QWEN = "Qwen3.8-27B-UD-Q4_K_M"
WINDOWS_GGUF_ID = "C:\\Models\\Ternary-Bonsai-2-27B-PQ2_0.gguf"
GRANTED = 262144
ROUTER_GRANTED = 65536


def _serve(handler_cls):
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


@pytest.fixture
def fork():
    """Stub of the manual fork server: ONE model listed under its GGUF path
    (backslashes, exactly as llama-server reports a Windows path id); bare
    /props answers name-blind; /v1/models/{id} 404s."""

    class _Fork(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/v1/models":
                body = {"data": [{
                    "id": WINDOWS_GGUF_ID,
                    "owned_by": "llamacpp",
                    "meta": {"n_ctx": GRANTED, "n_ctx_train": GRANTED},
                }]}
            elif self.path.startswith("/props") or self.path.startswith("/v1/props"):
                body = {
                    "default_generation_settings": {"n_ctx": GRANTED},
                    "model_alias": WINDOWS_GGUF_ID,
                }
            else:  # /v1/models/{id} -> 404, as the fork answers unknown ids
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

    yield from _serve(_Fork)


@pytest.fixture
def router():
    """Stub of the managed router with the Qwen child loaded at its own window."""

    class _Router(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/v1/models":
                body = {"data": [{
                    "id": QWEN,
                    "owned_by": "llamacpp",
                    "meta": {"n_ctx": ROUTER_GRANTED},
                    "status": {"value": "loaded"},
                }]}
            elif self.path.startswith("/props") or self.path.startswith("/v1/props"):
                body = {
                    "default_generation_settings": {"n_ctx": ROUTER_GRANTED},
                    "model_alias": QWEN,
                }
            else:
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

    yield from _serve(_Router)


def _uncached(monkeypatch, tmp_path):
    """Isolate the persistent cache to tmp and force llamacpp probing."""
    cache_file = tmp_path / "context_length_cache.yaml"
    monkeypatch.setattr(mm, "_get_context_cache_path", lambda: cache_file)
    monkeypatch.setattr(mm, "detect_local_server_type", lambda *a, **k: "llamacpp")
    monkeypatch.setattr(mm, "_endpoint_blackholed", lambda *a, **k: False)
    mm._LOCAL_CTX_PROBE_CACHE.clear()
    mm._endpoint_model_metadata_cache.clear()
    return cache_file


def _cached_lengths(cache_file):
    try:
        return yaml.safe_load(cache_file.read_text(encoding="utf-8")).get("context_lengths", {})
    except OSError:
        return {}


def test_stem_named_request_persists(fork, tmp_path, monkeypatch):
    """The serving model's own id (weight-file stem) keeps today's behavior:
    live window returned AND cached."""
    cache_file = _uncached(monkeypatch, tmp_path)
    ctx = mm._probe_local_context_length(TERNARY, fork, api_key="", provider="custom")
    assert ctx == GRANTED
    assert _cached_lengths(cache_file).get(f"{TERNARY}@{fork.rstrip('/')}") == GRANTED


def test_foreign_id_resolves_live_but_never_persists(fork, tmp_path, monkeypatch):
    """A stale tab's id gets the engine's real window for THIS call (the engine
    serves any id) but must not pin it on disk under the foreign name."""
    cache_file = _uncached(monkeypatch, tmp_path)
    ctx = mm._probe_local_context_length(QWEN, fork, api_key="", provider="custom")
    assert ctx == GRANTED
    assert f"{QWEN}@{fork.rstrip('/')}" not in _cached_lengths(cache_file)


def test_matched_router_child_still_persists(router, tmp_path, monkeypatch):
    """No regression for the name-aware path: the managed router's own window
    for its listed id keeps persisting."""
    cache_file = _uncached(monkeypatch, tmp_path)
    ctx = mm._probe_local_context_length(QWEN, router, api_key="", provider="custom")
    assert ctx == ROUTER_GRANTED
    assert _cached_lengths(cache_file).get(f"{QWEN}@{router.rstrip('/')}") == ROUTER_GRANTED


def test_stem_matching_helpers():
    assert mm._listed_id_names_model(WINDOWS_GGUF_ID, TERNARY)
    assert mm._listed_id_names_model("org/" + TERNARY, TERNARY)
    assert mm._listed_id_names_model(TERNARY, TERNARY)
    assert not mm._listed_id_names_model(WINDOWS_GGUF_ID, QWEN)
    assert not mm._listed_id_names_model("", TERNARY)
    assert not mm._listed_id_names_model(WINDOWS_GGUF_ID, "")
