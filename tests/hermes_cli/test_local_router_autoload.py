"""Status reads never load a model.

llama-server's router (b11370, ``--models-autoload``) loads an unloaded model for any request that
names it, /props, /slots and /metrics included. Clicking Eject on the local models page unloaded the
model, and a capability read 76 ms later loaded it straight back. A per-request ``autoload=false``
makes the router answer 400 "model is not loaded" instead (checked against a router serving a fake
model: zero spawns with it, a spawn per request without it).
"""

import io
import urllib.error
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import capabilities, endpoint, load_progress
from hermes_cli.local_runtime.supervisor import LlamaServerSupervisor


def test_child_query_opts_out_of_autoload():
    assert endpoint.child_query("org/model q4") == "model=org/model%20q4&autoload=false"


def test_capability_read_does_not_load_the_model(monkeypatch):
    seen = []
    monkeypatch.setattr(endpoint, "managed_root", lambda: ("http://127.0.0.1:1", "k"))
    monkeypatch.setattr(endpoint, "managed_get_json", lambda base, key, path, timeout_s: seen.append(path) or {})
    capabilities._props_modalities("qwen")
    assert seen == ["/props?model=qwen&autoload=false"]


def test_prefill_read_does_not_load_the_model(monkeypatch):
    seen = []
    monkeypatch.setattr(load_progress, "_endpoint", lambda: ("http://127.0.0.1:1", "k"))
    monkeypatch.setattr(endpoint, "managed_get_json", lambda base, key, path, timeout_s: seen.append(path) or [])
    assert load_progress.get_prefill_progress("qwen") is None
    assert seen == ["/slots?model=qwen&autoload=false"]


class _Client:
    """httpx-shaped client over a router whose model is unloaded."""

    def __init__(self):
        self.paths = []

    def get(self, url):
        path = url.split("127.0.0.1:1", 1)[1]
        self.paths.append(path)
        if path == "/models":
            return SimpleNamespace(status_code=200, json=lambda: {"data": [{"id": "qwen", "status": {
                "value": "unloaded", "args": ["llama-server.exe", "--ctx-size", "65536", "--jinja"]}}]})
        return SimpleNamespace(status_code=400, json=lambda: {"error": {"message": "model is not loaded"}})


def test_context_probe_reads_launch_args_for_an_unloaded_model():
    from agent.model_metadata_llamacpp import llamacpp_context

    client = _Client()
    assert llamacpp_context(client, "http://127.0.0.1:1", "qwen") == 65536
    assert "/props?model=qwen&autoload=false" in client.paths
    assert "/props?model=qwen" not in client.paths


@pytest.mark.parametrize("body", [["not", "an", "object"], {"data": "oops"}, {"data": [{"id": "qwen", "status": "loading"}]}])
def test_context_probe_survives_an_odd_models_body(body):
    from agent.model_metadata_llamacpp import router_launch_ctx

    client = SimpleNamespace(get=lambda url: SimpleNamespace(status_code=200, json=lambda: body))
    assert router_launch_ctx(client, "http://127.0.0.1:1", "qwen") is None


def test_an_unloaded_model_counts_as_idle(monkeypatch):
    """Growth and the idle sweeper ask whether a model is busy; one that isn't loaded is not."""
    sup = LlamaServerSupervisor.__new__(LlamaServerSupervisor)

    def request(route, body=None, timeout_s=30):
        raise urllib.error.HTTPError(route, 400, "model is not loaded", {}, io.BytesIO(b""))

    monkeypatch.setattr(sup, "_request", request, raising=False)
    assert sup._probe_idle("qwen") is True
