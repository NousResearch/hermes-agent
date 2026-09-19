"""``local_runtime.detect_ports`` must reach the detector from the provider-resolution
entry points, and the default probe-port contract must not move.

Regression: an externally run llama-server on a non-default port (``llama-server --port
8081``) plus ``local_runtime.detect_ports: [8081]`` in config left the ``llamacpp`` alias
invisible — both ``providers._llamacpp_pdef()`` and the runtime path
(``runtime_provider_custom._resolve_llamacpp_runtime()``) called the resolver without a
config, so only the hardcoded ``DEFAULT_PROBE_PORTS`` were probed.

A stub ``/props`` server stands in for llama-server; sharing one config file with a
per-test HERMES_HOME keeps the ``load_config()`` cache out of the picture.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml


class _LlamaPropsStub(BaseHTTPRequestHandler):
    """Enough llama-server to be fingerprinted: real ``build_info`` on /props, /models for
    router mode. Nothing else is served."""

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/props":
            body = {"build_info": "b10520-8e1a2c3d4",
                    "model_path": "stub-model.gguf",
                    "default_generation_settings": {"n_ctx": 4096}}
        elif path == "/models":
            body = {"data": [{"id": "stub-model", "owned_by": "llamacpp"}]}
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

    def log_message(self, *args):  # silence
        pass


@pytest.fixture
def llama_port():
    """Port of a stub llama-server answering /props + /models (no GPU, no real binary)."""
    server = HTTPServer(("127.0.0.1", 0), _LlamaPropsStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """A HERMES_HOME the resolver can read config.yaml from (per-test, so a cached or
    on-disk config cannot leak in from another test)."""
    home = tmp_path / "detect-ports-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_config(home, config) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _only_configured_ports(monkeypatch) -> None:
    """Empty the hardcoded default ports so the CONFIGURED port is the only possible hit
    (also keeps a real server on the dev machine's 8080 out of the assertion)."""
    monkeypatch.setattr("hermes_cli.local_runtime.detect.DEFAULT_PROBE_PORTS", ())


# ── the reported bug: detect_ports must be honored ───────────────────────────


def test_llamacpp_pdef_probes_configured_detect_ports(config_home, llama_port, monkeypatch):
    """Provider visibility (desktop model picker / `hermes model`): llama-server on a
    non-default port + detect_ports in config => a real ProviderDef, not None."""
    _only_configured_ports(monkeypatch)
    _write_config(config_home, {"local_runtime": {"detect_ports": [llama_port]}})

    from hermes_cli.providers import _llamacpp_pdef

    pdef = _llamacpp_pdef()
    assert pdef is not None, "llamacpp provider invisible although detect_ports names the live port"
    assert pdef.base_url == f"http://127.0.0.1:{llama_port}/v1"
    assert pdef.source == "local-runtime"


def test_resolve_provider_full_llamacpp_probes_configured_detect_ports(config_home, llama_port, monkeypatch):
    """The issue's repro, through the public chain: resolve_provider_full('llamacpp')."""
    _only_configured_ports(monkeypatch)
    _write_config(config_home, {"local_runtime": {"detect_ports": [llama_port]}})

    from hermes_cli.providers import resolve_provider_full

    pdef = resolve_provider_full("llamacpp")
    assert pdef is not None
    assert pdef.base_url == f"http://127.0.0.1:{llama_port}/v1"


def test_llamacpp_runtime_probes_configured_detect_ports(config_home, llama_port, monkeypatch):
    """The second call site (runtime path): `hermes chat --provider llamacpp` must land on
    the detected server instead of the "local model server is turned off" error."""
    _only_configured_ports(monkeypatch)
    _write_config(config_home, {"local_runtime": {"detect_ports": [llama_port]}})

    from hermes_cli.runtime_provider import _resolve_named_custom_runtime

    runtime = _resolve_named_custom_runtime(requested_provider="llamacpp")
    assert runtime is not None, "runtime path refused a llama-server the config points at"
    assert runtime["source"] == "local-runtime"
    assert runtime["base_url"] == f"http://127.0.0.1:{llama_port}/v1"


# ── protection: behavior that must NOT change ────────────────────────────────


def test_default_probe_port_still_resolves_without_configured_ports(config_home, llama_port, monkeypatch):
    """No detect_ports configured: the hardcoded default port is still probed."""
    _only_configured_ports(monkeypatch)
    monkeypatch.setattr("hermes_cli.local_runtime.detect.DEFAULT_PROBE_PORTS", (llama_port,))
    _write_config(config_home, {"local_runtime": {"enabled": True}})

    from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint

    resolved = resolve_llamacpp_endpoint(wait_for_boot_s=0)
    assert resolved == {"base_url": f"http://127.0.0.1:{llama_port}/v1", "api_key": ""}


def test_no_detect_ports_configured_adds_no_extra_ports(config_home, llama_port, monkeypatch):
    """An empty/absent detect_ports must not turn detection into a port scan: with the
    defaults emptied and nothing configured, resolution stays None."""
    _only_configured_ports(monkeypatch)
    _write_config(config_home, {"local_runtime": {"detect_ports": []}})

    from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint

    assert resolve_llamacpp_endpoint(wait_for_boot_s=0) is None


def test_explicit_config_still_overrides_disk_config(config_home, llama_port, monkeypatch):
    """An explicitly passed config (even an empty one) is authoritative — only a None
    config may fall back to loading the active one, else callers that intentionally pass
    ``{}`` would silently pick up on-disk detect_ports."""
    _only_configured_ports(monkeypatch)
    _write_config(config_home, {"local_runtime": {"detect_ports": [llama_port]}})

    from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint

    assert resolve_llamacpp_endpoint(config={}, wait_for_boot_s=0) is None
    assert resolve_llamacpp_endpoint(config={"local_runtime": {"detect_ports": []}},
                                     wait_for_boot_s=0) is None
