"""#145452 on its own config: a ``providers.<id>`` route resolves to ``provider: custom``, and the
``custom``-keyed catalog paths (titling's fast model, the ``custom`` catalog, the picker's bare
custom row) must probe it with the entry's own key, never the ambient ``OPENAI_API_KEY``.

Real config.yaml under a temp ``HERMES_HOME``, real runtime resolution, a loopback ``/v1/models``
server recording ``Authorization``.
"""

from __future__ import annotations

import http.server
import json
import threading
from unittest.mock import patch

import hermes_yaml as yaml
import pytest

from hermes_cli import models as models_mod
from hermes_cli import models_pricing

ROUTE_KEY = "route-key-secret"
OPENAI_KEY = "sk-proj-ambient-openai"


@pytest.fixture
def catalog_server():
    seen: list[str | None] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            body = json.dumps({"data": [{"id": "proxy-small-mini", "pricing": {
                "prompt": "0.000001", "completion": "0.000002"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1", seen
    server.shutdown()
    server.server_close()


@pytest.fixture
def issue_config(tmp_path, monkeypatch, catalog_server):
    """The issue's config: ``model.provider: litellm`` on the entry's own URL, key in ``key_env``."""
    url, seen = catalog_server
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "my-combo", "provider": "litellm", "base_url": url, "key_env": "HERMES_LITELLM_KEY"},
        "providers": {"litellm": {"name": "my litellm", "base_url": url, "key_env": "HERMES_LITELLM_KEY",
                                  "models": {"my-combo": {"context_length": 500000}}}},
        "auxiliary": {"title_generation": {"prefer_fast_model": True}},
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_LITELLM_KEY", ROUTE_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    for var in ("OPENAI_BASE_URL", "CUSTOM_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    models_pricing._pricing_cache.clear()
    models_pricing._pricing_cache_retry_after.clear()
    yield url, seen
    models_pricing._pricing_cache.clear()
    models_pricing._pricing_cache_retry_after.clear()


def test_titling_fast_model_reads_the_declared_entrys_catalog(issue_config):
    """The runtime is ``custom`` with ``requested_provider: litellm``; titling must read the
    owning entry's catalog with its key instead of a ``custom`` lookup that never probes."""
    from agent.auxiliary_client import _main_route_target
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _url, seen = issue_config
    runtime = resolve_runtime_provider()
    assert (runtime["provider"], runtime.get("requested_provider")) == ("custom", "litellm")

    _provider, model, *_ = _main_route_target(runtime, "title_generation")

    assert model == "proxy-small-mini"
    assert seen == [f"Bearer {ROUTE_KEY}"]


def test_custom_catalog_probe_never_sends_the_ambient_openai_key(issue_config):
    """``provider_model_ids("custom")`` probes ``model.base_url``; the entry owning it supplies the
    key, and an OpenAI key is never a fallback for a third-party host (#91308)."""
    _url, seen = issue_config

    assert models_mod.provider_model_ids("custom", force_refresh=True) == ["proxy-small-mini"]
    assert seen == [f"Bearer {ROUTE_KEY}"]


def test_custom_catalog_probe_with_no_credential_is_keyless(issue_config, monkeypatch):
    """With no route credential at all the probe goes keyless, not with the OpenAI key."""
    _url, seen = issue_config
    monkeypatch.delenv("HERMES_LITELLM_KEY")

    models_mod.provider_model_ids("custom", force_refresh=True)

    assert OPENAI_KEY not in " ".join(h or "" for h in seen)


def test_picker_bare_custom_row_probes_with_the_entrys_key(issue_config):
    """The ``/model`` picker's bare custom row probes the current custom endpoint with the
    owning entry's key, not keyless (the issue's ``hermes-cli`` no-auth fingerprint)."""
    from hermes_cli.model_switch_providers import list_authenticated_providers

    url, seen = issue_config
    with patch("agent.models_dev.fetch_models_dev", return_value={}):
        list_authenticated_providers(
            current_provider="custom", current_base_url=url, current_model="my-combo",
            user_providers={}, custom_providers=[], refresh=True, probe_current_custom_provider=True)

    assert seen and set(seen) == {f"Bearer {ROUTE_KEY}"}
