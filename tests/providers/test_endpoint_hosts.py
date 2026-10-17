"""A ``$HERMES_HOME`` provider plugin can claim tenant-scoped endpoint hosts (``endpoint_hosts``).

Named custom providers on a claimed host take the plugin's Anthropic auth style (``bearer_auth``)
and its model catalog (``fetch_models``); unclaimed hosts keep the built-in behavior.
"""

from __future__ import annotations

import json
import sys
import textwrap
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import patch

import pytest

_PLUGIN = textwrap.dedent(
    """
    from providers import register_provider
    from providers.base import ProviderProfile

    class TenantGateway(ProviderProfile):
        def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0):
            return {catalog!r} if "/catalog" in (base_url or "") else []

    register_provider(TenantGateway(name="tenant-gateway-test", hidden=True,
                                    endpoint_hosts={hosts!r}, bearer_auth=True))
    """
)

# A pre-``endpoint_hosts`` duck-typed profile: register_provider accepts it.
_LEGACY_PLUGIN = textwrap.dedent(
    """
    from providers import register_provider

    class LegacyProfile:
        name = "legacy-profile-test"
        aliases = ()

    register_provider(LegacyProfile())
    """
)


@pytest.fixture
def claim(tmp_path, monkeypatch):
    import providers

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_HOME_LAYERS", {}, raising=False)

    def install(hosts=("gw.example",), catalog=("alias-a",), source=_PLUGIN):
        plugin = tmp_path / "plugins" / "model-providers" / "tenant-gateway"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text("name: tenant-gateway\nkind: model-provider\n", encoding="utf-8")
        (plugin / "__init__.py").write_text(source.format(hosts=hosts, catalog=list(catalog)), encoding="utf-8")
        monkeypatch.setattr(providers, "_HOME_LAYERS", {})  # rescan now, not after the 1 s stamp TTL

    yield install
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


@pytest.fixture
def server():
    seen = []

    class Models(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, dict(self.headers)))
            body = json.dumps({"data": [{"id": "generic-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Models) as httpd:
        worker = Thread(target=httpd.serve_forever)
        worker.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_port}", seen
        finally:
            httpd.shutdown()
            worker.join()


def _sdk_kwargs(base_url):
    from agent.anthropic_adapter import build_anthropic_client

    with patch("agent.anthropic_adapter._anthropic_sdk") as sdk:
        build_anthropic_client("static-key", base_url=base_url)
        return sdk.Anthropic.call_args[1]


def _headers(seen):
    return {k.lower(): v for k, v in seen[0][1].items()}


@pytest.mark.parametrize("hosts", [("gw.example",), "gw.example"])
def test_claimed_tenant_host_builds_anthropic_client_with_bearer(claim, hosts):
    claim(hosts=hosts)
    kwargs = _sdk_kwargs("https://acme.gw.example/anthropic")
    assert kwargs["auth_token"] == "static-key" and "api_key" not in kwargs
    lookalike = _sdk_kwargs("https://gw.example.evil.test/anthropic")
    assert lookalike["api_key"] == "static-key" and "auth_token" not in lookalike


def test_profile_without_endpoint_hosts_does_not_break_client_builds(claim):
    claim(source=_LEGACY_PLUGIN)
    kwargs = _sdk_kwargs("https://anthropic-compatible.invalid")
    assert kwargs["api_key"] == "static-key" and "auth_token" not in kwargs


def test_claimed_host_catalog_answers_the_probe_without_a_models_request(claim, server):
    from hermes_cli.models import probe_api_models

    base, seen = server
    claim(hosts=("127.0.0.1",))
    probe = probe_api_models("k", base + "/catalog", api_mode="anthropic_messages")
    assert probe["models"] == ["alias-a"] and seen == []


def test_claimed_host_with_an_empty_catalog_falls_back_to_a_bearer_models_probe(claim, server):
    from hermes_cli.models import probe_api_models

    base, seen = server
    claim(hosts=("127.0.0.1",))
    probe = probe_api_models("k", base + "/other", api_mode="anthropic_messages")
    assert probe["models"] == ["generic-model"]
    headers = _headers(seen)
    assert headers["authorization"] == "Bearer k" and "x-api-key" not in headers
    assert headers["anthropic-version"] == "2023-06-01"

