"""Endpoint-probe contract for the Desktop local/custom endpoint validators (#63472).

httpx honours ``HTTP(S)_PROXY`` (and the Windows system proxy) but never the proxy bypass list,
so a system proxy answered ``127.0.0.1`` probes with its own error page. The GUI then reported
"advertised no models" for a llama.cpp server the CLI (urllib, honours the bypass) saw fine.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.parametrize(
    "url, trusts_env",
    [
        ("http://127.0.0.1:8080/v1/models", False),
        ("http://localhost:11434/v1/models", False),
        ("http://192.168.1.20:8000/v1/models", False),
        ("https://api.example.com/v1/models", True),
    ],
)
def test_local_endpoint_probes_bypass_env_proxy(url, trusts_env, monkeypatch):
    from hermes_cli.web_routers.config_env import _endpoint_probe_client

    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    client = _endpoint_probe_client(url, 1.0)
    assert client.trust_env is trusts_env


def test_openai_base_url_probe_names_the_http_status_instead_of_no_models(monkeypatch):
    """A reachable endpoint answering non-2xx with no model list is a failure the user can act on,
    not an empty catalog the GUI turns into 'start a model on that endpoint'."""
    import hermes_cli.web_routers.config_env as mod
    from hermes_cli.web_models import EnvVarUpdate

    class _Resp:
        status_code = 502
        is_success = False

        def json(self):
            return {"error": "proxy upstream unavailable"}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(mod, "_endpoint_probe_client", lambda url, timeout: _Client())
    monkeypatch.setattr(mod, "_require_token", lambda request: None)

    body = EnvVarUpdate(key="OPENAI_BASE_URL", value="http://127.0.0.1:8080/v1", api_key="")
    out = asyncio.run(mod.validate_provider_credential(body, request=None))  # type: ignore[arg-type]

    assert out["ok"] is False and out["reachable"] is True
    assert "HTTP 502" in out["message"]


@pytest.mark.parametrize("route", ["/api/providers/validate", "/api/providers/custom-endpoints/validate"])
def test_bare_root_probe_resolves_to_the_v1_base_that_served_models(route, monkeypatch):
    """A custom endpoint typed without ``/v1`` (#65488): the probe must fall through to
    ``{base}/v1/models`` AND report that base as ``resolved_base_url`` so the Desktop persists a URL
    the runtime can POST ``/chat/completions`` to — detection green + every chat 404 is the bug."""
    import hermes_cli.web_routers.config_env as mod
    from hermes_cli.web_models import CustomEndpointUpdate, EnvVarUpdate

    class _Resp:
        def __init__(self, status):
            self.status_code, self.is_success = status, status == 200

        def json(self):
            return {"data": [{"id": "local-model"}]} if self.is_success else {"error": "Unexpected endpoint"}

    seen = []

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, *a, **k):
            seen.append(url)
            return _Resp(200 if url.endswith("/v1/models") else 404)

    monkeypatch.setattr(mod, "_endpoint_probe_client", lambda url, timeout: _Client())
    monkeypatch.setattr(mod, "_require_token", lambda request: None)
    if route == "/api/providers/validate":
        body = EnvVarUpdate(key="OPENAI_BASE_URL", value="http://127.0.0.1:39080/", api_key="")
        data = asyncio.run(mod.validate_provider_credential(body, request=None))
    else:
        body = CustomEndpointUpdate(id="", name="local", base_url="http://127.0.0.1:39080/", api_key="", model="")
        data = asyncio.run(mod.validate_custom_endpoint(body))

    assert seen == ["http://127.0.0.1:39080/models", "http://127.0.0.1:39080/v1/models"]
    assert data["ok"] is True and data["models"] == ["local-model"]
    assert data["resolved_base_url"] == "http://127.0.0.1:39080/v1"


@pytest.mark.parametrize("route", ["/api/providers/validate", "/api/providers/custom-endpoints/validate"])
def test_bare_root_probe_reports_the_v1_key_rejection_not_the_root_404(route, monkeypatch):
    """Server lives at ``/v1`` and wants a key: typed root 404s, ``/v1/models`` answers 401. The
    verdict must be the key rejection from the candidate that produced it, not the first 404."""
    import hermes_cli.web_routers.config_env as mod
    from hermes_cli.web_models import CustomEndpointUpdate, EnvVarUpdate

    class _Resp:
        def __init__(self, status):
            self.status_code, self.is_success = status, False

        def json(self):
            return {"error": "unauthorized"}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, *a, **k):
            return _Resp(401 if url.endswith("/v1/models") else 404)

    monkeypatch.setattr(mod, "_endpoint_probe_client", lambda url, timeout: _Client())
    monkeypatch.setattr(mod, "_require_token", lambda request: None)
    if route == "/api/providers/validate":
        body = EnvVarUpdate(key="OPENAI_BASE_URL", value="http://127.0.0.1:39080", api_key="k")
        data = asyncio.run(mod.validate_provider_credential(body, request=None))
        assert data["message"] == "http://127.0.0.1:39080/v1/models answered HTTP 401."
    else:
        body = CustomEndpointUpdate(id="", name="local", base_url="http://127.0.0.1:39080", api_key="k", model="")
        data = asyncio.run(mod.validate_custom_endpoint(body))
        assert data["message"] == "The endpoint rejected the API key."
    assert data["ok"] is False and data["reachable"] is True
