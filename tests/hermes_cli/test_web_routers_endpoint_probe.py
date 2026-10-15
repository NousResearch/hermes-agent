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
        assert "401" in data["message"]
    else:
        body = CustomEndpointUpdate(id="", name="local", base_url="http://127.0.0.1:39080", api_key="k", model="")
        data = asyncio.run(mod.validate_custom_endpoint(body))
    assert data["ok"] is False and data["reachable"] is True
    assert "404" not in data["message"]


def _models_probe_host(monkeypatch, content_type, body):
    """A custom-endpoint ``/models`` probe answering 200 with the given body/content type.
    The transport POST answers 200 too — an SPA catch-all serves every route, so the
    transport probe alone cannot catch it."""
    import hermes_cli.web_routers.config_env as mod

    class _Resp:
        status_code = 200
        is_success = True
        headers = {"content-type": content_type}

        def json(self):
            if isinstance(body, str):
                raise ValueError("Expecting value: not JSON")
            return body

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, *a, **k):
            return _Resp()

        async def post(self, url, *a, **k):
            return _Resp()

    monkeypatch.setattr(mod, "_endpoint_probe_client", lambda url, timeout: _Client())


@pytest.mark.parametrize(
    "name, content_type, body, needle",
    [
        ("spa_html", "text/html", "<html><body><div id=root></div></body></html>", "text/html"),
        ("non_json_body", "text/plain", "OK", "instead of a JSON model list"),
        ("empty_model_list", "application/json", {"data": []}, "advertised no models"),
    ],
)
def test_custom_endpoint_probe_warns_when_no_models_parse(name, content_type, body, needle, monkeypatch):
    """A 200 ``/models`` answer is only a pass when it is JSON with a usable model list: an SPA
    catch-all (200 + HTML), a non-JSON body, or an empty list must be ``ok:false`` with a
    message naming the cause, not a green check (#83128)."""
    import hermes_cli.web_routers.config_env as mod
    from hermes_cli.web_models import CustomEndpointUpdate

    _models_probe_host(monkeypatch, content_type, body)
    req = CustomEndpointUpdate(id="", name="x", base_url="https://spa.example.com/v1",
                               api_key="", model="")
    data = asyncio.run(mod.validate_custom_endpoint(req))

    assert data["ok"] is False, name
    assert data["reachable"] is True, name  # warning, not a hard block: Save stays possible
    assert needle in data["message"], (name, data["message"])
    assert "https://spa.example.com/v1" in data["message"], name
    assert data["models"] == [], name


def test_custom_endpoint_probe_passes_on_a_json_models_reply(monkeypatch):
    """A genuine OpenAI-compatible reply (200, JSON, ``{"data": [...]}``) still passes — the
    no-models guard classifies only empty/unparseable answers, never honest ones."""
    import hermes_cli.web_routers.config_env as mod
    from hermes_cli.web_models import CustomEndpointUpdate

    _models_probe_host(monkeypatch, "application/json",
                       {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}], "object": "list"})
    req = CustomEndpointUpdate(id="", name="x", base_url="https://api.example.com/v1",
                               api_key="", model="")
    data = asyncio.run(mod.validate_custom_endpoint(req))

    assert data["ok"] is True and data["reachable"] is True
    assert data["models"] == ["gpt-4o", "gpt-4o-mini"]
    assert data["message"] == ""
