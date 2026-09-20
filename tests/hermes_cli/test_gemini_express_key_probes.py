"""Key-validation surfaces route Gemini keys to Google AI Studio by default (#116053),
and route to Vertex AI express when GEMINI_BASE_URL points to aiplatform.
"""

from __future__ import annotations

import asyncio

from agent.gemini_native_adapter import VERTEX_EXPRESS_BASE_URL

_STUDIO_MODELS = "https://generativelanguage.googleapis.com/v1beta/models"


def test_doctor_gemini_probe_defaults_to_studio_for_aq_and_aiza_keys():
    from hermes_cli.doctor_connectivity import _apikey_request

    # Current AQ.* Google AI Studio keys default to the Studio host (#116053).
    _, url, headers = _apikey_request("AQ.studio-key", None, _STUDIO_MODELS)
    assert url == _STUDIO_MODELS
    assert headers["x-goog-api-key"] == "AQ.studio-key" and "Authorization" not in headers

    # Legacy AIza keys keep hitting the Studio host.
    _, url_aiza, headers_aiza = _apikey_request("AIza-studio-key", None, _STUDIO_MODELS)
    assert url_aiza == _STUDIO_MODELS
    assert headers_aiza["x-goog-api-key"] == "AIza-studio-key" and "Authorization" not in headers_aiza


def test_doctor_gemini_probe_routes_to_aiplatform_when_base_url_configured(monkeypatch):
    from hermes_cli.doctor_connectivity import _apikey_request

    # Explicit Vertex AI express base routes to aiplatform express surface.
    monkeypatch.setenv("GEMINI_BASE_URL", "https://aiplatform.googleapis.com")
    _, url, headers = _apikey_request("AQ.express-key", "GEMINI_BASE_URL", _STUDIO_MODELS)
    assert url == VERTEX_EXPRESS_BASE_URL + "/models"
    assert headers["x-goog-api-key"] == "AQ.express-key" and "Authorization" not in headers


def test_dashboard_gemini_key_probe_uses_default_and_configured_bases(monkeypatch):
    import hermes_cli.web_routers.config_env as mod
    from hermes_cli.web_models import EnvVarUpdate

    seen = {}

    class _Resp:
        status_code = 200
        is_success = True

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None, **k):
            seen["url"] = url
            seen["headers"] = headers or {}
            return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(mod, "_require_token", lambda request: None)

    body = EnvVarUpdate(key="GEMINI_API_KEY", value="AQ.studio-key")
    out = asyncio.run(mod.validate_provider_credential(body, request=None))  # type: ignore[arg-type]

    assert out["ok"] is True
    assert seen["url"] == _STUDIO_MODELS
    assert seen["headers"].get("x-goog-api-key") == "AQ.studio-key"

    seen.clear()
    monkeypatch.setattr(mod, "_gemini_base_url_for_profile", lambda profile: "https://aiplatform.googleapis.com")
    body = EnvVarUpdate(key="GEMINI_API_KEY", value="AQ.express-key")
    out = asyncio.run(mod.validate_provider_credential(body, request=None))  # type: ignore[arg-type]

    assert out["ok"] is True
    assert seen["url"] == VERTEX_EXPRESS_BASE_URL + "/models"
    assert seen["headers"].get("x-goog-api-key") == "AQ.express-key"
