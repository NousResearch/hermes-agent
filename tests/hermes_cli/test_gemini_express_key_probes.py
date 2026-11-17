"""Key-validation surfaces never reroute a key by its prefix, matching chat (#115306).

Google issues ``AQ.`` keys for both Google AI Studio and Vertex express mode, so a key prefix can
no longer pick the surface: ``hermes doctor`` and the dashboard key test probe the default Studio
host, and an express key reaches aiplatform only through an explicitly configured base URL.
"""

from __future__ import annotations

import asyncio

from agent.gemini_native_adapter import VERTEX_EXPRESS_BASE_URL

_STUDIO_MODELS = "https://generativelanguage.googleapis.com/v1beta/models"


def test_doctor_gemini_probe_keeps_aq_keys_on_the_studio_host(monkeypatch):
    from hermes_cli.doctor_connectivity import _apikey_request

    _, url, headers = _apikey_request(
        "AQ.studio-key", "GEMINI_BASE_URL", _STUDIO_MODELS
    )
    assert url == _STUDIO_MODELS
    assert (
        headers["x-goog-api-key"] == "AQ.studio-key" and "Authorization" not in headers
    )

    # Legacy AIza Studio keys keep hitting the Studio host as well.
    assert (
        _apikey_request("AIza-studio-key", "GEMINI_BASE_URL", _STUDIO_MODELS)[1]
        == _STUDIO_MODELS
    )

    # An explicitly configured aiplatform base is still completed to the publishers form.
    monkeypatch.setenv("GEMINI_BASE_URL", "https://aiplatform.googleapis.com")
    _, url, headers = _apikey_request(
        "AQ.express-key", "GEMINI_BASE_URL", _STUDIO_MODELS
    )
    assert url == VERTEX_EXPRESS_BASE_URL + "/models"
    assert headers["x-goog-api-key"] == "AQ.express-key"


def test_dashboard_gemini_key_probe_keeps_aq_keys_on_the_studio_host(monkeypatch):
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

        async def get(self, url, **k):
            seen["url"] = url
            return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(mod, "_require_token", lambda request: None)

    body = EnvVarUpdate(key="GEMINI_API_KEY", value="AQ.studio-key")
    out = asyncio.run(mod.validate_provider_credential(body, request=None))  # type: ignore[arg-type]

    assert out["ok"] is True
    assert seen["url"] == _STUDIO_MODELS
