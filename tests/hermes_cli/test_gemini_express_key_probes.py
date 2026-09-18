"""Key-validation surfaces route Vertex AI express keys (``AQ.``) to aiplatform, matching chat (#114335).

Sending an express key to generativelanguage.googleapis.com always 403s, so ``hermes doctor`` and the
dashboard key test would report a working key as rejected while chat succeeded.
"""

from __future__ import annotations

import asyncio

from agent.gemini_native_adapter import VERTEX_EXPRESS_BASE_URL

_STUDIO_MODELS = "https://generativelanguage.googleapis.com/v1beta/models"


def test_doctor_gemini_probe_routes_express_key_to_aiplatform():
    from hermes_cli.doctor_connectivity import _apikey_request

    _, url, headers = _apikey_request("AQ.express-key", None, _STUDIO_MODELS)
    assert url == VERTEX_EXPRESS_BASE_URL + "/models"
    assert headers["x-goog-api-key"] == "AQ.express-key" and "Authorization" not in headers

    # AI Studio keys keep hitting the Studio host.
    assert _apikey_request("AIza-studio-key", None, _STUDIO_MODELS)[1] == _STUDIO_MODELS


def test_dashboard_gemini_key_probe_routes_express_key_to_aiplatform(monkeypatch):
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

    body = EnvVarUpdate(key="GEMINI_API_KEY", value="AQ.express-key")
    out = asyncio.run(mod.validate_provider_credential(body, request=None))  # type: ignore[arg-type]

    assert out["ok"] is True
    assert seen["url"] == VERTEX_EXPRESS_BASE_URL + "/models"
