"""Vertex visibility in the /model picker (follow-up to PR #56688).

Community verification of the vertex-registration fix found two remaining
gaps that kept the provider invisible/unusable in the /model menu:

1. ``list_authenticated_providers`` had no credential gate for the
   ``vertex`` auth_type (only ``aws_sdk`` was special-cased), so the
   provider was silently hidden even when fully configured.
2. Vertex's OpenAI-compatible endpoint has no ``/models`` listing route,
   so without a curated ``_PROVIDER_MODELS["vertex"]`` entry the picker
   only ever showed the currently-configured model.

No network calls.
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.models import _PROVIDER_MODELS


def test_vertex_has_curated_model_list():
    """Vertex has no /models route — the picker needs a static curated list."""
    models = _PROVIDER_MODELS.get("vertex")
    assert models, "_PROVIDER_MODELS must have a non-empty 'vertex' entry"
    # Vertex's openapi endpoint expects the google/ publisher prefix.
    assert all(m.startswith("google/") for m in models)


def test_vertex_appears_when_credentials_configured():
    """has_vertex_credentials() == True must surface vertex in the picker."""
    with patch("agent.vertex_adapter.has_vertex_credentials", return_value=True):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    vertex = next((p for p in providers if p["slug"] == "vertex"), None)
    assert vertex is not None, "vertex should appear when credentials are configured"
    assert vertex["models"], "vertex row must carry the curated model list"
    assert set(vertex["models"]) <= set(_PROVIDER_MODELS["vertex"])


def test_vertex_hidden_without_credentials():
    """No service-account path / project override → vertex stays hidden."""
    with patch("agent.vertex_adapter.has_vertex_credentials", return_value=False):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    vertex = next((p for p in providers if p["slug"] == "vertex"), None)
    assert vertex is None, "vertex should not appear without credentials"


def test_vertex_auth_status_and_provider_detection():
    """get_auth_status('vertex') must dispatch auth_type='vertex' so /model auto-detects vertex."""
    from hermes_cli.auth import get_auth_status
    from hermes_cli.models import detect_provider_for_model

    with patch("agent.vertex_adapter.has_vertex_credentials", return_value=True):
        status = get_auth_status("vertex")
        assert status["logged_in"] is True
        assert status["provider"] == "vertex"
        detected = detect_provider_for_model("google/gemini-4-argon", "anthropic")
        assert detected == ("vertex", "google/gemini-4-argon")

    with patch("agent.vertex_adapter.has_vertex_credentials", return_value=False):
        assert get_auth_status("vertex")["logged_in"] is False


def test_vertex_normalize_repairs_bare_gemini_model_ids():
    """Bare Gemini/Gemma IDs on Vertex must receive the required google/ publisher prefix."""
    from hermes_cli.model_normalize import normalize_model_for_provider

    assert normalize_model_for_provider("gemini-3.8-flash", "vertex") == "google/gemini-3.8-flash"
    assert normalize_model_for_provider("gemini-4-argon", "vertex") == "google/gemini-4-argon"
    assert normalize_model_for_provider("gemini-2.5-pro", "vertex") == "google/gemini-2.5-pro"
    assert normalize_model_for_provider("vertex:gemini-3.8-flash", "vertex") == "google/gemini-3.8-flash"
    assert normalize_model_for_provider("google/gemini-4-argon", "vertex") == "google/gemini-4-argon"


def test_vertex_validate_requested_model_uses_static_catalog_without_live_probe():
    """validate_requested_model on vertex must not probe the nonexistent /models endpoint."""
    from hermes_cli.models_validate import validate_requested_model

    with patch("hermes_cli.models.fetch_api_models", side_effect=AssertionError("/models must not be called")):
        res_prefixed = validate_requested_model("google/gemini-3.8-flash", "vertex")
        assert res_prefixed == {"accepted": True, "persist": True, "recognized": True, "message": None}

        res_bare = validate_requested_model("gemini-3.8-flash", "vertex")
        assert res_bare == {"accepted": True, "persist": True, "recognized": True, "message": None}

        res_uncatalogued = validate_requested_model("google/gemini-2.5-pro", "vertex")
        assert res_uncatalogued["accepted"] is True
        assert res_uncatalogued["persist"] is True
        assert res_uncatalogued["recognized"] is False
        assert "Vertex AI does not expose a /models endpoint" in (res_uncatalogued["message"] or "")


def test_vertex_switch_model_resets_sticky_region():
    """Switching models on Vertex must reset sticky region affinity so a new model can acquire europe-west2."""
    from typing import Any
    from unittest.mock import MagicMock
    from agent.vertex_adapter import _record_session_vertex_region_affinity, get_session_vertex_region_affinity
    from run_agent import AIAgent

    agent: Any = AIAgent.__new__(AIAgent)
    agent.session_id = "test_vertex_switch_sess"
    agent.provider = "vertex"
    agent.model = "google/gemini-3.8-flash"
    agent.base_url = "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi"
    agent.api_key = "ya29.dummy"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent._client_kwargs = {"api_key": "ya29.dummy", "base_url": agent.base_url}
    agent.context_compressor = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None

    _record_session_vertex_region_affinity("test_vertex_switch_sess", "global")
    assert get_session_vertex_region_affinity("test_vertex_switch_sess") == "global"

    with patch.object(agent, "_create_openai_client", return_value=MagicMock()), \
         patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        agent.switch_model(
            new_model="google/gemini-4-argon",
            new_provider="vertex",
            api_key="ya29.dummy",
            base_url=agent.base_url,
            api_mode="chat_completions",
        )

    assert get_session_vertex_region_affinity("test_vertex_switch_sess") is None


