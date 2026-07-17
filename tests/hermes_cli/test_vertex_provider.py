"""Tests for Vertex AI runtime-provider resolution and profile registration.

Covers: provider-profile registration + aliases, alias canonicalization,
resolve_runtime_provider(vertex) minting an OAuth token, and the friendly
AuthError when credentials can't be resolved. No network calls.
"""

from __future__ import annotations

import pytest


def test_vertex_profile_registered():
    from providers import get_provider_profile

    p = get_provider_profile("vertex")
    assert p is not None
    assert p.name == "vertex"
    assert p.api_mode == "chat_completions"
    assert p.auth_type == "vertex"


@pytest.mark.parametrize("alias", ["google-vertex", "vertex-ai", "gcp-vertex"])
def test_vertex_aliases_resolve(alias):
    from providers import get_provider_profile

    assert get_provider_profile(alias).name == "vertex"


@pytest.mark.parametrize("alias", ["google-vertex", "vertex-ai", "gcp-vertex", "vertexai"])
def test_alias_canonicalizes_to_vertex(alias):
    from hermes_cli.models import _PROVIDER_ALIASES

    assert _PROVIDER_ALIASES[alias] == "vertex"


def test_google_vertex_not_confused_with_gemini():
    """`google-vertex` must map to vertex, not the AI-Studio `gemini` provider."""
    from hermes_cli.models import _PROVIDER_ALIASES

    assert _PROVIDER_ALIASES["google-vertex"] == "vertex"
    assert _PROVIDER_ALIASES["google-gemini"] == "gemini"


def test_resolve_runtime_provider_mints_token(monkeypatch):
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(
        va, "get_vertex_config",
        lambda: ("ya29.TOKEN", "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi"),
    )
    rt = rp.resolve_runtime_provider(requested="vertex")
    assert rt["provider"] == "vertex"
    assert rt["api_mode"] == "chat_completions"
    assert rt["source"] == "vertex-oauth"
    assert rt["api_key"] == "ya29.TOKEN"
    assert "aiplatform.googleapis.com" in rt["base_url"]


def test_resolve_runtime_provider_alias(monkeypatch):
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(va, "get_vertex_config", lambda: ("t", "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi"))
    rt = rp.resolve_runtime_provider(requested="google-vertex")
    assert rt["provider"] == "vertex"


def test_resolve_runtime_provider_raises_autherror_when_unresolved(monkeypatch):
    import agent.vertex_adapter as va
    from hermes_cli import runtime_provider as rp
    from hermes_cli.auth import AuthError

    monkeypatch.setattr(va, "get_vertex_config", lambda: (None, None))
    with pytest.raises(AuthError) as exc:
        rp.resolve_runtime_provider(requested="vertex")
    msg = str(exc.value)
    assert "OAuth2" in msg
    assert "not a static API key" in msg


def test_vertex_auth_status_uses_adc_credential_detector(monkeypatch):
    import agent.vertex_adapter as va
    from hermes_cli.auth import get_auth_status

    monkeypatch.setattr(va, "has_vertex_credentials", lambda: True)

    assert get_auth_status("vertex") == {
        "logged_in": True,
        "provider": "vertex",
    }


def test_vertex_adc_surfaces_in_task_model_options(monkeypatch):
    import agent.models_dev as models_dev
    import agent.vertex_adapter as va
    import hermes_cli.model_switch as model_switch
    import hermes_cli.models as models
    from hermes_cli.inventory import ConfigContext, build_models_payload

    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda: {})
    monkeypatch.setattr(va, "has_vertex_credentials", lambda: True)
    monkeypatch.setattr(
        model_switch,
        "_credential_pool_is_usable",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(models, "cached_provider_model_ids", lambda provider: [])

    payload = build_models_payload(
        ConfigContext(
            current_provider="openai-codex",
            current_model="gpt-test",
            current_base_url="",
            user_providers={},
            custom_providers=[],
        ),
        include_unconfigured=True,
        picker_hints=True,
    )
    vertex = next(row for row in payload["providers"] if row["slug"] == "vertex")

    assert vertex["authenticated"] is True
    assert vertex["source"] == "canonical"
    assert vertex["models"] == [
        "google/gemini-3-pro-preview",
        "google/gemini-3-flash-preview",
    ]


def test_vertex_extra_body_thinking_config():
    from providers import get_provider_profile

    p = get_provider_profile("vertex")
    body = p.build_extra_body(
        model="google/gemini-3-pro-preview",
        reasoning_config={"effort": "high"},
    )
    assert "extra_body" in body
    assert "google" in body["extra_body"]
    assert "thinking_config" in body["extra_body"]["google"]


def test_vertex_extra_body_empty_without_reasoning():
    from providers import get_provider_profile

    p = get_provider_profile("vertex")
    assert p.build_extra_body(model="google/gemini-3-flash-preview") == {}
