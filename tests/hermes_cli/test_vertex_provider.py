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

# --- new tests appended to tests/hermes_cli/test_vertex_provider.py ---
# (These verify the PROVIDER_REGISTRY entry + list_authenticated_providers picker fix
# that unblocks [#56687](https://github.com/NousResearch/hermes-agent/issues/56687) — the desktop provider-setup dead-end for Vertex.)


def test_vertex_registered_in_provider_registry():
    """Regression for [#56687](https://github.com/NousResearch/hermes-agent/issues/56687): PROVIDER_REGISTRY must include vertex so that
    list_authenticated_providers's credential-detection loop can see it. Without
    this row, `_auth_registry.get("vertex")` returns None and vertex never reaches
    the auth_type branches at all.
    """
    from hermes_cli.auth import PROVIDER_REGISTRY

    assert "vertex" in PROVIDER_REGISTRY, (
        "vertex must be registered in PROVIDER_REGISTRY for the desktop model picker "
        "to see it as an authenticated provider (see [#56687](https://github.com/NousResearch/hermes-agent/issues/56687))"
    )
    v = PROVIDER_REGISTRY["vertex"]
    assert v.auth_type == "vertex", (
        "auth_type must be 'vertex' so list_authenticated_providers dispatches to "
        "the get_vertex_config() credential probe rather than falling through to "
        "the default api_key path"
    )
    # Non-api_key providers should have empty env-var tuple (mirrors bedrock/aws_sdk)
    assert v.api_key_env_vars == ()


def test_vertex_appears_in_authenticated_providers_when_configured(monkeypatch):
    """End-to-end contract for the desktop model picker.

    When get_vertex_config() returns a live (token, base_url) tuple — i.e. ADC or a
    service-account JSON is resolvable — list_authenticated_providers must emit a
    vertex row with a non-empty models list. Without this, the desktop app renders
    a "Set up Vertex" button whose target modal doesn't list Vertex, dead-ending
    the user ([#56687](https://github.com/NousResearch/hermes-agent/issues/56687)).
    """
    import agent.vertex_adapter as va
    from hermes_cli.model_switch import list_authenticated_providers

    monkeypatch.setattr(
        va,
        "get_vertex_config",
        lambda: (
            "ya29.TOKEN",
            "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi",
        ),
    )

    rows = list_authenticated_providers()
    vertex_rows = [r for r in rows if r.get("slug") == "vertex"]

    assert vertex_rows, (
        "vertex must appear in list_authenticated_providers() when credentials "
        "resolve, otherwise the desktop picker renders a dead-end Set-up button"
    )
    row = vertex_rows[0]
    assert row.get("models"), (
        "vertex picker row must have a non-empty model list — the OpenAI-compat "
        "endpoint has no /models route, so we fall back to a curated Gemini list"
    )
    # The fallback list must include the current-generation flagship
    assert any("gemini" in m.lower() for m in row["models"])


def test_vertex_hidden_from_picker_when_creds_unresolved(monkeypatch):
    """Mirror-image contract: if credentials can't be resolved, vertex must NOT
    appear as authenticated. This prevents false-positive picker rows that would
    500 on first request.
    """
    import agent.vertex_adapter as va
    from hermes_cli.model_switch import list_authenticated_providers

    monkeypatch.setattr(va, "get_vertex_config", lambda: (None, None))

    rows = list_authenticated_providers()
    vertex_rows = [r for r in rows if r.get("slug") == "vertex"]
    # Absent is fine; if present, it must be marked as source=canonical skeleton
    # (i.e. not surfaced as authenticated). We assert the strict "absent" contract
    # because that's what desktop's isProviderReady() actually gates on.
    assert not vertex_rows, (
        "vertex must not appear as an authenticated provider when "
        "get_vertex_config() returns (None, None)"
    )
