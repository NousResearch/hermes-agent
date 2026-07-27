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


def test_vertex_registered_in_provider_registry():
    """PROVIDER_REGISTRY (hermes_cli.auth) is what agent/auxiliary_client.py's
    resolve_provider_client() looks up before dispatching on auth_type. Without
    an entry here, the ``elif pconfig.auth_type == "vertex":`` branch there is
    unreachable dead code — every auxiliary Vertex call (vision, title
    generation, MoA reference/aggregator slots, ...) fails at the
    ``pconfig is None`` guard before ever reaching it."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    cfg = PROVIDER_REGISTRY.get("vertex")
    assert cfg is not None
    assert cfg.auth_type == "vertex"


def test_vertex_registered_in_hermes_overlays():
    """hermes_cli.providers.get_provider("vertex") backs
    _preserve_provider_with_base_url() in agent/auxiliary_client.py, which
    decides whether a MoA slot's resolved Vertex (base_url, api_key) pair
    keeps its "vertex" provider identity or silently collapses to "custom" —
    losing the identity _refresh_provider_credentials() needs to re-mint an
    expired OAuth2 token on a 401."""
    from hermes_cli.providers import get_provider

    resolved = get_provider("vertex")
    assert resolved is not None
    assert resolved.auth_type == "vertex"
