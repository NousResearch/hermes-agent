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


# ---------------------------------------------------------------------------
# Model name normalization for Vertex AI
# ---------------------------------------------------------------------------


class TestVertexModelNormalization:
    """Vertex AI requires the 'google/' publisher prefix on all model IDs."""

    @pytest.mark.parametrize(
        "model_input, expected",
        [
            ("gemini-3.1-flash-lite", "google/gemini-3.1-flash-lite"),
            ("gemini-2.5-pro", "google/gemini-2.5-pro"),
            ("gemini-3-flash-preview", "google/gemini-3-flash-preview"),
        ],
        ids=["flash-lite", "pro", "flash-preview"],
    )
    def test_bare_name_gets_google_prefix(self, model_input, expected):
        from hermes_cli.model_normalize import normalize_model_for_provider

        assert normalize_model_for_provider(model_input, "vertex") == expected

    @pytest.mark.parametrize(
        "model_input",
        [
            "google/gemini-3.1-flash-lite",
            "google/gemini-2.5-pro",
            "google/gemini-3-pro-preview",
        ],
        ids=["flash-lite", "pro", "pro-preview"],
    )
    def test_already_prefixed_unchanged(self, model_input):
        from hermes_cli.model_normalize import normalize_model_for_provider

        assert normalize_model_for_provider(model_input, "vertex") == model_input

    def test_vertex_prefix_stripped_then_google_added(self):
        """'vertex/gemini-2.5-flash' → strip 'vertex/' → add 'google/'."""
        from hermes_cli.model_normalize import normalize_model_for_provider

        assert normalize_model_for_provider("vertex/gemini-2.5-flash", "vertex") == "google/gemini-2.5-flash"

    def test_empty_input_returns_empty(self):
        from hermes_cli.model_normalize import normalize_model_for_provider

        assert normalize_model_for_provider("", "vertex") == ""
        assert normalize_model_for_provider(None, "vertex") == ""
