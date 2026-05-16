"""Smoke tests for the xAI video gen plugin — load & register surface."""

from __future__ import annotations

import pytest

from agent import video_gen_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    video_gen_registry._reset_for_tests()
    yield
    video_gen_registry._reset_for_tests()


def test_xai_provider_registers():
    from plugins.video_gen.xai import XAIVideoGenProvider

    provider = XAIVideoGenProvider()
    video_gen_registry.register_provider(provider)

    assert video_gen_registry.get_provider("xai") is provider
    assert provider.display_name == "xAI"
    assert provider.default_model() == "grok-imagine-video"


def test_xai_capabilities_text_and_image_only():
    """xAI was previously advertised with edit/extend operations. The
    simplified surface only exposes text-to-video and image-to-video —
    confirm those are the only modalities advertised."""
    from plugins.video_gen.xai import XAIVideoGenProvider

    caps = XAIVideoGenProvider().capabilities()
    assert caps["modalities"] == ["text", "image"]
    # No 'operations' key in the simplified surface
    assert "operations" not in caps
    assert caps["max_reference_images"] == 7


def test_xai_unavailable_without_key(monkeypatch):
    from plugins.video_gen.xai import XAIVideoGenProvider

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert XAIVideoGenProvider().is_available() is False


def test_xai_generate_requires_xai_key(monkeypatch):
    from plugins.video_gen.xai import XAIVideoGenProvider

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    result = XAIVideoGenProvider().generate("a happy dog")
    assert result["success"] is False
    assert result["error_type"] == "auth_required"


def test_xai_available_with_oauth_only(monkeypatch):
    """The plugin must honour xAI Grok OAuth credentials, not just
    XAI_API_KEY. Otherwise the agent's tool-availability check filters
    ``video_generate`` out of the toolbelt and the agent silently falls
    back to whatever skill advertises video generation (e.g. comfyui).
    """
    import plugins.video_gen.xai as xai_plugin

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "tools.xai_http.resolve_xai_http_credentials",
        lambda: {
            "provider": "xai-oauth",
            "api_key": "oauth-bearer-token",
            "base_url": "https://api.x.ai/v1",
        },
    )

    assert xai_plugin.XAIVideoGenProvider().is_available() is True


def test_xai_resolved_credentials_threaded_through_request(monkeypatch):
    """OAuth-resolved creds must reach the HTTP layer — bug class where
    ``is_available()`` says yes but the request still hits with no key.
    """
    import plugins.video_gen.xai as xai_plugin

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "tools.xai_http.resolve_xai_http_credentials",
        lambda: {
            "provider": "xai-oauth",
            "api_key": "oauth-bearer-token",
            "base_url": "https://api.x.ai/v1",
        },
    )

    api_key, base_url = xai_plugin._resolve_xai_credentials()
    assert api_key == "oauth-bearer-token"
    assert base_url == "https://api.x.ai/v1"
    headers = xai_plugin._xai_headers(api_key)
    assert headers["Authorization"] == "Bearer oauth-bearer-token"


def test_xai_no_operation_kwarg():
    """The ABC's generate() signature no longer accepts 'operation'.
    Passing it through **kwargs should be ignored (forward-compat)."""
    from plugins.video_gen.xai import XAIVideoGenProvider

    # We're not actually hitting the network — just verify the call
    # doesn't TypeError on the unexpected kwarg.
    # Will fail with auth_required (no XAI_API_KEY), but should NOT
    # fail with TypeError.
    result = XAIVideoGenProvider().generate("x", operation="generate")
    assert result["success"] is False
    # auth_required, NOT some signature error
    assert result["error_type"] in ("auth_required", "api_error")
