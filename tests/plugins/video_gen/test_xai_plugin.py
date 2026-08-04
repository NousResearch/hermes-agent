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


@pytest.mark.asyncio
async def test_video_input_from_public_url_uses_url_field():
    from plugins.video_gen.xai import _video_input_from_public_url

    url = "https://files-cdn.x.ai/kRQVP6PRQlioVAUNC3GAdg/file_1faca9c3-9411-46ad-bb41-b9b8527789e6.mp4"
    result = await _video_input_from_public_url(
        url,
        api_key="test-key",
        base_url="https://api.x.ai/v1",
    )
    assert result == {"url": url}


def test_xai_video_image_input_blocks_credential_store_symlink(tmp_path, monkeypatch):
    from plugins.video_gen.xai import _image_ref_to_xai_input

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    auth_json = hermes_home / "auth.json"
    auth_json.write_text('{"api_key":"sk-secret"}', encoding="utf-8")
    image_link = hermes_home / "leak.png"
    try:
        image_link.symlink_to(auth_json)
    except OSError as exc:
        pytest.skip(f"symlink unavailable on this platform: {exc}")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(ValueError, match="credential store"):
        _image_ref_to_xai_input(str(image_link))


