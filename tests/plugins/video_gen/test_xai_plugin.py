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


def test_xai_provider_lists_text_and_current_image_video_models():
    from plugins.video_gen.xai import XAIVideoGenProvider

    models = XAIVideoGenProvider().list_models()
    ids = [model["id"] for model in models]

    assert ids[0] == "grok-imagine-video"
    assert ids[1] == "grok-imagine-video-1.5"
    assert models[1]["modalities"] == ["image"]
    assert "aliases" not in models[1]


def test_xai_routes_default_models_by_modality():
    from plugins.video_gen.xai import _resolve_model_for_modality

    assert _resolve_model_for_modality(
        "grok-imagine-video",
        modality="text",
        explicit_model=False,
    ) == "grok-imagine-video"
    assert _resolve_model_for_modality(
        "grok-imagine-video",
        modality="image",
        explicit_model=False,
    ) == "grok-imagine-video-1.5"
    assert _resolve_model_for_modality(
        "grok-imagine-video-1.5-preview",
        modality="text",
        explicit_model=False,
    ) == "grok-imagine-video"
    assert _resolve_model_for_modality(
        "grok-imagine-video-1.5-preview",
        modality="text",
        explicit_model=True,
    ) == "grok-imagine-video-1.5-preview"


def test_xai_capabilities_keep_generate_surface_only():
    from plugins.video_gen.xai import XAIVideoGenProvider

    caps = XAIVideoGenProvider().capabilities()
    assert caps["modalities"] == ["text", "image"]
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
    assert result["error_type"] in {"auth_required", "api_error"}


def test_xai_video_output_urls_prefers_stored_public_url():
    from plugins.video_gen.xai import _xai_video_output_urls

    public_url, temporary, stored = _xai_video_output_urls({
        "url": "https://vidgen.x.ai/xai-vidgen-bucket/out.mp4",
        "file_output": {
            "public_url": "https://files-cdn.x.ai/token/file_abc.mp4",
            "file_id": "file_abc",
        },
    })
    assert public_url == "https://files-cdn.x.ai/token/file_abc.mp4"
    assert stored == "https://files-cdn.x.ai/token/file_abc.mp4"
    assert temporary == "https://vidgen.x.ai/xai-vidgen-bucket/out.mp4"


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


def test_video_input_from_public_url_rejects_bare_file_id():
    import asyncio
    from plugins.video_gen.xai import _video_input_from_public_url

    result = asyncio.run(
        _video_input_from_public_url(
            "file_1faca9c3-9411-46ad-bb41-b9b8527789e6",
            api_key="test-key",
            base_url="https://api.x.ai/v1",
        )
    )
    assert result is None


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


def test_xai_video_file_input_blocks_credential_store_symlink(tmp_path, monkeypatch):
    from plugins.video_gen.xai import _video_ref_to_xai_url

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    auth_json = hermes_home / "auth.json"
    auth_json.write_text('{"api_key":"sk-secret"}', encoding="utf-8")
    video_link = hermes_home / "leak.mp4"
    try:
        video_link.symlink_to(auth_json)
    except OSError as exc:
        pytest.skip(f"symlink unavailable on this platform: {exc}")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(ValueError, match="credential store"):
        _video_ref_to_xai_url(str(video_link))


import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx


def test_f5_submit_propagates_refreshed_bearer_to_poll():
    """F5: submit 401 → refresh → poll uses the NEW bearer."""
    from plugins.video_gen.xai import _submit, _poll

    stale = "stale-bearer"
    fresh = "fresh-bearer"
    poll_keys = []

    class FakeResponse:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload
            self.text = str(payload)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "err",
                    request=MagicMock(),
                    response=self,
                )

        def json(self):
            return self._payload

    submit_calls = {"n": 0}

    async def fake_post(url, **kwargs):
        submit_calls["n"] += 1
        headers = kwargs.get("headers") or {}
        authz = headers.get("Authorization") or headers.get("authorization") or ""
        if submit_calls["n"] == 1:
            assert stale in authz
            return FakeResponse(401, {"error": "unauthorized"})
        assert fresh in authz
        return FakeResponse(200, {"request_id": "req-1"})

    async def fake_get(url, **kwargs):
        headers = kwargs.get("headers") or {}
        authz = headers.get("Authorization") or headers.get("authorization") or ""
        poll_keys.append(authz)
        return FakeResponse(200, {"status": "done", "video": {"url": "https://x/v.mp4"}})

    client = MagicMock()
    client.post = fake_post
    client.get = fake_get

    def fake_refresh(rejected=None):
        assert rejected == stale
        return {"api_key": fresh, "base_url": "https://api.x.ai/v1"}

    async def run():
        with patch(
            "tools.xai_http.force_refresh_xai_http_credentials",
            side_effect=fake_refresh,
        ):
            request_id, active_key = await _submit(
                client,
                {"model": "grok-imagine-video", "prompt": "hi"},
                api_key=stale,
                base_url="https://api.x.ai/v1",
            )
            assert request_id == "req-1"
            assert active_key == fresh
            result = await _poll(
                client,
                request_id,
                api_key=active_key,
                base_url="https://api.x.ai/v1",
                timeout_seconds=5,
                poll_interval=1,
            )
            assert result["status"] == "done"

    asyncio.run(run())
    assert any(fresh in k for k in poll_keys)
    assert not any(stale in k for k in poll_keys)


def test_f5_poll_one_canonical_401_recovery():
    """F5: poll 401 → one canonical recovery then succeeds."""
    from plugins.video_gen.xai import _poll

    stale = "stale-poll"
    fresh = "fresh-poll"
    get_calls = {"n": 0}

    class FakeResponse:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload
            self.text = str(payload)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "err",
                    request=MagicMock(),
                    response=self,
                )

        def json(self):
            return self._payload

    async def fake_get(url, **kwargs):
        get_calls["n"] += 1
        headers = kwargs.get("headers") or {}
        authz = headers.get("Authorization") or ""
        if get_calls["n"] == 1:
            assert stale in authz
            return FakeResponse(401, {"error": "unauthorized"})
        assert fresh in authz
        return FakeResponse(200, {"status": "done", "video": {}})

    client = MagicMock()
    client.get = fake_get

    def fake_refresh(rejected=None):
        return {"api_key": fresh}

    async def run():
        with patch(
            "tools.xai_http.force_refresh_xai_http_credentials",
            side_effect=fake_refresh,
        ):
            result = await _poll(
                client,
                "req-2",
                api_key=stale,
                base_url="https://api.x.ai/v1",
                timeout_seconds=5,
                poll_interval=1,
            )
            assert result["status"] == "done"

    asyncio.run(run())
    assert get_calls["n"] == 2
