"""Smoke tests for the xAI video gen plugin — load & register surface."""

from __future__ import annotations

import pytest

from agent import video_gen_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    video_gen_registry._reset_for_tests()
    yield
    video_gen_registry._reset_for_tests()


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


