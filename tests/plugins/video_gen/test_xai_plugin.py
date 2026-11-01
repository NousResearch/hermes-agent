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




def test_run_xai_video_offloads_when_event_loop_running(monkeypatch):
    """Regression: calling _run_xai_video from inside a running event loop (gateway async context)
    must offload to a worker thread instead of raising 'Cannot run the event loop while another loop is running'."""
    import asyncio

    from plugins.video_gen import xai as xai_plugin

    monkeypatch.setattr(xai_plugin, "_resolve_xai_credentials", lambda: ("test-key", "https://api.x.ai/v1"))

    seen = {}

    async def flow(**kwargs):
        seen["api_key"] = kwargs["api_key"]
        return {"ok": True, "video": "https://example.com/v.mp4"}

    async def outer():
        # simulating the gateway: generate() called in the middle of a running loop
        return xai_plugin._run_xai_video("generation", flow, prompt="p", model=None)

    result = asyncio.run(outer())
    assert result["ok"] is True
    assert seen["api_key"] == "test-key"


def test_run_xai_video_direct_when_no_loop_running(monkeypatch):
    """Without a running loop, keep the original direct path (no thread hop)."""
    import asyncio

    from plugins.video_gen import xai as xai_plugin

    monkeypatch.setattr(xai_plugin, "_resolve_xai_credentials", lambda: ("test-key", "https://api.x.ai/v1"))
    loops = []

    async def flow(**kwargs):
        loops.append(asyncio.get_running_loop())
        return {"ok": True}

    result = xai_plugin._run_xai_video("generation", flow, prompt="p", model=None)
    assert result["ok"] is True
    assert len(loops) == 1  # the flow ran on its own fresh loop (closed after run_until_complete, expected)
