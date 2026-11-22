"""Tests for the bundled DeepInfra video_gen plugin.

Invariants only — no snapshots of specific model ids. The plugin is a thin
subclass of ``agent.video_gen_provider.OpenAICompatibleVideoGenProvider``;
these tests pin the plugin-specific bits (tag filtering, identity) and the
shared base behaviour exercised through it (OpenAI ``videos`` call shape,
t2v vs i2v routing, download → save).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import plugins.video_gen.deepinfra as deepinfra_plugin


@pytest.fixture(autouse=True)
def _isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_cli.models as _models_mod
    monkeypatch.setattr(_models_mod, "_deepinfra_catalog_cache", {})
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    yield


def test_availability_follows_api_key(monkeypatch):
    p = deepinfra_plugin.DeepInfraVideoGenProvider()
    assert p.is_available() is True
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    assert p.is_available() is False


def _fake_openai_with_capture(captured: dict, *, status="succeeded",
                               data=None, download=b"\x00\x00mp4bytes"):
    """Build a fake ``openai`` module whose videos resource records the call.

    Defaults mirror the real DeepInfra job shape: status ``"succeeded"`` and a
    ``data`` list carrying the delivery URL.
    """
    if data is None:
        data = [{"url": "https://cdn.example/out.mp4"}]

    class _FakeVideos:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            # Return a terminal status immediately so the bounded poll in
            # OpenAICompatibleVideoGenProvider._create_and_poll exits without
            # calling retrieve() or sleeping.
            return SimpleNamespace(status=status, id="vid_123", error=None, data=data)

        def retrieve(self, video_id):
            return SimpleNamespace(status=status, id=video_id, error=None, data=data)

        def download_content(self, video_id):
            captured["downloaded_id"] = video_id
            return SimpleNamespace(read=lambda: download)

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None, http_client=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["http_client"] = http_client
            self.videos = _FakeVideos()

    fake = MagicMock()
    fake.OpenAI = _FakeClient
    return fake


@contextmanager
def _mock_url_download(captured: dict, raise_exc: Exception | None = None):
    """Patch the shared ``save_url_video`` helper the base provider calls."""
    import agent.video_gen_provider as base
    from pathlib import Path

    def _fake_save_url_video(url, *, prefix="video", **kw):
        captured["url"] = url
        if raise_exc:
            raise raise_exc
        return Path(f"/home/x/.hermes/cache/videos/{prefix}_test.mp4")

    with patch.object(base, "save_url_video", _fake_save_url_video):
        yield


def test_generate_uses_env_only_proxy_http_client(monkeypatch):
    """The SDK client is built on Hermes' env-only-proxy httpx client: a macOS system proxy (seen by
    httpx via ``getproxies()``, ExceptionsList dropped) must not be mounted for a custom endpoint
    (#64888), unlike a plain ``httpx.Client()`` under the same conditions (control)."""
    import httpx
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy",
                "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEEPINFRA_BASE_URL", "http://localhost:18081/v1")
    sys_proxy = {"http": "http://sysproxy:3128", "https": "http://sysproxy:3128"}

    def proxy_mounts(client):
        return [m for m in client._mounts.values() if type(getattr(m, "_pool", None)).__name__ == "HTTPProxy"]

    captured: dict = {}
    with patch("httpx._utils.getproxies", return_value=sys_proxy), \
            patch.dict("sys.modules", {"openai": _fake_openai_with_capture(captured)}), \
            _mock_url_download(captured):
        with httpx.Client() as control:
            assert len(proxy_mounts(control)) == 2
        assert deepinfra_plugin.DeepInfraVideoGenProvider().generate(prompt="a cube", model="vendor/x")["success"]
    assert captured["base_url"] == "http://localhost:18081/v1"
    assert isinstance(captured["http_client"], httpx.Client) and proxy_mounts(captured["http_client"]) == []
    captured["http_client"].close()


def test_generate_text_to_video_downloads_url_and_saves_locally():
    """t2v happy path: SDK called with DeepInfra base_url + key; status
    'succeeded' + data[].url → bytes downloaded and saved to a local file."""
    captured: dict = {}
    with patch.dict("sys.modules", {"openai": _fake_openai_with_capture(captured)}), \
            _mock_url_download(captured):
        result = deepinfra_plugin.DeepInfraVideoGenProvider().generate(
            prompt="a red cube rotating", model="vendor/test-vid", duration=5,
        )
    assert result["success"] is True
    assert result["modality"] == "text"
    assert result["video"].endswith(".mp4") and str(Path("cache") / "videos") in result["video"]
    assert captured["url"] == "https://cdn.example/out.mp4"
    assert "deepinfra" in captured["base_url"]
    assert captured["api_key"] == "test-key"
    assert captured["kwargs"]["model"] == "vendor/test-vid"
    assert captured["kwargs"]["seconds"] == "5"
    # No image_url ⇒ no image-to-video field passed through.
    assert "image_url" not in captured["kwargs"].get("extra_body", {})


def test_credentials_follow_the_profile_secret_scope(monkeypatch):
    """On a multiplexed gateway os.environ is the launch profile's .env: the key and base URL come from the
    routed profile's scope, and a profile without a key is unavailable instead of borrowing the launch key."""
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope

    monkeypatch.setenv("DEEPINFRA_BASE_URL", "https://launch.example/v1")
    provider = deepinfra_plugin.DeepInfraVideoGenProvider()
    captured: dict = {}
    set_multiplex_active(True)
    try:
        token = set_secret_scope({"DEEPINFRA_API_KEY": "profile-b-key", "DEEPINFRA_BASE_URL": "https://profile-b.example/v1"})
        try:
            with patch.dict("sys.modules", {"openai": _fake_openai_with_capture(captured)}), \
                    _mock_url_download(captured):
                assert provider.generate(prompt="a cube", model="vendor/x")["success"]
        finally:
            reset_secret_scope(token)
        token = set_secret_scope({})
        try:
            assert provider.is_available() is False
        finally:
            reset_secret_scope(token)
    finally:
        set_multiplex_active(False)
        if captured.get("http_client") is not None:
            captured["http_client"].close()
    assert captured["api_key"] == "profile-b-key"
    assert captured["base_url"] == "https://profile-b.example/v1"
