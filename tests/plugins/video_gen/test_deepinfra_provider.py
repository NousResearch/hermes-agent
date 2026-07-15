"""Tests for the bundled DeepInfra video_gen plugin.

Invariants only — no snapshots of specific model ids. The plugin is a thin
subclass of ``agent.video_gen_provider.OpenAICompatibleVideoGenProvider``;
these tests pin the plugin-specific bits (tag filtering, identity) and the
shared base behaviour exercised through it (OpenAI ``videos`` call shape,
t2v vs i2v routing, download → save).
"""

from __future__ import annotations

from contextlib import contextmanager
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


def test_identity_and_availability(monkeypatch):
    p = deepinfra_plugin.DeepInfraVideoGenProvider()
    assert p.name == "deepinfra"
    assert p.display_name == "DeepInfra"
    assert p._base_url() == "https://api.deepinfra.com/v1/openai"
    assert p.is_available() is True
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    assert p.is_available() is False


def test_list_models_filters_by_video_gen_tag(monkeypatch):
    """list_models() returns only ``video-gen``-tagged catalog entries."""
    import hermes_cli.models as _models_mod

    def _fake_by_tag(tag, **kw):
        assert tag == "video-gen"
        return [
            {"id": "vendor/p-video", "metadata": {"description": "fast t2v"}},
            {"id": "vendor/wan-t2v", "metadata": {}},
        ]

    monkeypatch.setattr(_models_mod, "_fetch_deepinfra_models_by_tag", _fake_by_tag)
    rows = deepinfra_plugin.DeepInfraVideoGenProvider().list_models()
    ids = {row["id"] for row in rows}
    assert ids == {"vendor/p-video", "vendor/wan-t2v"}
    assert all("display" in r for r in rows)


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
        def __init__(self, api_key=None, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
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
    assert result["video"].endswith(".mp4") and "cache/videos" in result["video"]
    assert captured["url"] == "https://cdn.example/out.mp4"
    assert "deepinfra" in captured["base_url"]
    assert captured["api_key"] == "test-key"
    assert captured["kwargs"]["model"] == "vendor/test-vid"
    assert captured["kwargs"]["seconds"] == "5"
    # No image_url ⇒ no image-to-video field passed through.
    assert "image_url" not in captured["kwargs"].get("extra_body", {})


def test_generate_returns_url_when_local_save_fails():
    """If downloading the delivery URL fails, fall back to returning the URL."""
    captured: dict = {}
    with patch.dict("sys.modules", {"openai": _fake_openai_with_capture(captured)}), \
            _mock_url_download(captured, raise_exc=OSError("network down")):
        result = deepinfra_plugin.DeepInfraVideoGenProvider().generate(
            prompt="x", model="vendor/test-vid",
        )
    assert result["success"] is True
    assert result["video"] == "https://cdn.example/out.mp4"


def test_generate_falls_back_to_download_when_no_url():
    """OpenAI/Sora style: no data[].url → download_content bytes saved locally."""
    captured: dict = {}
    fake = _fake_openai_with_capture(captured, status="completed", data=[])
    with patch.dict("sys.modules", {"openai": fake}):
        result = deepinfra_plugin.DeepInfraVideoGenProvider().generate(
            prompt="x", model="vendor/test-vid",
        )
    assert result["success"] is True
    assert captured["downloaded_id"] == "vid_123"
    assert result["video"].endswith(".mp4")


def test_generate_image_to_video_routes_via_extra_body():
    """Presence of image_url routes to i2v and rides in extra_body."""
    captured: dict = {}
    with patch.dict("sys.modules", {"openai": _fake_openai_with_capture(captured)}), \
            _mock_url_download(captured):
        result = deepinfra_plugin.DeepInfraVideoGenProvider().generate(
            prompt="animate this", model="vendor/test-vid",
            image_url="https://example.com/cat.jpg", negative_prompt="blurry",
        )
    assert result["success"] is True
    assert result["modality"] == "image"
    extra = captured["kwargs"]["extra_body"]
    assert extra["image_url"] == "https://example.com/cat.jpg"
    assert extra["negative_prompt"] == "blurry"


def test_generate_errors_when_key_missing(monkeypatch):
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    result = deepinfra_plugin.DeepInfraVideoGenProvider().generate(
        prompt="x", model="vendor/test-vid",
    )
    assert result["success"] is False
    assert result["error_type"] == "missing_credentials"


def test_generate_errors_when_job_not_completed():
    """A non-completed job status surfaces a JSON-serializable job_failed error.

    ``video.error`` is a structured SDK object (pydantic ``VideoCreateError``),
    not a string — the provider must str() it so the response dict survives the
    tool layer's ``json.dumps``. We simulate that with a non-serializable object.
    """
    import json

    captured: dict = {}
    fake = _fake_openai_with_capture(captured)

    class _NonSerializableError:
        def __str__(self):
            return "content policy violation"

    class _FailingVideos:
        def create(self, **kwargs):
            return SimpleNamespace(
                status="failed", id="vid_x", error=_NonSerializableError(), data=None
            )

        def retrieve(self, video_id):  # pragma: no cover - status already terminal
            return SimpleNamespace(status="failed", id=video_id, error=None, data=None)

    def _client(api_key=None, base_url=None):
        return SimpleNamespace(videos=_FailingVideos())

    fake.OpenAI = _client
    with patch.dict("sys.modules", {"openai": fake}):
        result = deepinfra_plugin.DeepInfraVideoGenProvider().generate(
            prompt="x", model="vendor/test-vid",
        )
    assert result["success"] is False
    assert result["error_type"] == "job_failed"
    assert "content policy violation" in result["error"]
    # Must not raise — this is the regression the str() guard prevents.
    json.dumps(result)
