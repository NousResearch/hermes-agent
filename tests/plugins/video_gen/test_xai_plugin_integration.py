"""Integration tests for the xAI video gen plugin's simplified surface.

xAI exposes only text-to-video and image-to-video through the unified
``video_generate`` tool. We assert the endpoint hit and the payload shape
because routing is the part most likely to break silently.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest

from agent import video_gen_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    video_gen_registry._reset_for_tests()
    yield
    video_gen_registry._reset_for_tests()


class _FakeResponse:
    def __init__(
        self,
        status: int = 200,
        payload: Optional[Dict[str, Any]] = None,
        body_chunks: Optional[List[bytes]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.status_code = status
        self._payload = payload or {}
        self._body_chunks = body_chunks
        self.headers = headers or {}
        self.text = json.dumps(self._payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("err", request=None, response=self)  # type: ignore

    def json(self):
        return self._payload

    async def aiter_bytes(self, chunk_size=None):
        chunks = self._body_chunks
        if chunks is None:
            chunks = [self.text.encode("utf-8")]
        for chunk in chunks:
            yield chunk


class _FakeStream:
    def __init__(self, response: _FakeResponse):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return None


class _FakeAsyncClient:
    def __init__(self):
        self.posts: List[Dict[str, Any]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append({"url": url, "json": json})
        return _FakeResponse(200, {"request_id": "req-123"})

    async def get(self, url, headers=None, timeout=None):
        return _FakeResponse(200, {
            "status": "done",
            "video": {"url": "https://xai-cdn/out.mp4", "duration": 8},
            "model": self.posts[-1]["json"]["model"],
        })

    def stream(self, method, url, headers=None, json=None, timeout=None):
        if method == "POST":
            self.posts.append({"url": url, "json": json})
            return _FakeStream(_FakeResponse(200, {"request_id": "req-123"}))
        return _FakeStream(_FakeResponse(200, {
            "status": "done",
            "video": {"url": "https://xai-cdn/out.mp4", "duration": 8},
            "model": self.posts[-1]["json"]["model"],
        }))


@pytest.fixture
def xai_provider(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    import plugins.video_gen.xai as xai_plugin

    captured: Dict[str, _FakeAsyncClient] = {}

    def _client_factory():
        captured["client"] = _FakeAsyncClient()
        return captured["client"]

    monkeypatch.setattr(xai_plugin.httpx, "AsyncClient", _client_factory)

    async def _no_sleep(*a, **k):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    provider = xai_plugin.XAIVideoGenProvider()
    return provider, captured


def _last_post(captured) -> Dict[str, Any]:
    return captured["client"].posts[-1]


class TestXAIEndpoint:
    """xAI uses one endpoint — ``/videos/generations`` — for both modes."""

    def test_text_to_video_hits_generations(self, xai_provider):
        provider, captured = xai_provider
        result = provider.generate("a dog on a skateboard")
        assert result["success"] is True
        assert _last_post(captured)["url"].endswith("/videos/generations")
        assert result["modality"] == "text"

    def test_image_to_video_hits_generations(self, xai_provider):
        provider, captured = xai_provider
        result = provider.generate(
            "animate this",
            image_url="https://example.com/cat.png",
        )
        assert result["success"] is True
        assert _last_post(captured)["url"].endswith("/videos/generations")
        assert result["modality"] == "image"

    @pytest.mark.asyncio
    async def test_submit_rejects_oversized_content_length(self, monkeypatch):
        import plugins.video_gen.xai as xai_plugin

        monkeypatch.setattr(xai_plugin, "_XAI_VIDEO_RESPONSE_MAX_BYTES", 8)

        class _Client:
            def stream(self, *args, **kwargs):
                return _FakeStream(_FakeResponse(
                    200,
                    body_chunks=[b"{}"],
                    headers={"content-length": "9"},
                ))

        with pytest.raises(ValueError, match="xAI video response exceeds 8 bytes"):
            await xai_plugin._submit(
                _Client(),  # type: ignore[arg-type]
                {"model": "grok-imagine-video", "prompt": "dog"},
                api_key="key",
                base_url="https://api.x.ai/v1",
            )

    @pytest.mark.asyncio
    async def test_poll_rejects_oversized_streamed_body(self, monkeypatch):
        import plugins.video_gen.xai as xai_plugin

        monkeypatch.setattr(xai_plugin, "_XAI_VIDEO_RESPONSE_MAX_BYTES", 8)

        class _Client:
            def stream(self, *args, **kwargs):
                return _FakeStream(_FakeResponse(200, body_chunks=[b"x" * 9]))

        with pytest.raises(ValueError, match="xAI video response exceeds 8 bytes"):
            await xai_plugin._poll(
                _Client(),  # type: ignore[arg-type]
                "req-123",
                api_key="key",
                base_url="https://api.x.ai/v1",
                timeout_seconds=5,
                poll_interval=1,
            )

    def test_generate_reports_oversized_submit_response(self, monkeypatch):
        import plugins.video_gen.xai as xai_plugin

        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr(xai_plugin, "_XAI_VIDEO_RESPONSE_MAX_BYTES", 8)

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            def stream(self, *args, **kwargs):
                return _FakeStream(_FakeResponse(
                    200,
                    body_chunks=[b"{}"],
                    headers={"content-length": "9"},
                ))

        monkeypatch.setattr(xai_plugin.httpx, "AsyncClient", lambda: _Client())

        result = xai_plugin.XAIVideoGenProvider().generate("a dog on a skateboard")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "xAI submit failed: xAI video response exceeds 8 bytes" in result["error"]

    def test_generate_reports_oversized_poll_response(self, monkeypatch):
        import plugins.video_gen.xai as xai_plugin

        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.setattr(xai_plugin, "_XAI_VIDEO_RESPONSE_MAX_BYTES", 32)

        class _Client:
            def __init__(self):
                self._calls = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            def stream(self, method, *args, **kwargs):
                self._calls += 1
                if method == "POST":
                    return _FakeStream(_FakeResponse(200, {"request_id": "req-123"}))
                return _FakeStream(_FakeResponse(200, body_chunks=[b"x" * 33]))

        monkeypatch.setattr(xai_plugin.httpx, "AsyncClient", lambda: _Client())

        result = xai_plugin.XAIVideoGenProvider().generate("a dog on a skateboard")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "xAI poll failed: xAI video response exceeds 32 bytes" in result["error"]


class TestXAIPayload:
    def test_text_payload_has_no_image_field(self, xai_provider):
        provider, captured = xai_provider
        provider.generate("a dog at sunset")
        payload = _last_post(captured)["json"]
        assert payload["model"] == "grok-imagine-video"
        assert payload["prompt"] == "a dog at sunset"
        assert "image" not in payload
        assert "reference_images" not in payload

    def test_image_payload_has_image_field(self, xai_provider):
        provider, captured = xai_provider
        provider.generate("animate this", image_url="https://example.com/cat.png")
        payload = _last_post(captured)["json"]
        assert payload["model"] == "grok-imagine-video-1.5"
        assert payload["image"] == {"url": "https://example.com/cat.png"}

    def test_local_image_path_is_sent_as_data_uri(self, xai_provider, tmp_path):
        provider, captured = xai_provider
        image_path = tmp_path / "frame.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

        provider.generate("animate this", image_url=str(image_path))

        payload = _last_post(captured)["json"]
        assert payload["model"] == "grok-imagine-video-1.5"
        assert payload["image"]["url"].startswith("data:image/png;base64,")

    def test_explicit_model_override_is_honored_for_image(self, xai_provider):
        provider, captured = xai_provider
        provider.generate(
            "animate this",
            image_url="https://example.com/cat.png",
            model="grok-imagine-video",
            _model_override_explicit=True,
        )
        payload = _last_post(captured)["json"]
        assert payload["model"] == "grok-imagine-video"

    def test_reference_images_payload(self, xai_provider):
        provider, captured = xai_provider
        provider.generate(
            "keep this character",
            reference_image_urls=[
                "https://example.com/a.png",
                "https://example.com/b.png",
            ],
        )
        payload = _last_post(captured)["json"]
        assert payload["reference_images"] == [
            {"url": "https://example.com/a.png"},
            {"url": "https://example.com/b.png"},
        ]


class TestXAIValidation:
    def test_missing_prompt_rejects(self, xai_provider):
        provider, captured = xai_provider
        result = provider.generate("")
        assert result["success"] is False
        assert result["error_type"] == "missing_prompt"
        # Never hit the network
        assert "client" not in captured or not captured["client"].posts

    def test_image_plus_refs_rejects(self, xai_provider):
        provider, captured = xai_provider
        result = provider.generate(
            "x",
            image_url="https://example.com/i.png",
            reference_image_urls=["https://example.com/r.png"],
        )
        assert result["success"] is False
        assert result["error_type"] == "conflicting_inputs"
        assert "client" not in captured or not captured["client"].posts

    def test_too_many_references_rejects(self, xai_provider):
        provider, captured = xai_provider
        result = provider.generate(
            "x",
            reference_image_urls=[f"https://example.com/r{i}.png" for i in range(8)],
        )
        assert result["success"] is False
        assert result["error_type"] == "too_many_references"


class TestXAIClamping:
    def test_duration_clamped_to_15(self, xai_provider):
        provider, captured = xai_provider
        provider.generate("x", duration=30)
        assert _last_post(captured)["json"]["duration"] == 15

    def test_duration_clamped_when_refs_present(self, xai_provider):
        provider, captured = xai_provider
        provider.generate(
            "x",
            duration=15,
            reference_image_urls=["https://example.com/r.png"],
        )
        # refs present caps to 10
        assert _last_post(captured)["json"]["duration"] == 10

    def test_invalid_aspect_ratio_soft_clamps(self, xai_provider):
        provider, captured = xai_provider
        provider.generate("x", aspect_ratio="21:9")
        assert _last_post(captured)["json"]["aspect_ratio"] == "16:9"
