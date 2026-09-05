from __future__ import annotations

from dataclasses import dataclass

import pytest

from tools.video_generation_tool import VIDEO_GENERATE_SCHEMA
from agent import video_gen_registry
from plugins.video_gen.openrouter import (
    DEFAULT_MODEL,
    OpenRouterVideoGenProvider,
    _build_payload,
)


@dataclass
class _Response:
    payload: dict
    status_code: int = 200
    content: bytes = b""

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.payload}")


class _Session:
    def __init__(self):
        self.posts = []
        self.gets = []
        self.polls = [
            _Response({"id": "job-1", "status": "in_progress"}),
            _Response(
                {
                    "id": "job-1",
                    "status": "completed",
                    "unsigned_urls": ["https://cdn.example/video.mp4"],
                    "usage": {"cost": 0.4},
                }
            ),
        ]

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return _Response(
            {
                "id": "job-1",
                "polling_url": "https://openrouter.ai/api/v1/videos/job-1",
                "status": "pending",
            },
            status_code=202,
        )

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        return self.polls.pop(0)


def test_poll_caps_request_timeout_and_rejects_late_terminal_response(monkeypatch):
    provider = OpenRouterVideoGenProvider()
    provider._poll_deadline_s = 10
    provider._request_timeout_s = 60
    session = _Session()
    session.polls = [_Response({"id": "job-1", "status": "completed"})]
    ticks = iter([100.0, 109.0, 111.0])
    monkeypatch.setattr("plugins.video_gen.openrouter.time.monotonic", lambda: next(ticks))

    with pytest.raises(TimeoutError, match="did not finish within 10s"):
        provider._poll(session, "job-1")

    assert session.gets[0][1]["timeout"] == 1.0


def test_unified_video_schema_exposes_hailuo_resolution_and_aspect_ratio():
    properties = VIDEO_GENERATE_SCHEMA["parameters"]["properties"]

    assert "768p" in properties["resolution"]["enum"]
    assert "21:9" in properties["aspect_ratio"]["enum"]


def test_hailuo_catalog_and_capabilities_are_pinned_to_openrouter_contract():
    provider = OpenRouterVideoGenProvider()

    assert DEFAULT_MODEL == "minimax/hailuo-3-max"
    assert provider.default_model() == DEFAULT_MODEL
    assert provider.list_models() == [
        {
            "id": DEFAULT_MODEL,
            "display": "MiniMax H3 Max",
            "speed": "~20-60s",
            "strengths": "Fast text-to-video and first-frame image-to-video.",
            "price": "$0.05/s (480p), $0.08/s (768p)",
            "modalities": ["text", "image"],
        }
    ]
    assert provider.capabilities() == {
        "modalities": ["text", "image"],
        "aspect_ratios": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
        "resolutions": ["480p", "768p"],
        "max_duration": 15,
        "min_duration": 5,
        "supports_audio": False,
        "supports_negative_prompt": False,
        "supports_seed": False,
        "supports_upscale": False,
        "max_reference_images": 0,
    }


def test_build_payload_uses_openrouter_video_fields_and_clamps_values():
    payload = _build_payload(
        prompt="A lighthouse in a storm",
        image_url="https://example.com/start.png",
        duration=99,
        aspect_ratio="2:3",
        resolution="720p",
    )

    assert payload == {
        "model": DEFAULT_MODEL,
        "prompt": "A lighthouse in a storm",
        "duration": 15,
        "resolution": "768p",
        "aspect_ratio": "16:9",
        "frame_images": [
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/start.png"},
                "frame_type": "first_frame",
            }
        ],
    }


def test_generate_submits_polls_and_materializes_completed_video(monkeypatch, tmp_path):
    provider = OpenRouterVideoGenProvider()
    session = _Session()
    saved = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(provider, "_session", lambda: session)
    monkeypatch.setattr("plugins.video_gen.openrouter.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "plugins.video_gen.openrouter.save_url_video",
        lambda url, prefix, headers, require_video_content_type: saved.append(
            (url, prefix, headers, require_video_content_type)
        )
        or tmp_path / "hailuo.mp4",
    )

    result = provider.generate(
        "A slow cinematic push-in",
        duration=5,
        aspect_ratio="9:16",
        resolution="480p",
    )

    assert result["success"] is True
    assert result["video"] == str(tmp_path / "hailuo.mp4")
    assert result["provider"] == "openrouter"
    assert result["model"] == DEFAULT_MODEL
    assert result["modality"] == "text"
    assert result["duration"] == 5
    assert result["cost"] == 0.4
    assert session.posts[0][0] == "https://openrouter.ai/api/v1/videos"
    assert session.posts[0][1]["json"]["resolution"] == "480p"
    assert [url for url, _kwargs in session.gets] == [
        "https://openrouter.ai/api/v1/videos/job-1",
        "https://openrouter.ai/api/v1/videos/job-1",
    ]
    assert saved == [
        (
            "https://openrouter.ai/api/v1/videos/job-1/content",
            "openrouter-hailuo",
            {
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
                "User-Agent": "hermes-agent/video_gen",
            },
            True,
        )
    ]


def test_generate_rejects_non_http_image_input_without_calling_api(monkeypatch):
    provider = OpenRouterVideoGenProvider()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    result = provider.generate("animate this", image_url="/private/start.png")

    assert result["success"] is False
    assert result["error_type"] == "invalid_request"
    assert "public HTTP" in result["error"]


def test_generate_rejects_private_first_frame_without_calling_api(monkeypatch):
    provider = OpenRouterVideoGenProvider()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        provider,
        "_session",
        lambda: (_ for _ in ()).throw(AssertionError("API must not be called")),
    )

    result = provider.generate("animate this", image_url="https://127.0.0.1/start.png")

    assert result["success"] is False
    assert result["error_type"] == "invalid_request"


def test_reference_images_are_rejected_before_session_creation(monkeypatch):
    provider = OpenRouterVideoGenProvider()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        provider,
        "_session",
        lambda: (_ for _ in ()).throw(AssertionError("API must not be called")),
    )

    result = provider.generate(
        "use these references",
        reference_image_urls=["https://example.com/reference.png"],
    )

    assert result["success"] is False
    assert result["error_type"] == "unsupported_input"


def test_generate_rejects_unknown_model_without_calling_api(monkeypatch):
    provider = OpenRouterVideoGenProvider()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    result = provider.generate("test", model="other/video-model")

    assert result["success"] is False
    assert result["error_type"] == "invalid_model"
    assert DEFAULT_MODEL in result["error"]


def test_register_and_picker_discovery_expose_openrouter(monkeypatch):
    from hermes_cli import tools_config
    from hermes_cli import plugins as plugin_loader
    from plugins.video_gen.openrouter import register

    class _Context:
        def register_video_gen_provider(self, provider):
            video_gen_registry.register_provider(provider)

    video_gen_registry._reset_for_tests()
    try:
        register(_Context())
        monkeypatch.setattr(plugin_loader, "_ensure_plugins_discovered", lambda: None)

        registered = video_gen_registry.get_provider("openrouter")
        rows = tools_config._plugin_video_gen_providers()

        assert isinstance(registered, OpenRouterVideoGenProvider)
        row = next(item for item in rows if item["video_gen_plugin_name"] == "openrouter")
        assert row["name"] == "OpenRouter"
        assert row["env_vars"][0]["key"] == "OPENROUTER_API_KEY"
    finally:
        video_gen_registry._reset_for_tests()


def test_dynamic_schema_is_capability_scoped(monkeypatch):
    from tools import video_generation_tool

    provider = OpenRouterVideoGenProvider()
    monkeypatch.setattr(video_generation_tool, "_resolve_active_provider", lambda: provider)
    monkeypatch.setattr(video_generation_tool, "_read_configured_video_model", lambda: DEFAULT_MODEL)

    schema = video_generation_tool._build_dynamic_video_schema()
    properties = schema["parameters"]["properties"]

    assert properties["duration"]["minimum"] == 5
    assert properties["duration"]["maximum"] == 15
    assert properties["resolution"]["enum"] == ["480p", "768p"]
    assert properties["aspect_ratio"]["enum"] == [
        "21:9", "16:9", "4:3", "1:1", "3:4", "9:16",
    ]
    assert "image_url" in properties
    assert "reference_image_urls" not in properties
    assert "audio" not in properties
    assert "seed" not in properties
