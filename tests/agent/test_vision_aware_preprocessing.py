"""Tests for the vision-aware image preprocessing in run_agent.py.

Covers:

* ``_prepare_anthropic_messages_for_api`` — passes image parts through
  unchanged when the active model reports ``supports_vision=True`` (the
  adapter handles them natively), and falls back to text-description
  replacement when the model lacks vision.

* ``_prepare_messages_for_non_vision_model`` — the mirror method for the
  chat.completions / codex_responses paths. Same contract.
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import patch


from run_agent import AIAgent


def _make_agent() -> AIAgent:
    """Build a bare-bones AIAgent instance without running __init__.

    Avoids the heavy provider/credential setup for these pure-method tests.
    """
    agent = object.__new__(AIAgent)
    agent.provider = "anthropic"
    agent.model = "claude-sonnet-4"
    agent._anthropic_image_fallback_cache = {}
    return agent


IMG_PARTS_USER_MSG = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ],
}

PLAIN_USER_MSG = {"role": "user", "content": "hello, no images here"}


# ─── _prepare_anthropic_messages_for_api ─────────────────────────────────────


class TestPrepareAnthropicMessages:
    def test_no_images_passes_through(self):
        agent = _make_agent()
        msgs = [PLAIN_USER_MSG]
        out = agent._prepare_anthropic_messages_for_api(msgs)
        assert out is msgs  # unchanged reference

    def test_vision_capable_passes_images_through(self):
        """The Anthropic adapter handles image_url/input_image natively."""
        agent = _make_agent()
        with patch.object(agent, "_model_supports_vision", return_value=True):
            out = agent._prepare_anthropic_messages_for_api([IMG_PARTS_USER_MSG])
        # Passes through unchanged — image_url parts still present.
        assert out[0]["content"][1]["type"] == "image_url"

    def test_non_vision_replaces_images_with_text(self):
        agent = _make_agent()
        with patch.object(agent, "_model_supports_vision", return_value=False), \
             patch.object(
                 agent,
                 "_describe_image_for_anthropic_fallback",
                 return_value="[Image description: a cat]",
             ):
            out = agent._prepare_anthropic_messages_for_api([IMG_PARTS_USER_MSG])
        # Content collapsed to a string containing the description + user text.
        content = out[0]["content"]
        assert isinstance(content, str)
        assert "[Image description: a cat]" in content
        assert "What's in this image?" in content
        # No more image parts.
        assert "image_url" not in content


# ─── _prepare_messages_for_non_vision_model ──────────────────────────────────


class TestPrepareMessagesForNonVision:
    def test_no_images_passes_through(self):
        agent = _make_agent()
        msgs = [PLAIN_USER_MSG]
        out = agent._prepare_messages_for_non_vision_model(msgs)
        assert out is msgs

    def test_vision_capable_passes_through(self):
        """For vision-capable models on chat.completions path, provider handles pixels."""
        agent = _make_agent()
        agent.provider = "openrouter"
        agent.model = "anthropic/claude-sonnet-4"
        with patch.object(agent, "_model_supports_vision", return_value=True):
            out = agent._prepare_messages_for_non_vision_model([IMG_PARTS_USER_MSG])
        assert out[0]["content"][1]["type"] == "image_url"

    def test_non_vision_strips_images(self):
        agent = _make_agent()
        agent.provider = "openrouter"
        agent.model = "qwen/qwen3-235b-a22b"
        with patch.object(agent, "_model_supports_vision", return_value=False), \
             patch.object(
                 agent,
                 "_describe_image_for_anthropic_fallback",
                 return_value="[Image description: a dog]",
             ):
            out = agent._prepare_messages_for_non_vision_model([IMG_PARTS_USER_MSG])
        content = out[0]["content"]
        assert isinstance(content, str)
        assert "[Image description: a dog]" in content
        assert "image_url" not in content

    def test_multiple_messages_with_mixed_content(self):
        agent = _make_agent()
        agent.model = "qwen/qwen3-235b"
        msgs = [
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": "ack"},
            IMG_PARTS_USER_MSG,
        ]
        with patch.object(agent, "_model_supports_vision", return_value=False), \
             patch.object(
                 agent,
                 "_describe_image_for_anthropic_fallback",
                 return_value="[Image: thing]",
             ):
            out = agent._prepare_messages_for_non_vision_model(msgs)
        # First two messages unchanged (no images), third stripped.
        assert out[0]["content"] == "first turn"
        assert out[1]["content"] == "ack"
        assert isinstance(out[2]["content"], str)
        assert "[Image: thing]" in out[2]["content"]


# ─── _model_supports_vision ──────────────────────────────────────────────────


class TestModelSupportsVision:
    def test_missing_provider_or_model_returns_false(self):
        agent = _make_agent()
        agent.provider = ""
        agent.model = "claude-sonnet-4"
        assert agent._model_supports_vision() is False
        agent.provider = "anthropic"
        agent.model = ""
        assert agent._model_supports_vision() is False


    def test_none_caps_returns_false(self):
        agent = _make_agent()
        with patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert agent._model_supports_vision() is False


    def test_top_level_model_override_wins(self):
        agent = _make_agent()
        agent.provider = "custom"
        agent.model = "my-llava"
        with patch("hermes_cli.config.load_config", return_value={"model": {"supports_vision": True}}), \
             patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert agent._model_supports_vision() is True


    def test_named_custom_provider_resolved_via_config_provider(self):
        # Named custom providers get runtime self.provider rewritten to
        # "custom" while the config keeps the original name under
        # model.provider. The override must still resolve.
        agent = _make_agent()
        agent.provider = "custom"
        agent.model = "my-llava"
        cfg = {
            "model": {"provider": "my-vllm", "default": "my-llava"},
            "providers": {"my-vllm": {"models": {"my-llava": {"supports_vision": True}}}},
        }
        with patch("hermes_cli.config.load_config", return_value=cfg), \
             patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert agent._model_supports_vision() is True



# ─── oversized inline images: downscale before giving up (#132605 follow-up) ──


def _noise_jpeg(width: int, height: int, quality: int = 95) -> bytes:
    """Random pixels compress badly, so this lands well over the byte cap."""
    import io
    import os

    from PIL import Image

    img = Image.frombytes("RGB", (width, height), os.urandom(width * height * 3))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class TestOversizedDataUrlDownscale:
    """A 20-64 MB inline image should be shrunk, not silently dropped."""

    def test_oversized_payload_is_downscaled_instead_of_skipped(self, monkeypatch):
        from agent.vision_message_prep import VisionMessagePrepMixin

        raw = _noise_jpeg(3000, 2000)
        payload = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
        cap = 1_000_000
        monkeypatch.setattr(VisionMessagePrepMixin, "_MAX_DATA_URL_BASE64_BYTES", cap)
        assert len(payload.partition(",")[2]) > cap

        path, cleanup = VisionMessagePrepMixin._materialize_data_url_for_vision(payload)
        try:
            assert path, "oversized image should be materialized after downscaling"
            out = Path(path)
            assert out.suffix == ".jpg"
            assert out.stat().st_size < cap
        finally:
            if cleanup is not None and cleanup.exists():
                cleanup.unlink()

    def test_payload_past_hard_cap_is_still_skipped(self, monkeypatch):
        from agent.vision_message_prep import VisionMessagePrepMixin

        monkeypatch.setattr(VisionMessagePrepMixin, "_MAX_DATA_URL_BASE64_BYTES", 1_024)
        monkeypatch.setattr(VisionMessagePrepMixin, "_MAX_DATA_URL_DECODE_BYTES", 2_048)
        payload = "data:image/png;base64," + base64.b64encode(b"x" * 4_096).decode("ascii")
        assert VisionMessagePrepMixin._materialize_data_url_for_vision(payload) == ("", None)

    def test_pixel_bomb_is_refused(self, monkeypatch):
        from agent import vision_message_prep as vmp

        monkeypatch.setattr(vmp.VisionMessagePrepMixin, "_MAX_DATA_URL_BASE64_BYTES", 1_024)
        monkeypatch.setattr(vmp.VisionMessagePrepMixin, "_MAX_DATA_URL_DECODE_BYTES", 10 ** 9)

        class _FakeProbe:
            size = (12_000, 12_000)
            is_animated = False

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        encoded = base64.b64encode(b"x" * 4_096).decode("ascii")
        with patch("PIL.Image.open", return_value=_FakeProbe()):
            assert vmp.VisionMessagePrepMixin._decode_oversized_data_url(encoded, "image/png") is None

    def test_non_image_data_url_is_not_downscaled(self, monkeypatch):
        from agent.vision_message_prep import VisionMessagePrepMixin

        monkeypatch.setattr(VisionMessagePrepMixin, "_MAX_DATA_URL_BASE64_BYTES", 1_024)
        monkeypatch.setattr(VisionMessagePrepMixin, "_MAX_DATA_URL_DECODE_BYTES", 10 ** 9)
        encoded = base64.b64encode(b"%PDF-1.7" + b"x" * 4_096).decode("ascii")
        assert VisionMessagePrepMixin._decode_oversized_data_url(encoded, "application/pdf") is None
