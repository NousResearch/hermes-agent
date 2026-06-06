"""Tests for the base image generation provider edit contract."""

from __future__ import annotations

from typing import Any, Dict

from agent.image_gen_provider import ImageGenProvider


class GenerateOnlyProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "generate_only"

    def generate(self, prompt: str, aspect_ratio: str = "landscape", **kwargs: Any) -> Dict[str, Any]:
        return {"success": True, "prompt": prompt, "aspect_ratio": aspect_ratio}


def test_generate_only_provider_can_instantiate_without_edit_override():
    provider = GenerateOnlyProvider()

    assert provider.name == "generate_only"
    assert provider.supports_edit() is False


def test_default_edit_returns_structured_unsupported_response():
    provider = GenerateOnlyProvider()

    result = provider.edit(
        "make it brighter",
        images=["https://example.test/source.png"],
        image="https://example.test/alias.png",
        mask="https://example.test/mask.png",
        aspect_ratio="square",
        model="future-model",
        quality_tier="high",
        ignored_future_kwarg=True,
    )

    assert result["success"] is False
    assert result["image"] is None
    assert result["error_type"] == "unsupported"
    assert "does not support image editing" in result["error"]
    assert result["provider"] == "generate_only"
    assert result["prompt"] == "make it brighter"
    assert result["aspect_ratio"] == "square"
    assert result["model"] == "future-model"


def test_default_edit_clamps_invalid_aspect_ratio():
    result = GenerateOnlyProvider().edit("prompt", image="source", aspect_ratio="wide")

    assert result["aspect_ratio"] == "landscape"


def test_default_edit_accepts_no_reference_images():
    result = GenerateOnlyProvider().edit("prompt")

    assert result["success"] is False
    assert result["error_type"] == "unsupported"
    assert result["model"] == ""


def test_default_edit_accepts_none_and_empty_edge_values():
    result = GenerateOnlyProvider().edit(
        "prompt",
        images=None,
        image=None,
        mask=None,
        aspect_ratio="",
        model=None,
        quality_tier=None,
        ignored_future_kwarg=None,
    )

    assert result["success"] is False
    assert result["aspect_ratio"] == "landscape"
    assert result["model"] == ""


def test_default_edit_accepts_empty_images_and_extra_kwargs():
    result = GenerateOnlyProvider().edit(
        "prompt",
        images=[],
        aspect_ratio=None,
        nested_future_kwarg={"x": object()},
    )

    assert result["success"] is False
    assert result["aspect_ratio"] == "landscape"
