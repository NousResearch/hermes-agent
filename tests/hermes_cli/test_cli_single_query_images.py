"""Tests for the one-shot (``-q``) image-routing path in hermes_cli/cli_single_query.py."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.cli_single_query import _route_single_query_images


def _png_bytes() -> bytes:
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg=="
    )


def _cli() -> SimpleNamespace:
    return SimpleNamespace(provider="deepseek", model="deepseek-flash", requested_provider="")


def test_route_single_query_images_path_mode_marker_only_no_vision(monkeypatch, tmp_path):
    """mode=path: the one-shot path attaches local paths as marker text and makes
    no call to the hosted vision tool."""
    img = tmp_path / "a.png"
    img.write_bytes(_png_bytes())

    monkeypatch.setattr("agent.image_routing.decide_image_input_mode", lambda *a, **k: "path")
    monkeypatch.setattr("hermes_cli.config.load_config", dict)

    with patch("tools.vision_tools.vision_analyze_tool") as vision_tool:
        result = _route_single_query_images(_cli(), "what is this", "what is this", [img], [])

    assert result.startswith(
        f"[The user sent an image, not described automatically. Local path: {img}]"
    )
    assert "what is this" in result
    vision_tool.assert_not_called()


def test_route_single_query_images_path_mode_url_marker_no_vision(monkeypatch, tmp_path):
    """mode=path with only a URL attaches the URL marker (no local file, no fetch)."""
    monkeypatch.setattr("agent.image_routing.decide_image_input_mode", lambda *a, **k: "path")
    monkeypatch.setattr("hermes_cli.config.load_config", dict)

    with patch("tools.vision_tools.vision_analyze_tool") as vision_tool:
        result = _route_single_query_images(
            _cli(), "what is this", "what is this", [], ["https://example.com/photo.png"],
        )

    assert result.startswith(
        "[The user sent an image from a URL, not described automatically: https://example.com/photo.png]"
    )
    assert "what is this" in result
    vision_tool.assert_not_called()
