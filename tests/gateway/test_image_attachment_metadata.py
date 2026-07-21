"""Regression tests for gateway image attachment metadata injection."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.run import GatewayRunner


def _png_bytes() -> bytes:
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg=="
    )


def _assert_attachment_metadata(text: str, img: Path):
    assert "[Hermes image attachment]" in text
    assert f"local_image_path={img}" in text
    assert f"size_bytes={img.stat().st_size}" in text
    assert "mime_type=image/png" in text
    assert "[/Hermes image attachment]" in text


@pytest.mark.asyncio
async def test_text_route_success_preserves_local_image_metadata(tmp_path: Path):
    img = tmp_path / "chat.png"
    img.write_bytes(_png_bytes())
    runner = GatewayRunner.__new__(GatewayRunner)

    async def _vision_success(**_kwargs):
        return json.dumps({"success": True, "analysis": "a visible screenshot"})

    with patch("tools.vision_tools.vision_analyze_tool", side_effect=_vision_success):
        result = await runner._enrich_message_with_vision("create defect", [str(img)])

    _assert_attachment_metadata(result, img)
    assert "a visible screenshot" in result
    assert "create defect" in result


@pytest.mark.asyncio
async def test_text_route_vision_failure_still_preserves_local_image_metadata(tmp_path: Path):
    img = tmp_path / "chat.png"
    img.write_bytes(_png_bytes())
    runner = GatewayRunner.__new__(GatewayRunner)

    async def _vision_failure(**_kwargs):
        return json.dumps({"success": False, "analysis": "provider rejected image"})

    with patch("tools.vision_tools.vision_analyze_tool", side_effect=_vision_failure):
        result = await runner._enrich_message_with_vision("create defect", [str(img)])

    _assert_attachment_metadata(result, img)
    assert "couldn't quite see it" in result
    assert "create defect" in result


@pytest.mark.asyncio
async def test_text_route_vision_exception_still_preserves_local_image_metadata(tmp_path: Path):
    img = tmp_path / "chat.png"
    img.write_bytes(_png_bytes())
    runner = GatewayRunner.__new__(GatewayRunner)

    async def _vision_explodes(**_kwargs):
        raise RuntimeError("vision model cannot read images")

    with patch("tools.vision_tools.vision_analyze_tool", side_effect=_vision_explodes):
        result = await runner._enrich_message_with_vision("create defect", [str(img)])

    _assert_attachment_metadata(result, img)
    assert "something went wrong" in result
    assert "create defect" in result
