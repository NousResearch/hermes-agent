"""Shared vision analysis backend for gateway/runtime callers."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


async def _analyze_image_impl(
    *,
    image_ref: str,
    user_prompt: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    from tools.vision_tools import _legacy_vision_analyze_tool_impl

    result = await _legacy_vision_analyze_tool_impl(
        image_url=image_ref,
        user_prompt=user_prompt,
        model=model,
    )
    if isinstance(result, str):
        return json.loads(result)
    return dict(result or {})


async def analyze_image(
    *,
    image_ref: str,
    user_prompt: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze one image and return a structured result payload."""
    return await _analyze_image_impl(
        image_ref=image_ref,
        user_prompt=user_prompt,
        model=model,
    )
