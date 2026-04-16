import json
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_vision_backend_delegates_to_shared_analyzer(monkeypatch):
    from agent import vision_backend

    analyze = AsyncMock(
        return_value={
            "success": True,
            "analysis": "a screen with a pink pig icon",
            "provider": "custom",
        }
    )
    monkeypatch.setattr(vision_backend, "_analyze_image_impl", analyze)

    result = await vision_backend.analyze_image(
        image_ref="https://example.com/pig.png",
        user_prompt="describe it",
    )

    assert result["success"] is True
    assert "pink pig" in result["analysis"]
    analyze.assert_awaited_once_with(
        image_ref="https://example.com/pig.png",
        user_prompt="describe it",
        model=None,
    )


@pytest.mark.asyncio
async def test_vision_tool_wraps_backend_result_as_json(monkeypatch):
    from tools import vision_tools

    analyze = AsyncMock(
        return_value={
            "success": True,
            "analysis": "a simple screenshot",
            "provider": "custom",
        }
    )
    monkeypatch.setattr("agent.vision_backend.analyze_image", analyze)

    result = await vision_tools.vision_analyze_tool(
        "https://example.com/demo.png",
        "describe it",
    )

    payload = json.loads(result)
    assert payload["success"] is True
    assert payload["analysis"] == "a simple screenshot"
    analyze.assert_awaited_once()
