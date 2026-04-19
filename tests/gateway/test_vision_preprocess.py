"""Gateway vision pre-process defaults should stay concise."""

import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_enrich_message_with_vision_uses_concise_prompt_and_lower_token_cap():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)

    with patch(
        "tools.vision_tools.vision_analyze_tool",
        new_callable=AsyncMock,
        return_value=json.dumps({"success": True, "analysis": "A cat on a chair."}),
    ) as mock_vision:
        result = await runner._enrich_message_with_vision(
            user_text="What is happening here?",
            image_paths=["/tmp/cat.png"],
        )

    assert "A cat on a chair." in result
    assert "What is happening here?" in result
    assert mock_vision.await_args.kwargs["max_tokens"] == 500
    assert "Concisely describe this image in 2-4 sentences" in mock_vision.await_args.kwargs["user_prompt"]
    assert "Skip decorative details." in mock_vision.await_args.kwargs["user_prompt"]
