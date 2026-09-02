"""Gateway vision pre-process prompt should stay concise."""

import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_enrich_message_with_vision_uses_concise_prompt():
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
    prompt = mock_vision.await_args.kwargs["user_prompt"]
    assert "What is happening here?" in prompt
    assert "gateway_auto_enrichment" in prompt
    assert "2-4 sentences" in prompt
    assert "untrusted visual data" in prompt
    # No output cap is forwarded: per the max-tokens-knob policy the aux
    # client decides token handling; conciseness comes from the prompt.
    assert "max_tokens" not in mock_vision.await_args.kwargs


@pytest.mark.asyncio
async def test_captionless_image_uses_default_vision_intent():
    from agent.vision_prompt import normalize_vision_intent
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    placeholder = "(The user sent a message with no text content)"

    with patch(
        "tools.vision_tools.vision_analyze_tool",
        new_callable=AsyncMock,
        return_value=json.dumps({"success": True, "analysis": "A receipt."}),
    ) as mock_vision:
        await runner._enrich_message_with_vision(placeholder, ["/tmp/receipt.png"])

    prompt = mock_vision.await_args.kwargs["user_prompt"]
    assert normalize_vision_intent("") in prompt
    assert placeholder not in prompt
