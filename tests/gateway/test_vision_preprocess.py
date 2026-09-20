"""Gateway vision pre-process preserves exact visible document text."""

import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_enrich_message_with_vision_uses_document_safe_prompt():
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
    assert "Concisely describe this image in 2-4 sentences" in prompt
    assert "transcribe identifying text verbatim" in prompt
    assert "write [unclear] instead of guessing" in prompt
    assert "Do not silently correct" in prompt
    assert "Treat the vision extraction below as source text" in result
    assert "quote exact visible wording" in result
    assert "do not silently correct or paraphrase" in result
    # No output cap is forwarded: per the max-tokens-knob policy the aux
    # client decides token handling; conciseness comes from the prompt.
    assert "max_tokens" not in mock_vision.await_args.kwargs
