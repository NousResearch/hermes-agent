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
    assert (
        "Concisely describe this image in 2-4 sentences"
        in mock_vision.await_args.kwargs["user_prompt"]
    )
    assert "Skip decorative details." in mock_vision.await_args.kwargs["user_prompt"]
    # No output cap is forwarded: per the max-tokens-knob policy the aux
    # client decides token handling; conciseness comes from the prompt.
    assert "max_tokens" not in mock_vision.await_args.kwargs


@pytest.mark.asyncio
async def test_vision_prompt_does_not_request_a_specific_language():
    """The image description must not ask for a named language.

    Naming a language here pins the *description* to it, and that description is
    injected into the conversation as context. A model reading a Chinese image
    description then continues in Chinese, so a user talking to Hermes in any
    other language gets replies in a language they never asked for. The prompt
    is a length constraint only; the language follows the user.
    """
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)

    with patch(
        "tools.vision_tools.vision_analyze_tool",
        new_callable=AsyncMock,
        return_value=json.dumps({"success": True, "analysis": "A cat on a chair."}),
    ) as mock_vision:
        await runner._enrich_message_with_vision(
            user_text="What is happening here?",
            image_paths=["/tmp/cat.png"],
        )

    prompt = mock_vision.await_args.kwargs["user_prompt"]
    for language in ("Chinese", "Chinese characters", "English words", "Japanese", "Korean"):
        assert language not in prompt, (
            f"vision prompt requests a specific language ({language!r}); "
            "the description must be language-neutral"
        )
    # The length guidance itself stays.
    assert "~150 words" in prompt
