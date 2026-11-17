"""Gateway vision pre-process merges the analysis with the user text."""

import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_enrich_message_with_vision_merges_analysis_without_output_cap():
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
    # No output cap is forwarded: per the max-tokens-knob policy the aux
    # client decides token handling; conciseness comes from the prompt.
    assert "max_tokens" not in mock_vision.await_args.kwargs


async def _prompt_for_language(monkeypatch, lang):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("agent.i18n.get_language", lambda: lang)
    runner = GatewayRunner.__new__(GatewayRunner)
    with patch(
        "tools.vision_tools.vision_analyze_tool",
        new_callable=AsyncMock,
        return_value=json.dumps({"success": True, "analysis": "A cat."}),
    ) as mock_vision:
        await runner._enrich_message_with_vision(user_text="", image_paths=["/tmp/cat.png"])
    return mock_vision.await_args.kwargs["user_prompt"]


@pytest.mark.asyncio
async def test_vision_prompt_defaults_to_english_without_chinese_hint(monkeypatch):
    prompt = await _prompt_for_language(monkeypatch, "en")
    assert "written in English" in prompt
    assert "Chinese" not in prompt


@pytest.mark.asyncio
async def test_vision_prompt_follows_configured_language(monkeypatch):
    prompt = await _prompt_for_language(monkeypatch, "ru")
    assert "written in Русский" in prompt
    assert "Chinese" not in prompt


@pytest.mark.asyncio
async def test_vision_prompt_pack_language_falls_back_to_id(monkeypatch):
    prompt = await _prompt_for_language(monkeypatch, "xx-pack")
    assert "written in xx-pack" in prompt
