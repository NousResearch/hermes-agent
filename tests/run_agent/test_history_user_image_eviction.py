"""Request-path coverage for historical user-image eviction."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.chat_completion_helpers import handle_max_iterations
from agent.context_compressor import _is_image_part
from run_agent import AIAgent


def _image(label: str) -> dict:
    payload = base64.b64encode(label.encode()).decode()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{payload}"},
    }


def _images(message: dict) -> list[dict]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [part for part in content if _is_image_part(part)]


def _response(text: str = "ok") -> SimpleNamespace:
    message = SimpleNamespace(
        content=text,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="openai/gpt-4o",
        usage=None,
    )


@pytest.fixture
def agent() -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        instance = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="openai/gpt-4o",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance.client = MagicMock()
    instance.client.chat.completions.create.return_value = _response()
    instance._cached_system_prompt = "You are helpful."
    instance._use_prompt_caching = False
    instance.compression_enabled = False
    instance.save_trajectories = False
    instance.max_history_user_images = 1
    instance._model_supports_vision = lambda: True
    return instance


def test_main_loop_caps_historical_images_without_mutating_history(agent):
    image_turn = {
        "role": "user",
        "content": [
            {"type": "text", "text": "compare"},
            _image("first"),
            _image("second"),
        ],
    }
    history = [image_turn, {"role": "assistant", "content": "done"}]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("continue", conversation_history=history)

    assert result["completed"] is True
    sent = agent.client.chat.completions.create.call_args.kwargs["messages"]
    sent_image_turn = next(
        message
        for message in sent
        if message.get("role") == "user" and isinstance(message.get("content"), list)
    )
    assert len(_images(sent_image_turn)) == 1
    assert _images(sent_image_turn)[0] == _image("second")
    assert len(_images(image_turn)) == 2


def test_max_iterations_summary_applies_same_cap(agent):
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return "RAW"

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    transport = SimpleNamespace(
        normalize_response=lambda _response: SimpleNamespace(content="SUMMARY")
    )
    image_turn = {
        "role": "user",
        "content": [_image("first"), _image("second")],
    }
    messages = [image_turn, {"role": "assistant", "content": "done"}]

    with (
        patch.object(agent, "_ensure_primary_openai_client", return_value=client),
        patch.object(agent, "_get_transport", return_value=transport),
    ):
        output = handle_max_iterations(agent, messages, 5)

    assert output == "SUMMARY"
    sent_image_turn = next(
        message
        for message in captured["messages"]
        if message.get("role") == "user" and isinstance(message.get("content"), list)
    )
    assert len(_images(sent_image_turn)) == 1
    assert len(_images(image_turn)) == 2
