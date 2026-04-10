import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_prepare_user_message_with_images_uses_native_multimodal_for_codex(tmp_path):
    runner = object.__new__(GatewayRunner)
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = await runner._prepare_user_message_with_images(
        "Describe this",
        [str(img)],
        {
            "runtime": {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex",
            }
        },
    )

    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    assert result[1]["type"] == "image_url"
    assert result[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_prepare_user_message_with_images_uses_vision_fallback_for_text_only_route(tmp_path):
    runner = object.__new__(GatewayRunner)
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    async def _fake_enrich(message_text, image_paths):
        assert message_text == "Describe this"
        assert image_paths == [str(img)]
        return "fallback text"

    runner._enrich_message_with_vision = _fake_enrich

    result = await runner._prepare_user_message_with_images(
        "Describe this",
        [str(img)],
        {
            "runtime": {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
            }
        },
    )

    assert result == "fallback text"


@pytest.mark.asyncio
async def test_run_agent_awaits_image_preparation(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    class FakeAgent:
        def __init__(self, **kwargs):
            self.tools = []

        def run_conversation(self, message, conversation_history=None, task_id=None):
            assert message == "native-ready"
            return {"final_response": "done", "messages": [], "api_calls": 1}

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner._prepare_user_message_with_images = AsyncMock(return_value="native-ready")
    runner._resolve_turn_agent_config = lambda message, model, runtime_kwargs: {
        "model": "gpt-5.4",
        "runtime": {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
        "api_key": "***",
    }

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        thread_id=None,
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key="agent:main:telegram:dm:123",
        image_paths=[str(tmp_path / "shot.png")],
    )

    assert result["final_response"] == "done"
    runner._prepare_user_message_with_images.assert_awaited_once()