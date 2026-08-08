"""Regression tests for Telegram topic-specific toolset scoping."""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._session_reasoning_overrides = {}
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


class _CapturingAgent:
    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
        }


def test_resolve_enabled_toolsets_for_telegram_topic_uses_topic_allowlist():
    user_config = {
        "platform_toolsets": {
            "telegram": ["web", "file", "memory"],
        },
        "mcp_servers": {
            "mcp-custom": {"enabled": True},
        },
        "telegram": {
            "group_topics": [
                {
                    "chat_id": "-1001",
                    "topics": [
                        {
                            "thread_id": 3,
                            "name": "Ops",
                            "enabled_toolsets": ["file", "web", "mcp-custom"],
                        }
                    ],
                }
            ]
        },
    }
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="3",
        user_id="user-1",
    )

    enabled = gateway_run._resolve_enabled_toolsets_for_source(user_config, source)

    assert enabled == ["file", "mcp-custom", "web"]


def test_run_agent_telegram_topic_scopes_enabled_toolsets(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "platform_toolsets:\n"
        "  telegram: [web, file, memory]\n"
        "telegram:\n"
        "  group_topics:\n"
        "    - chat_id: \"-1001\"\n"
        "      topics:\n"
        "        - thread_id: 3\n"
        "          name: Ops\n"
        "          enabled_toolsets: [file, web]\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
        },
    )
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    _CapturingAgent.last_init = None
    runner = _make_runner()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_name="Team Chat",
        chat_type="group",
        thread_id="3",
        user_id="user-1",
        user_name="Test User",
    )

    result = asyncio.run(
        runner._run_agent(
            message="ping",
            context_prompt="",
            history=[],
            source=source,
            session_id="session-1",
            session_key="agent:main:telegram:group:-1001:3",
        )
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init is not None
    assert _CapturingAgent.last_init["enabled_toolsets"] == ["file", "web"]
