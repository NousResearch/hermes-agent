"""Temporary compaction agents (session hygiene, manual /compress) tell the memory provider the truth:
the conversation's own platform and identity, and that they are a ``flush`` helper, not a primary
CLI session. Before, they reached providers as ``platform="cli"``, ``agent_context="primary"`` with
no user/chat identity, so a pre-compress checkpoint landed outside the user's scope."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.memory_provider import MemoryProvider
from gateway.config import Platform
from gateway.session import SessionSource
from hermes_constants import get_hermes_home


class _RecordingProvider(MemoryProvider):
    pre_compress_checkpoint_api_version = 2

    def __init__(self):
        self.init_kwargs: dict = {}

    @property
    def name(self) -> str:
        return "recording"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self.init_kwargs = kwargs

    def get_tool_schemas(self):
        return []


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id="chat-42", chat_type="group",
                         user_id="user-7", user_name="Ada", thread_id="9")


def _checkpointing_home() -> None:
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(json.dumps(  # JSON is valid YAML
        {"memory": {"provider": "recording"}, "compression": {"checkpoint_required": True}}), encoding="utf-8")


def test_hygiene_agent_reaches_the_provider_as_a_flush_helper_of_the_real_conversation():
    from gateway.run import GatewayRunner

    _checkpointing_home()
    provider = _RecordingProvider()

    async def _no_row(_sid):
        return None

    gateway = SimpleNamespace(_session_db=SimpleNamespace(get_session=_no_row))
    runtime = {"api_key": "k", "base_url": "http://127.0.0.1:9/v1", "provider": "custom"}
    with (
        patch("agent.agent_init.resolve_memory_provider", lambda *_a, **_k: provider),
        patch("model_tools.get_tool_definitions", lambda *a, **k: []),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent, _db = asyncio.run(GatewayRunner._hmwa_hygiene_build_agent(
            gateway, "m", runtime, SimpleNamespace(session_id="sess-1"), _source(), "agent:main:telegram:group:chat-42"))
    agent.close()

    kw = provider.init_kwargs
    assert kw["platform"] == "telegram"
    assert kw["agent_context"] == "flush"
    assert (kw["user_id"], kw["chat_id"], kw["thread_id"]) == ("user-7", "chat-42", "9")
    assert kw["gateway_session_key"] == "agent:main:telegram:group:chat-42"
