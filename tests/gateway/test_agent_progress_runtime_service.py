"""DEAD path: not imported by gateway/run.py — contract-only unit tests.

Unit tests for gateway progress runtime helpers.
"""

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.dead_runtime_service

from gateway.agent_progress_runtime_service import (
    build_gateway_progress_runtime,
)
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource

class _Adapter(BasePlatformAdapter):
    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self.typing = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="progress-1")

    async def edit_message(self, chat_id, message_id, content):
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

@pytest.mark.asyncio
async def test_build_gateway_progress_runtime_uses_slack_event_message_id_for_dm_threading():
    adapter = _Adapter(Platform.SLACK)
    runtime = build_gateway_progress_runtime(
        user_config={"display": {"tool_progress": "all"}},
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="D123",
            chat_type="dm",
            thread_id=None,
        ),
        event_message_id="171234.001",
        adapter=adapter,
        hooks_ref=SimpleNamespace(loaded_hooks=False),
        loop_for_step=asyncio.get_running_loop(),
        session_id="sess-1",
        logger=MagicMock(),
        should_forward_status=lambda source, event_type, message: True,
    )

    assert runtime.tool_progress_enabled is True
    assert runtime.thread_metadata == {"thread_id": "171234.001"}

@pytest.mark.asyncio
async def test_build_gateway_progress_runtime_ignores_telegram_dm_event_message_id():
    adapter = _Adapter(Platform.TELEGRAM)
    runtime = build_gateway_progress_runtime(
        user_config={"display": {"tool_progress": "all"}},
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
            thread_id=None,
        ),
        event_message_id="777",
        adapter=adapter,
        hooks_ref=SimpleNamespace(loaded_hooks=False),
        loop_for_step=asyncio.get_running_loop(),
        session_id="sess-2",
        logger=MagicMock(),
        should_forward_status=lambda source, event_type, message: True,
    )

    assert runtime.thread_metadata is None

@pytest.mark.asyncio
async def test_progress_runtime_step_callback_normalizes_tool_names():
    captured = []

    class _Hooks:
        loaded_hooks = True

        async def emit(self, event_type, data):
            captured.append((event_type, data))

    runtime = build_gateway_progress_runtime(
        user_config={"display": {"tool_progress": "all"}},
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
            user_id="u1",
        ),
        event_message_id=None,
        adapter=_Adapter(Platform.TELEGRAM),
        hooks_ref=_Hooks(),
        loop_for_step=asyncio.get_running_loop(),
        session_id="sess-3",
        logger=MagicMock(),
        should_forward_status=lambda source, event_type, message: True,
    )

    runtime.step_callback(
        3,
        [
            {"name": "terminal", "result": "ok"},
            "read_file",
        ],
    )
    await asyncio.sleep(0.05)

    assert captured == [
        (
            "agent:step",
            {
                "platform": "telegram",
                "user_id": "u1",
                "session_id": "sess-3",
                "iteration": 3,
                "tool_names": ["terminal", "read_file"],
                "tools": [
                    {"name": "terminal", "result": "ok"},
                    "read_file",
                ],
            },
        )
    ]

@pytest.mark.asyncio
async def test_progress_runtime_status_callback_honors_filter():
    adapter = _Adapter(Platform.QQ_NAPCAT)
    runtime = build_gateway_progress_runtime(
        user_config={"display": {"tool_progress": "all"}},
        source=SessionSource(
            platform=Platform.QQ_NAPCAT,
            chat_id="987654",
            chat_type="group",
        ),
        event_message_id=None,
        adapter=adapter,
        hooks_ref=SimpleNamespace(loaded_hooks=False),
        loop_for_step=asyncio.get_running_loop(),
        session_id="sess-4",
        logger=MagicMock(),
        should_forward_status=lambda source, event_type, message: event_type != "lifecycle",
    )

    runtime.status_callback("lifecycle", "skip me")
    runtime.status_callback("context_pressure", "show me")
    await asyncio.sleep(0.05)

    assert adapter.sent == [
        {
            "chat_id": "987654",
            "content": "show me",
            "reply_to": None,
            "metadata": None,
        }
    ]

@pytest.mark.asyncio
async def test_progress_runtime_sender_collapses_duplicate_messages():
    adapter = _Adapter(Platform.TELEGRAM)
    runtime = build_gateway_progress_runtime(
        user_config={"display": {"tool_progress": "all"}},
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="dm",
        ),
        event_message_id=None,
        adapter=adapter,
        hooks_ref=SimpleNamespace(loaded_hooks=False),
        loop_for_step=asyncio.get_running_loop(),
        session_id="sess-5",
        logger=MagicMock(),
        should_forward_status=lambda source, event_type, message: True,
    )

    sender_task = asyncio.create_task(runtime.send_progress_messages())
    runtime.progress_queue.put('⚙️ terminal: "pwd"')
    runtime.progress_queue.put(("__dedup__", '⚙️ terminal: "pwd"', 1))
    await asyncio.sleep(0.35)
    sender_task.cancel()
    await sender_task

    assert adapter.sent
    assert adapter.sent[0]["content"] == '⚙️ terminal: "pwd"'
    assert adapter.edits[-1]["content"].endswith("(×2)")
