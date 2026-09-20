import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_api_fanout_preserves_exact_raw_response():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    seen = []

    async def deliver(**kwargs):
        seen.append(kwargs)

    adapter.set_final_response_fanout_handler(deliver)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    await adapter._fanout_completed_api_turn(
        session_source=source, response_text="  exact response  ", surface="session_chat")
    await asyncio.gather(*adapter._fanout_tasks)
    assert seen == [{"session_source": source, "content": "  exact response  ", "surface": "session_chat"}]
    adapter._response_store.close()


@pytest.mark.asyncio
async def test_runner_delivers_to_exact_native_adapter():
    calls = []

    class Target:
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            calls.append((chat_id, content, reply_to, metadata))
            return SimpleNamespace(success=True)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: Target()}
    runner._profile_adapters = {}
    source = SessionSource(platform=Platform.DISCORD, chat_id="99", chat_type="group", thread_id="7")
    await runner._deliver_api_final_response(session_source=source, content="done", surface="session_chat")
    assert calls == [("99", "done", None, {"thread_id": "7"})]