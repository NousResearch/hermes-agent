"""Legacy /v1/chat/completions streams surface approval requests (#51871)."""
import asyncio

from gateway.platforms.api_server import ThreadSafeAsyncQueue
from gateway.platforms.api_server_openai_routes import OpenAICompatRoutesMixin


class _Host(OpenAICompatRoutesMixin):
    def __init__(self):
        self._run_approval_sessions = {}


def test_stream_approval_keyed_by_completion_id_and_enqueued():
    async def _go():
        host, q = _Host(), ThreadSafeAsyncQueue()
        notify = host._register_stream_approval("chatcmpl-1", q, "sess-1")
        assert host._run_approval_sessions == {"chatcmpl-1": "chatcmpl-1"}
        notify({"command": "rm -rf /", "description": "dangerous"})
        kind, event = await asyncio.wait_for(q.get(), 2)
        assert kind == "__approval__"
        assert event["event"] == "approval.request" and event["run_id"] == "chatcmpl-1"
        assert "once" in event["choices"] and "deny" in event["choices"]
    asyncio.run(_go())
