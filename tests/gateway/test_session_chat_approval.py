"""Session chat streams surface approval requests keyed by run id (#58856)."""
import asyncio

from gateway.platforms.api_server import APIServerAdapter, _SessionEventQueue


def test_session_stream_approval_keyed_by_run_id_and_enqueued():
    async def _go():
        host = APIServerAdapter.__new__(APIServerAdapter)
        host._run_approval_sessions = {}
        events = _SessionEventQueue("sess-1", "run_1")
        notify = host._register_session_stream_approval("run_1", events, "msg_1")
        assert host._run_approval_sessions == {"run_1": "run_1"}
        await asyncio.to_thread(notify, {"command": "rm -rf /", "description": "dangerous"})
        name, event = await asyncio.wait_for(events.queue.get(), 2)
        assert name == "approval.request"
        assert event["run_id"] == "run_1" and event["message_id"] == "msg_1"
        assert "once" in event["choices"] and "deny" in event["choices"]
    asyncio.run(_go())
