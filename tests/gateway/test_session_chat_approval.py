"""Session chat streams surface approval requests keyed by run id (#58856)."""
import asyncio

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _SessionEventQueue
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry


def test_session_stream_approval_keyed_by_run_id_and_enqueued():
    async def _go():
        host = APIServerAdapter(PlatformConfig(enabled=True))
        host._set_run_status("run_1", "running")
        events = _SessionEventQueue("sess-1", "run_1")
        notify = host._register_session_stream_approval("run_1", events, "msg_1")
        assert host._run_approval_sessions == {"run_1": "run_1"}
        entry = _ApprovalEntry({"command": "rm -rf /", "description": "dangerous"})
        with approval._lock:
            approval._gateway_queues["run_1"] = [entry]
        try:
            await asyncio.to_thread(notify, entry.data)
            name, event = await asyncio.wait_for(events.queue.get(), 2)
            assert host._run_statuses["run_1"]["status"] == "waiting_for_approval"
            assert name == "approval.request"
            assert event["run_id"] == "run_1" and event["message_id"] == "msg_1"
            assert "once" in event["choices"] and "deny" in event["choices"]
        finally:
            approval.unregister_gateway_notify("run_1")
    asyncio.run(_go())
