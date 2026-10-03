"""Session chat streams surface approval requests keyed by run id (#58856)."""
import asyncio
from unittest.mock import MagicMock, patch

from gateway.platforms.api_server import APIServerAdapter, _SessionEventQueue


def test_openai_stream_approval_includes_session_id():
    host = APIServerAdapter.__new__(APIServerAdapter)
    host._run_owners = {}
    host._run_approval_sessions = {}
    host._run_idempotency_scope = lambda request: "tenant"
    host._set_run_status = MagicMock()
    stream_q = MagicMock()
    request = MagicMock()

    notify, _ = host._register_stream_approval(request, "completion-1", stream_q, "session-1")
    notify({"request_id": "approval-1", "command": "rm -rf /"})

    event = stream_q.put_threadsafe.call_args.args[0][1]
    assert event["run_id"] == "completion-1"
    assert event["session_id"] == "session-1"



def test_session_stream_approval_keyed_by_run_id_and_enqueued():
    async def _go():
        host = APIServerAdapter.__new__(APIServerAdapter)
        host._run_approval_sessions = {}
        events = _SessionEventQueue("sess-1", "run_1")
        notify = host._register_session_stream_approval("run_1", events, "msg_1")
        assert host._run_approval_sessions == {"run_1": "run_1"}
        with patch.object(host, "_set_run_status") as set_status:
            await asyncio.to_thread(notify, {"command": "rm -rf /", "description": "dangerous"})
        assert set_status.call_args.args == ("run_1", "waiting_for_approval")
        name, event = await asyncio.wait_for(events.queue.get(), 2)
        assert name == "approval.request"
        assert event["run_id"] == "run_1" and event["message_id"] == "msg_1"
        assert "once" in event["choices"] and "deny" in event["choices"]
    asyncio.run(_go())
