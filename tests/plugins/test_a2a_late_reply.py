"""Final reply delivery must never claim success without a recipient."""
import asyncio
import logging

from gateway.config import PlatformConfig
from plugins.platforms.a2a.adapter import A2AAdapter


def test_final_without_waiter_fails_loudly(caplog):
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    with caplog.at_level(logging.WARNING):
        result = asyncio.run(adapter.send("ctx-late", "late answer", reply_to="task-expired", metadata={"notify": True}))
    assert result.success is False
    assert "no pending waiter" in result.error
    assert "ctx-late" in caplog.text
    assert "task-expired" in caplog.text


def test_reply_after_observation_deadline_reaches_original_task(monkeypatch):
    import time
    from plugins.platforms.a2a import adapter as module, protocol
    monkeypatch.setattr(module, "_SSE_KEEPALIVE", 0.001)
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    rec = adapter.tasks.create("task-late", "ctx-late", "caller")
    future = adapter._add_pending("task-late", "ctx-late")
    pending = {"task_id": "task-late", "context_id": "ctx-late", "peer": "caller",
               "future": future, "started": time.time() - 1000, "created_iso": rec["created_iso"]}

    def late_reply():
        result = asyncio.run(adapter.send("ctx-late", "late answer", reply_to="task-late", metadata={"notify": True}))
        assert result.success

    state, text = adapter._finalize_task(pending, *adapter._await_reply(pending, keepalive=late_reply))
    assert state == protocol.STATE_COMPLETED
    assert text == "late answer"
    assert adapter.tasks.get("task-late")["reply"] == "late answer"
