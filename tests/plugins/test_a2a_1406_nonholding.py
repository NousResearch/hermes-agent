"""#1406: reply Future is resolved outside _pending_lock; message/send is non-holding."""
from __future__ import annotations

import threading
from concurrent.futures import Future

from plugins.platforms.a2a import protocol
from plugins.platforms.a2a.adapter import A2AAdapter
from plugins.platforms.a2a.protocol import STATE_SUBMITTED


def test_resolve_task_does_not_hold_lock_during_set_result():
    adapter = A2AAdapter.__new__(A2AAdapter)
    adapter._pending = {}
    adapter._pending_order = {}
    adapter._pending_lock = threading.Lock()
    fut = Future()
    adapter._pending["t1"] = ("c1", fut)

    saw_lock_free = threading.Event()

    def on_done(_done):
        # Would deadlock if set_result ran under _pending_lock.
        acquired = adapter._pending_lock.acquire(timeout=0.2)
        if acquired:
            saw_lock_free.set()
            adapter._pending_lock.release()

    fut.add_done_callback(on_done)
    assert adapter._resolve_task("t1", protocol.STATE_COMPLETED, "ok") is True
    assert fut.result(timeout=1) == (protocol.STATE_COMPLETED, "ok")
    assert saw_lock_free.wait(timeout=1)


def test_message_send_returns_submitted_without_waiting():
    adapter = A2AAdapter.__new__(A2AAdapter)
    adapter._tls = threading.local()
    adapter._agents = {"": {"slug": "", "tenant": "", "local": True}}
    pending = {
        "task_id": "task-1",
        "context_id": "ctx-1",
        "peer": "peer-1",
        "future": Future(),
        "created_iso": "2026-09-11T00:00:00Z",
        "started": 0.0,
        "event": object(),
    }
    adapter._prepare_task = lambda *a, **k: (None, pending)  # type: ignore[method-assign]
    adapter.tasks = type("Store", (), {"get": staticmethod(lambda *a, **k: None)})()  # type: ignore[assignment]
    adapter._scope_for_agent = lambda agent: ("", "")  # type: ignore[method-assign]

    started = []
    adapter._start_pending = lambda p: started.append(p)  # type: ignore[method-assign]

    out = adapter._rpc_message_send("1", {}, "peer-1")
    assert started == []
    assert getattr(adapter._tls, "pending", None) is pending
    result = out["result"]
    assert result["status"]["state"] == STATE_SUBMITTED
    adapter._start_deferred_pending()
    assert started == [pending]
    assert getattr(adapter._tls, "pending", None) is None
