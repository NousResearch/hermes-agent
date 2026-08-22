"""Dispatcher concurrency-limit propagation tests."""

import threading
from contextlib import nullcontext
from types import SimpleNamespace

from hermes_cli import kanban_db as kb


def test_standalone_daemon_forwards_global_worker_limit(monkeypatch):
    captured = {}
    stop_event = threading.Event()

    def fake_dispatch(_conn, **kwargs):
        captured.update(kwargs)
        stop_event.set()
        return SimpleNamespace()

    monkeypatch.setattr(kb, "connect_closing", lambda: nullcontext(object()))
    monkeypatch.setattr(kb, "dispatch_once", fake_dispatch)

    kb.run_daemon(
        interval=0,
        max_spawn=None,
        max_in_progress=3,
        stop_event=stop_event,
    )

    assert captured["max_in_progress"] == 3
