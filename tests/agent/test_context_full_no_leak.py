"""No-leak regression coverage for raw full-context assembly."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

from tui_gateway import server


SECRET = "SECRET_TOKEN=abc123"


def _agent():
    return SimpleNamespace(
        model="test-model",
        tools=[],
        _cached_system_prompt="cached system",
        ephemeral_system_prompt=None,
        _memory_store=None,
        _memory_enabled=True,
        _user_profile_enabled=True,
        context_compressor=SimpleNamespace(context_length=4096, last_prompt_tokens=0),
    )


def test_compute_full_context_only_returns_raw_secret_in_return_value(monkeypatch):
    """Raw secret must live only in the returned payload, never in emit frames.

    Drive the real RPC handler and capture every frame the gateway would emit
    via ``server._emit`` while it runs. The secret must appear in the direct
    response payload and in NONE of the emitted event frames.
    """
    emitted_frames: list[object] = []
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: emitted_frames.append((args, kwargs)))

    sid = "context-full-noleak"
    history = [{"role": "user", "content": f"hello {SECRET} <script>alert(1)</script>"}]
    server._sessions[sid] = {
        "agent": _agent(),
        "history": history,
        "history_lock": threading.RLock(),
    }
    try:
        resp = server._methods["session.context_full"]("rid-noleak", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    # Secret present in the direct response payload...
    assert SECRET in json.dumps(resp)
    assert "<script>alert(1)</script>" in json.dumps(resp)
    # ...and absent from every emitted event frame (the no-leak invariant).
    assert all(SECRET not in json.dumps(frame, default=str) for frame in emitted_frames)


def test_context_full_rpc_exception_uses_static_error_without_raw_content(monkeypatch):
    sid = "context-full-error"
    secret_history = [{"role": "user", "content": f"raw payload {SECRET}"}]

    def boom(agent, history):
        raise RuntimeError(f"exploded while handling {SECRET}: {history!r}")

    monkeypatch.setattr("agent.context_breakdown.compute_session_context_full", boom)
    server._sessions[sid] = {
        "agent": _agent(),
        "history": secret_history,
        "history_lock": threading.RLock(),
    }
    try:
        resp = server._methods["session.context_full"]("rid-secret", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert resp["error"] == {"code": 5000, "message": "Could not compute full context"}
    assert SECRET not in json.dumps(resp)
