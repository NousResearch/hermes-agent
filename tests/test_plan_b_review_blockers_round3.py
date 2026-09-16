"""Round-three regressions for clarify/MCP response ownership."""

from __future__ import annotations

from unittest.mock import patch

import tui_gateway.server as server


class _RemoteTransport:
    def __init__(self, identity):
        self.auth_identity = identity


def test_remote_clarify_response_requires_owner_session_and_profile(monkeypatch):
    sid = "clarify-owner"
    rid = "clarify-request"
    owner = {
        "owner_principal": "principal-owner",
        "profile_name": "profile-a",
        "source": "dashboard",
    }
    monkeypatch.setattr(server, "current_transport", lambda: _RemoteTransport("principal-other"))
    server._sessions[sid] = owner
    with server._prompt_lock:
        server._pending[rid] = (sid, __import__("threading").Event())
        server._pending_prompt_payloads[rid] = (
            "clarify.request",
            {"request_id": rid, "choices": ["yes", "no"]},
        )
    try:
        method_token = server._current_rpc_method.set("clarify.respond")
        try:
            response = server._methods["clarify.respond"](
                "rpc-clarify",
                {"request_id": rid, "session_id": sid, "profile": "profile-a", "answer": "yes"},
            )
            assert response["error"]["code"] == 4032
        finally:
            server._current_rpc_method.reset(method_token)
    finally:
        server._sessions.pop(sid, None)
        with server._prompt_lock:
            server._pending.pop(rid, None)
            server._pending_prompt_payloads.pop(rid, None)


def test_clarify_single_request_is_one_shot(monkeypatch):
    sid = "clarify-one-shot"
    rid = "clarify-once"
    event = __import__("threading").Event()
    monkeypatch.setattr(server, "current_transport", lambda: None)
    server._sessions[sid] = {"source": "dashboard", "profile_name": "default"}
    with server._prompt_lock:
        server._pending[rid] = (sid, event)
        server._pending_prompt_payloads[rid] = (
            "clarify.request",
            {"request_id": rid, "choices": ["yes", "no"]},
        )
    try:
        method_token = server._current_rpc_method.set("clarify.respond")
        try:
            first = server._methods["clarify.respond"](
                "rpc-first", {"request_id": rid, "session_id": sid, "answer": "yes"}
            )
            second = server._methods["clarify.respond"](
                "rpc-second", {"request_id": rid, "session_id": sid, "answer": "no"}
            )
            assert first["result"]["status"] == "ok"
            assert second["error"]["code"] == 4091
        finally:
            server._current_rpc_method.reset(method_token)
    finally:
        server._sessions.pop(sid, None)
        with server._prompt_lock:
            server._pending.pop(rid, None)
            server._pending_prompt_payloads.pop(rid, None)
            server._answers.pop(rid, None)


# Profile propagation is covered behaviorally by the Web Vitest suite.  This
# Python file must not inspect TypeScript source text.
