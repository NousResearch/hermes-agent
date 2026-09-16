"""Round-five regressions from the substantive security review."""

from __future__ import annotations

import math
import threading
from types import SimpleNamespace
from unittest.mock import patch

import tui_gateway.server as server
import tools.approval as approval


class _RemoteTransport:
    def __init__(self, user_id="operator"):
        self.auth_identity = {"user_id": user_id, "provider": "dashboard"}


def _session(principal="dashboard:operator"):
    return {
        "agent": SimpleNamespace(session_id="stored-round5"),
        "owner_principal": principal,
        "profile_name": "default",
        "profile_home": None,
        "session_key": "round5-session",
        "cwd": ".",
    }


def test_remote_session_lookup_rejects_cross_principal(monkeypatch):
    sid = "round5-cross-principal"
    server._sessions[sid] = _session("dashboard:owner")
    monkeypatch.setattr(server, "current_transport", lambda: _RemoteTransport("attacker"))
    try:
        session, response = server._sess_nowait({"session_id": sid}, "rpc-owner")
        assert session is None
        assert response["error"]["code"] == 4032
    finally:
        server._sessions.pop(sid, None)


def test_remote_shell_exec_sets_dotenv_sentinel(monkeypatch):
    sid = "round5-shell-dotenv"
    server._sessions[sid] = _session()
    monkeypatch.setattr(server, "current_transport", lambda: _RemoteTransport())
    monkeypatch.setattr(server, "_run_remote_command_guard", lambda *_args: {"approved": True})
    completed = SimpleNamespace(stdout="ok", stderr="", returncode=0)
    try:
        with patch("subprocess.run", return_value=completed) as run:
            response = server._methods["shell.exec"](
                "rpc-shell-dotenv", {"session_id": sid, "command": "echo ok"}
            )
        assert response["result"]["code"] == 0
        assert run.call_args.kwargs["env"]["HERMES_DISABLE_DOTENV"] == "1"
    finally:
        server._sessions.pop(sid, None)


def test_remote_plugin_output_is_redacted(monkeypatch):
    sid = "round5-plugin-output"
    server._sessions[sid] = _session()
    monkeypatch.setattr(server, "current_transport", lambda: _RemoteTransport())
    monkeypatch.setattr(server, "_run_remote_command_guard", lambda *_args: {"approved": True})
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_command_handler", lambda _name: lambda _arg: "https://fixture-user:fixture-pass@example.test/?token=fixture-redacted-token")
    monkeypatch.setattr("hermes_cli.plugins.resolve_plugin_command_result", lambda value: value)
    try:
        response = server._methods["command.dispatch"](
            "rpc-plugin-output", {"session_id": sid, "name": "fixture-plugin", "arg": ""}
        )
        assert "fixture-secret-output" not in str(response)
        assert "fixture-pass" not in str(response)
        assert "fixture-redacted-token" not in str(response)
    finally:
        server._sessions.pop(sid, None)


def test_approval_entry_rejects_nonfinite_timeout():
    with __import__("pytest").raises(ValueError):
        approval._ApprovalEntry(
            {"command": "fixture"},
            timeout_seconds=math.inf,
        )


def test_batch_clarify_rejects_answer_after_final_question(monkeypatch):
    rid = "round5-batch-final"
    sid = "round5-batch-session"
    event = threading.Event()
    monkeypatch.setattr(server, "current_transport", lambda: None)
    with server._prompt_lock:
        server._pending[rid] = (sid, event)
        server._pending_prompt_payloads[rid] = ("clarify.request", {"request_id": rid})
        server._batch_clarify[rid] = {"qids": ["q0"], "answers": {}}
    try:
        method_token = server._current_rpc_method.set("clarify.respond")
        try:
            first = server._methods["clarify.respond"](
                "rpc-batch-first", {"request_id": rid, "question_id": "q0", "answer": "first"}
            )
            second = server._methods["clarify.respond"](
                "rpc-batch-second", {"request_id": rid, "question_id": "q0", "answer": "late"}
            )
            assert first["result"]["remaining"] == []
            assert second["error"]["code"] == 4091
        finally:
            server._current_rpc_method.reset(method_token)
    finally:
        with server._prompt_lock:
            server._pending.pop(rid, None)
            server._pending_prompt_payloads.pop(rid, None)
            server._batch_clarify.pop(rid, None)
            server._answers.pop(rid, None)
