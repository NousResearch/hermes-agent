"""Round-six regressions for final egress and headless approval lifecycle."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import tui_gateway.server as server
import tools.approval as approval
from gateway.platforms.api_server_runs import _safe_api_status


class _RemoteTransport:
    auth_identity = {"user_id": "operator", "provider": "dashboard"}


def _session():
    return {
        "agent": SimpleNamespace(session_id="round6-session"),
        "owner_principal": "dashboard:operator",
        "profile_name": "default",
        "profile_home": None,
        "session_key": "round6-session",
        "cwd": ".",
    }


def test_api_status_allowlist_drops_unknown_nested_fields():
    raw = "https://fixture-user:fixture-pass@example.test/?token=fixture-redacted-token"
    result = _safe_api_status(
        {
            "run_id": "run-1",
            "status": "completed",
            "unknown": {"raw": raw},
            "approval": {"request_id": "rid", "choices": ["garbage", "deny"], "raw": raw},
        }
    )
    assert "unknown" not in result
    assert "raw" not in result["approval"]
    assert result["approval"]["choices"] == ["deny"]
    assert "fixture-redacted-token" not in str(result)


def test_api_status_drops_malformed_allowed_types():
    result = _safe_api_status(
        {
            "run_id": {"secret": "fixture"},
            "status": ["fixture-secret"],
            "output": {"nested": "fixture-secret"},
            "approval": {"choices": [["fixture-secret"], "deny"]},
        }
    )
    assert "run_id" not in result
    assert "status" not in result
    assert result["output"] == {}
    assert result["approval"]["choices"] == ["deny"]


def test_invalid_approval_payload_is_non_actionable_and_redacted(monkeypatch):
    monkeypatch.setattr(
        "gateway.run._redact_approval_command",
        lambda _value: (_ for _ in ()).throw(RuntimeError("fixture")),
    )
    payload = server._approval_request_payload(
        {
            "request_id": "",
            "command": "https://fixture-user:fixture-pass@example.test/?token=fixture-redacted-token",
            "choices": ["always"],
        }
    )
    assert "request_id" not in payload
    assert payload["command"] == "[REDACTED]"
    assert payload["choices"] == ["deny"]


def test_clarification_choices_are_redacted():
    raw = "https://fixture-user:fixture-pass@example.test/?token=fixture-redacted-token"
    choices = server._redact_prompt_choices([raw, "safe"])
    assert choices is not None
    assert "fixture-redacted-token" not in str(choices)
    assert choices[1] == "safe"


def test_gateway_clarify_choices_helper_redacts_before_send():
    from gateway.run import _redact_clarify_choices

    raw = "https://fixture-user:fixture-pass@example.test/?token=fixture-redacted-token"
    result = _redact_clarify_choices([raw, "safe"])
    assert "fixture-redacted-token" not in str(result)
    assert result[1] == "safe"


def test_headless_pending_approval_uses_exact_id_and_expiry():
    session_key = "round6-headless"
    raw = "https://fixture-user:fixture-pass@example.test/?token=fixture-redacted-token"
    try:
        pending = approval.submit_pending(session_key, {"command": raw, "description": raw})
        assert isinstance(pending["request_id"], str) and pending["request_id"]
        assert isinstance(pending["expires_at"], float)
        visible = approval.get_pending_gateway_approval(session_key)
        assert visible["request_id"] == pending["request_id"]
        assert "fixture-redacted-token" not in str(visible)
        assert approval.resolve_gateway_approval(
            session_key, "deny", request_id=pending["request_id"]
        ) == 1
    finally:
        approval.clear_session(session_key)


def test_batch_clarify_late_response_is_rejected_after_waiter_cleanup():
    rid = "round6-batch-late"
    sid = "round6-batch-session"
    event = __import__("threading").Event()
    with server._prompt_lock:
        server._pending[rid] = (sid, event)
        server._pending_prompt_payloads[rid] = ("clarify.request", {"request_id": rid})
        server._batch_clarify[rid] = {"qids": ["q1", "q2"], "answers": {}}
    try:
        method_token = server._current_rpc_method.set("clarify.respond")
        try:
            assert server._respond("rpc-q1", {"request_id": rid, "question_id": "q1", "answer": "a1"}, "answer")["result"]["remaining"] == ["q2"]
            final = server._respond("rpc-q2", {"request_id": rid, "question_id": "q2", "answer": "a2"}, "answer")
            assert final["result"]["remaining"] == []
            with server._prompt_lock:
                server._pending.pop(rid, None)
                server._batch_clarify.pop(rid, None)
            late = server._respond("rpc-late", {"request_id": rid, "question_id": "q1", "answer": "late"}, "answer")
            assert late["error"]["code"] == 4091
        finally:
            server._current_rpc_method.reset(method_token)
    finally:
        with server._prompt_lock:
            server._pending.pop(rid, None)
            server._pending_prompt_payloads.pop(rid, None)
            server._batch_clarify.pop(rid, None)
            server._completed_prompt_ids.pop(rid, None)


def test_remote_quick_command_sets_dotenv_sentinel(monkeypatch):
    sid = "round6-quick-dotenv"
    server._sessions[sid] = _session()
    monkeypatch.setattr(server, "current_transport", lambda: _RemoteTransport())
    monkeypatch.setattr(server, "_load_cfg", lambda: {"quick_commands": {"quick": {"type": "exec", "command": "echo ok"}}})
    monkeypatch.setattr(server, "_run_remote_command_guard", lambda *_args: {"approved": True})
    completed = SimpleNamespace(stdout="ok", stderr="", returncode=0)
    try:
        with patch("subprocess.run", return_value=completed) as run:
            response = server._methods["command.dispatch"](
                "rpc-quick-dotenv", {"session_id": sid, "name": "quick", "arg": ""}
            )
        assert response["result"]["output"] == "ok"
        assert run.call_args.kwargs["env"]["HERMES_DISABLE_DOTENV"] == "1"
    finally:
        server._sessions.pop(sid, None)
