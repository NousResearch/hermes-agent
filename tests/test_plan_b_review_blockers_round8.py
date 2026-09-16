"""Adversarial regressions from the independent security review.

These tests intentionally exercise the public ownership/approval seams rather
than asserting private implementation details.  Keep them small and focused so
an isolated failure identifies one contract.
"""

import asyncio
import hashlib
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.platforms.api_server import (
    APIServerAdapter,
    ResponseStore,
    _redact_api_error_text,
)
from gateway.platforms.api_server_runs import _safe_api_status
from gateway.config import PlatformConfig
from tools.approval import _ApprovalEntry, resolve_gateway_approval, submit_pending
from tools.environments.local import hermes_subprocess_env
from tui_gateway import server


_SECRET = "sk-review-secret-123456789"


def test_response_store_binds_reads_and_conversations_to_owner_scope(tmp_path):
    store = ResponseStore(db_path=str(tmp_path / "responses.db"))
    try:
        store.put("resp-a", {"response": {"id": "resp-a"}}, owner_scope="profile-a:owner-a")
        store.put("resp-b", {"response": {"id": "resp-b"}}, owner_scope="profile-b:owner-b")
        store.set_conversation("chat", "resp-a", owner_scope="profile-a:owner-a")
        store.set_conversation("chat", "resp-b", owner_scope="profile-b:owner-b")

        assert store.get("resp-a", owner_scope="profile-a:owner-a") is not None
        assert store.get("resp-a", owner_scope="profile-b:owner-b") is None
        assert store.get_conversation("chat", owner_scope="profile-a:owner-a") == "resp-a"
        assert store.get_conversation("chat", owner_scope="profile-b:owner-b") == "resp-b"
        assert store.delete("resp-a", owner_scope="profile-b:owner-b") is False
    finally:
        store.close()


def test_api_error_redaction_removes_url_credentials():
    text = _redact_api_error_text(
        "provider failed at https://alice:pw_plain_review_12345@private.example/v1?api_key="
        + _SECRET
    )
    assert _SECRET not in text
    assert "pw_plain_review_12345" not in text
    assert "https://alice:***@" in text


def test_run_status_redacts_nested_output_tail_values():
    status = _safe_api_status(
        {
            "run_id": "run-1",
            "status": "running",
            "output_tail": [
                {"preview": f"tool printed {_SECRET}", "safe": "ok"},
                {"nested": [{"text": _SECRET}]},
            ],
        }
    )
    assert "output_tail" in status
    assert _SECRET not in repr(status)


@pytest.mark.asyncio
async def test_remote_run_event_callback_redacts_list_output_tail():
    adapter = MagicMock()
    adapter._run_statuses = {}
    adapter._run_streams = {"run-1": asyncio.Queue()}
    adapter._set_run_status = lambda *args, **kwargs: None
    # Use the real helper with a minimal API-server facade.
    from gateway.platforms import api_server_runs

    callback = api_server_runs._make_run_event_callback(
        adapter, "run-1", asyncio.get_running_loop(), _api_server=SimpleNamespace(
            redact_sensitive_text=lambda value, **_: value.replace(_SECRET, "[REDACTED]")
        )
    )
    callback(
        "subagent.complete",
        output_tail=[{"preview": _SECRET}],
        subagent_id="child-1",
    )
    await asyncio.sleep(0)
    event = adapter._run_streams["run-1"].get_nowait()
    assert _SECRET not in repr(event)


def test_run_events_uses_room_grant_status_authorizer():
    from gateway.platforms import api_server_runs

    adapter = MagicMock()
    request = MagicMock()
    request.match_info = {"run_id": "run-1"}
    expected = object()
    adapter._check_auth.return_value = object()
    adapter._check_run_auth.return_value = expected

    result = asyncio.run(
        api_server_runs._handle_run_events(
            adapter,
            request,
            _api_server=SimpleNamespace(
                _openai_error=lambda *a, **k: {},
                _sse_frame=lambda event: event,
            ),
        )
    )
    assert result is expected
    adapter._check_run_auth.assert_called_once_with(request, permission="status")
    adapter._check_auth.assert_not_called()


def test_canonical_profile_accepts_current_non_default_launch_profile(monkeypatch):
    monkeypatch.setattr(server, "_profile_home", lambda name: None)
    monkeypatch.setattr(server, "_current_profile_name", lambda: "work")
    assert server._canonical_profile_request("work") == "work"


def test_remote_exec_passes_selected_profile_to_session_lookup(monkeypatch):
    seen = {}
    transport = SimpleNamespace(auth_identity="principal:work:user")
    session = {
        "owner_principal": "principal:work:user",
        "profile_name": "work",
        "session_key": "session-key",
    }
    monkeypatch.setattr(server, "_authenticated_transport_principal", lambda: transport.auth_identity)
    monkeypatch.setattr(server, "_canonical_profile_request", lambda name: name)

    def lookup(params, rid):
        seen.update(params)
        return session, None

    monkeypatch.setattr(server, "_sess_nowait", lookup)
    token = server.bind_transport(transport)
    try:
        resolved, error = server._remote_exec_session(
            {"session_id": "sid", "profile": "work", "argv": []}, "rpc-1"
        )
    finally:
        server.reset_transport(token)
    assert seen["profile"] == "work"
    assert error is None
    assert resolved is not None


def test_remote_command_dispatch_rejects_foreign_session(monkeypatch):
    sid = "foreign-dispatch"
    session = {
        "owner_principal": "principal:other",
        "profile_name": "default",
        "session_key": "foreign-key",
    }
    transport = SimpleNamespace(auth_identity="principal:caller")
    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(server, "_canonical_profile_request", lambda name: "default")
    try:
        token = server.bind_transport(transport)
        try:
            result = server._methods["command.dispatch"](
                "rpc-2", {"name": "queue", "arg": "hello", "session_id": sid}
            )
        finally:
            server.reset_transport(token)
        assert result["error"]["code"] == 4032
    finally:
        server._sessions.pop(sid, None)


def test_remote_model_management_requires_session(monkeypatch):
    transport = SimpleNamespace(auth_identity="principal:caller")
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    token = server.bind_transport(transport)
    try:
        result = server._methods["model.disconnect"]("rpc-3", {"slug": "xai"})
    finally:
        server.reset_transport(token)
    assert result["error"]["code"] == 4032


def test_approval_entry_normalizes_capabilities_and_types():
    entry = _ApprovalEntry(
        {
            "command": {"secret": _SECRET},
            "description": ["bad"],
            "choices": ["bogus", "once", 3],
            "allow_session": "yes",
            "allow_permanent": True,
        }
    )
    assert entry.data["choices"] == ["once", "deny"]
    assert entry.data["allow_session"] is False
    assert entry.data["allow_permanent"] is True
    assert "command" not in entry.data
    assert "description" not in entry.data


def test_normalized_deny_choice_resolves_headless_entry():
    pending = submit_pending(
        "headless-review",
        {"choices": ["unknown"], "allow_session": False, "allow_permanent": False},
    )
    try:
        assert resolve_gateway_approval(
            "headless-review", "deny", request_id=pending["request_id"]
        ) == 1
    finally:
        from tools.approval import clear_session

        clear_session("headless-review")


def test_expired_headless_entries_are_pruned_before_inbox_read():
    import time
    from tools import approval as approval_module

    entry = _ApprovalEntry({"request_id": "expired-inbox", "choices": ["once", "deny"]}, timeout_seconds=1)
    entry.data["expires_at"] = time.time() - 1
    with approval_module._lock:
        approval_module._gateway_queues["expired-inbox"] = [entry]
    try:
        assert approval_module.list_gateway_approvals("expired-inbox") == []
        assert entry.event.is_set()
    finally:
        approval_module.clear_session("expired-inbox")


def test_headless_pending_can_wait_for_exact_response():
    from tools.approval import wait_for_pending_gateway_approval

    pending = submit_pending("headless-wait", {"choices": ["once", "deny"]})

    async def resolve():
        await asyncio.sleep(0.01)
        resolve_gateway_approval(
            "headless-wait", "once", request_id=pending["request_id"]
        )

    async def exercise():
        task = asyncio.create_task(resolve())
        result = await asyncio.to_thread(
            wait_for_pending_gateway_approval,
            "headless-wait",
            pending["request_id"],
            timeout_seconds=1,
        )
        await task
        return result

    try:
        result = asyncio.run(exercise())
        assert result["resolved"] is True
        assert result["choice"] == "once"
    finally:
        from tools.approval import clear_session

        clear_session("headless-wait")


def test_prompt_response_rejects_wrong_prompt_kind(monkeypatch):
    rid = "prompt-kind-review"
    ev = asyncio.Event()
    with server._prompt_lock:
        server._pending[rid] = ("sid", ev)
        server._pending_prompt_payloads[rid] = (
            "approval.request",
            {"request_id": rid},
        )
    try:
        token = server.bind_transport(server._stdio_transport)
        try:
            result = server._methods["clarify.respond"](
                "rpc-4", {"request_id": rid, "session_id": "sid", "answer": "yes"}
            )
        finally:
            server.reset_transport(token)
        assert result["error"]["code"] == 4002
    finally:
        with server._prompt_lock:
            server._pending.pop(rid, None)
            server._pending_prompt_payloads.pop(rid, None)


def test_remote_child_env_can_be_restricted_to_explicit_allowlist(monkeypatch):
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.setenv("REVIEW_FIXTURE_SECRET", _SECRET)
    env = hermes_subprocess_env(
        inherit_credentials=False,
        allowed_keys={"PATH", "PYTHONUTF8"},
    )
    assert "REVIEW_FIXTURE_SECRET" not in env
    assert env["PATH"]


def test_session_schema_declares_owner_principal():
    from hermes_state_common import SCHEMA_SQL

    sessions_sql = SCHEMA_SQL.split("CREATE TABLE IF NOT EXISTS messages", 1)[0]
    assert "owner_principal" in sessions_sql


def test_session_db_persists_owner_principal(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(
            "durable-owner",
            source="tui",
            profile_name="default",
            owner_principal="principal:default:user",
        )
        assert db.get_session("durable-owner")["owner_principal"] == "principal:default:user"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_get_response_applies_final_allowlist_and_redaction(tmp_path):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._response_store = ResponseStore(db_path=str(tmp_path / "responses.db"))
    adapter._response_store.put(
        "resp-safe",
        {
            "response": {
                "id": "resp-safe",
                "object": "response",
                "status": "completed",
                "output": [{"type": "message", "text": _SECRET, "unknown_secret": _SECRET}],
                "unknown_top_level": _SECRET,
            }
        },
    )
    request = MagicMock()
    request.headers = {}
    request.match_info = {"response_id": "resp-safe"}
    try:
        response = await adapter._handle_get_response(request)
        body = json.loads(response.text)
        assert _SECRET not in repr(body)
        assert "unknown_top_level" not in body
        assert "unknown_secret" not in repr(body)
    finally:
        adapter._response_store.close()
