from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import tui_gateway.server as server
from gateway.platforms.api_server_runs import _safe_api_status
from tools import approval as approval_module


def _remote_transport(user_id: str = "caller"):
    class _Transport:
        def __init__(self):
            self.auth_identity = {"user_id": user_id, "provider": "dashboard"}

    return _Transport()


def _owned_session(*, owner: str = "dashboard:caller", profile: str = "default", profile_home: str | None = None, agent=None):
    return {
        "agent": agent,
        "agent_ready": None,
        "created_at": 123.0,
        "history": [],
        "history_lock": threading.RLock(),
        "last_active": 123.0,
        "owner_principal": owner,
        "profile_home": profile_home,
        "profile_name": profile,
        "running": False,
        "session_key": "session-key",
        "source": "dashboard",
        "transport": server._stdio_transport,
    }


def test_safe_api_status_malformed_choices_are_deny_only():
    safe = _safe_api_status({
        "approval": {
            "request_id": "req-safe",
            "choices": ["unexpected-choice"],
            "allow_session": "not-a-bool",
            "allow_permanent": 1,
        }
    })
    assert safe["approval"]["choices"] == ["deny"]
    assert "allow_session" not in safe["approval"]
    assert "allow_permanent" not in safe["approval"]


def test_tui_approval_snapshot_malformed_capabilities_are_deny_only():
    safe = server._approval_request_payload({
        "request_id": "req-snapshot",
        "choices": ["session", "always"],
        "allow_session": "not-a-bool",
        "allow_permanent": 1,
    })
    assert safe["choices"] == ["deny"]
    missing_flags = server._approval_request_payload({
        "request_id": "req-snapshot-missing",
        "choices": ["session", "always"],
    })
    assert missing_flags["choices"] == ["deny"]
    assert missing_flags.get("allow_session") is not True
    assert missing_flags.get("allow_permanent") is not True


def test_remote_session_close_requires_owner_and_preserves_foreign_session(monkeypatch):
    sid = "foreign-close"
    session = _owned_session(owner="dashboard:other")
    server._sessions[sid] = session
    monkeypatch.setattr(server, "current_transport", lambda: _remote_transport())
    try:
        response = server._methods["session.close"]("close-foreign", {"session_id": sid, "profile": "default"})
        assert response["error"]["code"] == 4032
        assert server._sessions[sid] is session
    finally:
        server._sessions.pop(sid, None)


def test_session_close_requires_a_nonempty_session_id():
    response = server._methods["session.close"]("close-missing", {})
    assert response["error"]["code"] == 4006


def test_remote_session_events_since_requires_owner_before_replay(monkeypatch):
    sid = "foreign-replay"
    server._sessions[sid] = _owned_session(owner="dashboard:other")
    monkeypatch.setattr(server, "current_transport", lambda: _remote_transport())

    def fail_if_replayed(*_args, **_kwargs):
        raise AssertionError("foreign replay was accessed")

    monkeypatch.setattr("tui_gateway.event_replay.events_since", fail_if_replayed)
    try:
        response = server._methods["session.events.since"](
            "replay-foreign", {"session_id": sid, "profile": "default", "last_seen": 0}
        )
        assert response["error"]["code"] == 4032
    finally:
        server._sessions.pop(sid, None)


def test_remote_session_activate_preserves_requested_noncurrent_profile(monkeypatch, tmp_path):
    sid = "activate-work"
    profile_home = tmp_path / "profile-work"
    profile_home.mkdir()
    session = _owned_session(
        profile="work",
        profile_home=str(profile_home),
        agent=None,
    )
    session["session_key"] = sid
    server._sessions[sid] = session
    monkeypatch.setattr(server, "current_transport", lambda: _remote_transport())
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(server, "_profile_home", lambda name: profile_home if name == "work" else None)
    try:
        response = server._methods["session.activate"](
            "activate-work", {"session_id": sid, "profile": "work", "omit_messages": False}
        )
        assert "error" not in response
        assert response["result"]["info"]["profile_name"] == "work"
    finally:
        server._sessions.pop(sid, None)


def test_remote_global_mutations_require_owned_session(monkeypatch):
    transport = _remote_transport()
    monkeypatch.setattr(server, "current_transport", lambda: transport)
    monkeypatch.setattr(server, "_write_config_key", lambda *_args, **_kwargs: pytest.fail("global config write"))

    config_response = server._methods["config.set"](
        "config-no-session", {"key": "busy", "value": "queue"}
    )
    assert config_response["error"]["code"] == 4032

    reload_response = server._methods["reload.mcp"](
        "reload-no-session", {"confirm": True}
    )
    assert reload_response["error"]["code"] == 4032

    configure_response = server._methods["tools.configure"](
        "configure-no-session", {"action": "enable", "names": ["terminal"]}
    )
    assert configure_response["error"]["code"] == 4032


def test_cli_and_noninteractive_denial_messages_redact_description(monkeypatch):
    secret_marker = "TEST_SENTINEL"
    monkeypatch.setattr(approval_module, "_is_single_query_approval_context", lambda: True)
    monkeypatch.setattr(approval_module, "_get_single_query_approval_mode", lambda: "deny")
    result = approval_module._run_approval_gate(
        pattern_key="test-pattern",
        description=f"Authorization: Bearer {secret_marker}",
        display_target="test target",
        cron_deny_message=f"cron Authorization: Bearer {secret_marker}",
        single_query_deny_message=f"single Authorization: Bearer {secret_marker}",
        autoapprove_log_prefix="test",
    )
    assert secret_marker not in result["message"]
    assert secret_marker not in result["description"]


def test_compute_host_approval_response_is_routed_to_child(monkeypatch):
    sid = "compute-approval"
    session = _owned_session()
    session["session_key"] = "child-approval-key"
    session["approval_profile_home"] = str(server._approval_profile_home_for_name("default"))
    session["_compute_host_active"] = True
    server._sessions[sid] = session
    calls = []

    class _Supervisor:
        def control(self, control_sid, *, route_name, payload=None, wait=True, timeout=30.0):
            calls.append((control_sid, route_name, payload, wait))
            return {"type": "control.ack", "result": {"resolved": 1}}

    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda: _Supervisor())
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _session: True)
    try:
        response = server._methods["approval.respond"](
            "approval-child",
            {
                "session_id": sid,
                "profile": "default",
                "request_id": "child-request",
                "choice": "once",
            },
        )
        assert response["result"] == {"resolved": 1}
        assert calls and calls[0][0:2] == (sid, "approval.respond")
        assert calls[0][2]["params"]["request_id"] == "child-request"
    finally:
        server._sessions.pop(sid, None)
