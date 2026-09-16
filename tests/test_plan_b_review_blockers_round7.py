"""Round-seven regressions for remote ownership on durable/session controls."""

from __future__ import annotations

from types import SimpleNamespace

import tui_gateway.server as server


class _RemoteTransport:
    auth_identity = {"user_id": "operator", "provider": "dashboard"}


def _session(owner: str, sid: str):
    return {
        "agent": SimpleNamespace(session_id=sid),
        "owner_principal": owner,
        "profile_name": "default",
        "profile_home": None,
        "session_key": f"key-{sid}",
        "stored_session_id": f"stored-{sid}",
        "cwd": ".",
        "history_lock": __import__("threading").RLock(),
    }


def test_remote_owned_session_ids_exclude_foreign_same_profile_sessions(monkeypatch):
    own = "round7-own"
    foreign = "round7-foreign"
    server._sessions[own] = _session("dashboard:operator", own)
    server._sessions[foreign] = _session("dashboard:other", foreign)
    monkeypatch.setattr(server, "current_transport", lambda: _RemoteTransport())
    try:
        visible = server._remote_owned_session_ids({"profile": "default"}, "rpc-list")
        assert own in visible
        assert "key-round7-own" in visible
        assert foreign not in visible
        assert "key-round7-foreign" not in visible
    finally:
        server._sessions.pop(own, None)
        server._sessions.pop(foreign, None)


def test_set_hidden_does_not_fall_through_after_foreign_owner_rejection(monkeypatch):
    sid = "round7-hidden-foreign"
    server._sessions[sid] = _session("dashboard:other", sid)
    monkeypatch.setattr(server, "current_transport", lambda: _RemoteTransport())
    try:
        response = server._methods["session.set_hidden"](
            "rpc-hidden", {"session_id": sid, "hidden": True}
        )
        assert response["error"]["code"] == 4032
    finally:
        server._sessions.pop(sid, None)


def test_config_set_does_not_control_foreign_session(monkeypatch):
    sid = "round7-config-foreign"
    server._sessions[sid] = _session("dashboard:other", sid)
    monkeypatch.setattr(server, "current_transport", lambda: _RemoteTransport())
    try:
        response = server._methods["config.set"](
            "rpc-config", {"session_id": sid, "key": "model", "value": "fixture-model"}
        )
        assert response["error"]["code"] == 4032
    finally:
        server._sessions.pop(sid, None)


def test_workspace_move_rejects_foreign_live_session_before_db_write(monkeypatch):
    sid = "round7-workspace-foreign"
    server._sessions[sid] = _session("dashboard:other", sid)
    monkeypatch.setattr(server, "current_transport", lambda: _RemoteTransport())
    try:
        response = server._methods["session.workspace.move"](
            "rpc-workspace",
            {"session_key": f"key-{sid}", "cwd": ".", "profile": "default"},
        )
        assert response["error"]["code"] == 4032
    finally:
        server._sessions.pop(sid, None)
