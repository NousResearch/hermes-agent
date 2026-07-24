"""Unit tests for the matrix_leave_room + matrix_delete_room agent tools.

Mocks the raw CS-API call (_matrix_room_action) and the creds source
(_matrix_creds) — we test OUR logic (validation, leave/forget sequencing,
idempotency, error surfacing, gating), never a live Matrix server.
"""
import asyncio
import json

import pytest

from tools import matrix_room_tool as m


def _run(coro):
    return asyncio.run(coro)


def _parse(result):
    """Tool handlers return JSON strings."""
    assert isinstance(result, str)
    return json.loads(result)


@pytest.fixture()
def creds(monkeypatch):
    monkeypatch.setattr(m, "_matrix_creds", lambda: ("https://matrix.example.org", "tok"))


def _recorder(responses):
    """Build an async stand-in for _matrix_room_action returning canned
    (status, text) per action, recording every call."""
    calls = []

    async def fake(homeserver, token, room_id, action, body=None):
        calls.append({"room_id": room_id, "action": action, "body": body})
        return responses[action]

    fake.calls = calls
    return fake


# --------------------------------------------------------------------------
# matrix_leave_room
# --------------------------------------------------------------------------
class TestLeaveRoom:
    def test_leave_success(self, creds, monkeypatch):
        fake = _recorder({"leave": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!r:hs"})))
        assert out["success"] is True
        assert out["room_id"] == "!r:hs"
        assert out["action"] == "leave"
        assert [c["action"] for c in fake.calls] == ["leave"]

    def test_leave_passes_reason(self, creds, monkeypatch):
        fake = _recorder({"leave": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        _run(m._handle_matrix_leave_room({"room_id": "!r:hs", "reason": "cleanup"}))
        assert fake.calls[0]["body"] == {"reason": "cleanup"}

    def test_leave_missing_room_id(self, creds):
        out = _parse(_run(m._handle_matrix_leave_room({})))
        assert "room_id is required" in out["error"]

    def test_leave_not_configured(self, monkeypatch):
        monkeypatch.setattr(m, "_matrix_creds", lambda: ("", ""))
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!r:hs"})))
        assert "Matrix not configured" in out["error"]

    def test_leave_http_error(self, creds, monkeypatch):
        fake = _recorder({"leave": (404, '{"errcode":"M_NOT_FOUND"}')})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!r:hs"})))
        assert "Matrix leave error (404)" in out["error"]


# --------------------------------------------------------------------------
# matrix_delete_room  (leave + forget)
# --------------------------------------------------------------------------
class TestDeleteRoom:
    def test_delete_leave_then_forget(self, creds, monkeypatch):
        fake = _recorder({"leave": (200, "{}"), "forget": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_delete_room({"room_id": "!r:hs"})))
        assert out["success"] is True
        assert out["action"] == "leave+forget"
        assert [c["action"] for c in fake.calls] == ["leave", "forget"]

    def test_delete_tolerates_already_left(self, creds, monkeypatch):
        # leaving a room you're not in -> 403 M_FORBIDDEN; delete must still forget
        fake = _recorder({
            "leave": (403, '{"errcode":"M_FORBIDDEN","error":"not in room"}'),
            "forget": (200, "{}"),
        })
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_delete_room({"room_id": "!r:hs"})))
        assert out["success"] is True
        assert [c["action"] for c in fake.calls] == ["leave", "forget"]

    def test_delete_leave_hard_error_skips_forget(self, creds, monkeypatch):
        fake = _recorder({"leave": (500, "boom"), "forget": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_delete_room({"room_id": "!r:hs"})))
        assert "Matrix leave (during delete) error (500)" in out["error"]
        assert [c["action"] for c in fake.calls] == ["leave"]  # forget NOT attempted

    def test_delete_forget_error(self, creds, monkeypatch):
        fake = _recorder({"leave": (200, "{}"), "forget": (400, '{"errcode":"M_UNKNOWN"}')})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_delete_room({"room_id": "!r:hs"})))
        assert "Matrix forget error (400)" in out["error"]

    def test_delete_missing_room_id(self, creds):
        out = _parse(_run(m._handle_matrix_delete_room({})))
        assert "room_id is required" in out["error"]


# --------------------------------------------------------------------------
# gating
# --------------------------------------------------------------------------
class TestGate:
    @pytest.mark.parametrize("val,expected", [
        ("true", True), ("1", True), ("yes", True), ("TRUE", True),
        ("", False), ("false", False), ("no", False),
    ])
    def test_room_admin_gate(self, monkeypatch, val, expected):
        monkeypatch.setenv("MATRIX_TOOLS_ALLOW_ROOM_CREATE", val)
        assert m._check_matrix_room_admin() is expected

    def test_gate_unset(self, monkeypatch):
        monkeypatch.delenv("MATRIX_TOOLS_ALLOW_ROOM_CREATE", raising=False)
        assert m._check_matrix_room_admin() is False


# --------------------------------------------------------------------------
# registry wiring — tools are actually registered under hermes-matrix
# --------------------------------------------------------------------------
class TestRegistration:
    def test_tools_registered(self):
        from tools.registry import registry
        assert "matrix_leave_room" in registry._tools
        assert "matrix_delete_room" in registry._tools
        assert registry._tools["matrix_leave_room"].toolset == "hermes-matrix"
        assert registry._tools["matrix_delete_room"].toolset == "hermes-matrix"
