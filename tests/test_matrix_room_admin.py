"""Unit tests for the Matrix room-admin agent tools (create / leave / delete).

Mocks the raw CS-API call (_matrix_room_action), the createRoom HTTP session
(aiohttp.ClientSession) and the creds source (_matrix_creds) — we test OUR
logic (validation, body assembly, leave/forget sequencing, idempotency, error
surfacing, gating, registration), never a live Matrix server.
"""
import asyncio
import json

import aiohttp
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


class _FakeResponse:
    """Stand-in for aiohttp's response: usable as an async context manager,
    answers .status / .text() / .json() from a canned payload."""

    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def text(self):
        return json.dumps(self._payload) if isinstance(self._payload, dict) else str(self._payload)

    async def json(self):
        return self._payload if isinstance(self._payload, dict) else json.loads(await self.text())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Stand-in for aiohttp.ClientSession: async CM whose .post() records the
    request and hands back one canned _FakeResponse."""

    def __init__(self, response, calls):
        self._response = response
        self._calls = calls

    def post(self, url, headers=None, json=None):
        self._calls.append({"url": url, "headers": headers, "body": json})
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _fake_client_session(monkeypatch, status, payload):
    """Patch aiohttp.ClientSession to answer every POST with *status*/*payload*
    and record (url, headers, body) per request. Returns the call list."""
    calls = []
    response = _FakeResponse(status, payload)
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **_kwargs: _FakeSession(response, calls))
    return calls


# --------------------------------------------------------------------------
# matrix_create_room
# --------------------------------------------------------------------------
class TestCreateRoom:
    def test_create_success(self, creds, monkeypatch):
        calls = _fake_client_session(monkeypatch, 200, {"room_id": "!new:example.org"})
        out = _parse(
            _run(
                m._handle_matrix_create_room(
                    {"name": "ops", "topic": "t", "invite": ["@a:example.org"], "is_direct": True}
                )
            )
        )
        assert out["success"] is True
        assert out["room_id"] == "!new:example.org"
        assert out["invited"] == ["@a:example.org"]
        assert out["preset"] == "private_chat"
        assert out["encrypted"] is False
        req = calls[0]
        assert req["url"] == "https://matrix.example.org/_matrix/client/v3/createRoom"
        assert req["headers"]["Authorization"] == "Bearer tok"
        body = req["body"]
        assert body["name"] == "ops"
        assert body["topic"] == "t"
        assert body["invite"] == ["@a:example.org"]
        assert body["is_direct"] is True
        assert "initial_state" not in body

    def test_create_encrypted_adds_megolm_state(self, creds, monkeypatch):
        calls = _fake_client_session(monkeypatch, 200, {"room_id": "!e:example.org"})
        _run(m._handle_matrix_create_room({"encrypted": True}))
        body = calls[0]["body"]
        assert body["initial_state"] == [
            {
                "type": "m.room.encryption",
                "state_key": "",
                "content": {"algorithm": "m.megolm.v1.aes-sha2"},
            }
        ]

    def test_create_public_requires_flag(self, creds, monkeypatch):
        monkeypatch.delenv("MATRIX_ALLOW_PUBLIC_ROOMS", raising=False)
        out = _parse(_run(m._handle_matrix_create_room({"preset": "public_chat"})))
        assert "MATRIX_ALLOW_PUBLIC_ROOMS" in out["error"]

    def test_create_public_allowed_with_flag(self, creds, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOW_PUBLIC_ROOMS", "true")
        calls = _fake_client_session(monkeypatch, 201, {"room_id": "!pub:example.org"})
        out = _parse(_run(m._handle_matrix_create_room({"preset": "public_chat"})))
        assert out["success"] is True
        assert calls[0]["body"]["preset"] == "public_chat"

    def test_create_not_configured(self, monkeypatch):
        monkeypatch.setattr(m, "_matrix_creds", lambda: ("", ""))
        out = _parse(_run(m._handle_matrix_create_room({})))
        assert "Matrix not configured" in out["error"]

    def test_create_http_error(self, creds, monkeypatch):
        _fake_client_session(monkeypatch, 403, {"errcode": "M_FORBIDDEN"})
        out = _parse(_run(m._handle_matrix_create_room({})))
        assert "Matrix createRoom error (403)" in out["error"]

    def test_create_no_room_id_in_response(self, creds, monkeypatch):
        _fake_client_session(monkeypatch, 200, {"oops": True})
        out = _parse(_run(m._handle_matrix_create_room({})))
        assert "createRoom returned no room_id" in out["error"]

    def test_create_api_exception(self, creds, monkeypatch):
        def boom(**_kwargs):
            raise RuntimeError("conn reset")

        monkeypatch.setattr(aiohttp, "ClientSession", boom)
        out = _parse(_run(m._handle_matrix_create_room({})))
        assert "matrix_create_room request failed" in out["error"]
        assert "conn reset" in out["error"]


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
        assert m._check_matrix_create_room() is expected

    def test_gate_unset(self, monkeypatch):
        monkeypatch.delenv("MATRIX_TOOLS_ALLOW_ROOM_CREATE", raising=False)
        assert m._check_matrix_room_admin() is False
        assert m._check_matrix_create_room() is False


# --------------------------------------------------------------------------
# registry wiring — tools are actually registered under hermes-matrix
# --------------------------------------------------------------------------
class TestRegistration:
    def test_tools_registered(self):
        from tools.registry import registry
        for name in ("matrix_create_room", "matrix_leave_room", "matrix_delete_room"):
            assert name in registry._tools
            assert registry._tools[name].toolset == "hermes-matrix"

    def test_gates_wired_as_check_fn(self):
        from tools.registry import registry
        assert registry._tools["matrix_create_room"].check_fn is m._check_matrix_create_room
        assert registry._tools["matrix_leave_room"].check_fn is m._check_matrix_room_admin
        assert registry._tools["matrix_delete_room"].check_fn is m._check_matrix_room_admin
