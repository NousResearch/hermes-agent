"""Unit tests for the Matrix room-admin agent tools (create / leave / delete).

Mocks the raw CS-API call (_matrix_room_action), the createRoom HTTP session
(aiohttp.ClientSession) and the creds source (_matrix_creds) — we test OUR
logic (validation, body assembly, leave/forget sequencing, idempotency, error
surfacing, gating, registration), never a live Matrix server.
"""
import asyncio
import json
import sys
import types

import aiohttp
import pytest

from tools import matrix_room_tool as m


def _run(coro):
    # Run on the loop tests/conftest.py installs; asyncio.run() would replace it
    # and orphan it, surfacing as an "unclosed event loop" error in a later test.
    return asyncio.get_event_loop().run_until_complete(coro)


def _parse(result):
    """Tool handlers return JSON strings."""
    assert isinstance(result, str)
    return json.loads(result)


CURRENT_ROOM = "!r:hs"


def _bind_session(monkeypatch, **session):
    """Stand in for the gateway's per-turn session vars."""
    monkeypatch.setattr(m, "get_session_env", lambda name, default="": session.get(name, default))


@pytest.fixture()
def creds(monkeypatch):
    """Configured Matrix, a turn bound to CURRENT_ROOM, no cross-room opt-in, no live gateway."""
    monkeypatch.setattr(m, "_matrix_creds", lambda: ("https://matrix.example.org", "tok"))
    monkeypatch.setattr(m, "_live_matrix_adapter", lambda: None)
    monkeypatch.delenv("MATRIX_TOOLS_ALLOW_CROSS_ROOM", raising=False)
    _bind_session(monkeypatch, HERMES_SESSION_PLATFORM="matrix", HERMES_SESSION_CHAT_ID=CURRENT_ROOM)


class _FakeAdapter:
    def __init__(self):
        self._joined_rooms = {CURRENT_ROOM, "!other:hs"}
        self._dm_rooms = {CURRENT_ROOM: True, "!other:hs": False}


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

    @pytest.mark.parametrize("payload", ["<html>proxy error</html>", "not json"])
    def test_create_invalid_json(self, creds, monkeypatch, payload):
        _fake_client_session(monkeypatch, 200, payload)
        out = _parse(_run(m._handle_matrix_create_room({})))
        assert "createRoom returned invalid JSON" in out["error"]

    def test_create_json_that_is_not_an_object(self, creds, monkeypatch):
        _fake_client_session(monkeypatch, 200, "[1, 2]")
        out = _parse(_run(m._handle_matrix_create_room({})))
        assert "createRoom returned no room_id" in out["error"]

    def test_create_no_room_id_in_response(self, creds, monkeypatch):
        _fake_client_session(monkeypatch, 200, {"oops": True})
        out = _parse(_run(m._handle_matrix_create_room({})))
        assert "createRoom returned no room_id" in out["error"]

    def test_create_api_exception(self, creds, monkeypatch):
        def boom(**_kwargs):
            raise aiohttp.ClientConnectionError("conn reset")

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

    def test_leave_missing_room_id_outside_matrix(self, creds, monkeypatch):
        _bind_session(monkeypatch)  # CLI: no Matrix room to default to
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

    def test_delete_missing_room_id_outside_matrix(self, creds, monkeypatch):
        _bind_session(monkeypatch)  # CLI: no Matrix room to default to
        out = _parse(_run(m._handle_matrix_delete_room({})))
        assert "room_id is required" in out["error"]


# --------------------------------------------------------------------------
# transport failures + the raw CS-API call
# --------------------------------------------------------------------------
def _raising(*fail_on):
    """_matrix_room_action stand-in that raises for the given actions, 200s otherwise."""
    async def fake(homeserver, token, room_id, action, body=None):
        if action in fail_on:
            raise m.MatrixRoomRequestError(f"{action} unreachable")
        return 200, "{}"
    return fake


class TestTransport:
    def test_leave_request_exception(self, creds, monkeypatch):
        monkeypatch.setattr(m, "_matrix_room_action", _raising("leave"))
        out = _parse(_run(m._handle_matrix_leave_room({})))
        assert "matrix_leave_room request failed: leave unreachable" in out["error"]

    def test_delete_leave_exception(self, creds, monkeypatch):
        monkeypatch.setattr(m, "_matrix_room_action", _raising("leave"))
        out = _parse(_run(m._handle_matrix_delete_room({})))
        assert "matrix_delete_room leave failed: leave unreachable" in out["error"]

    def test_delete_forget_exception(self, creds, monkeypatch):
        monkeypatch.setattr(m, "_matrix_room_action", _raising("forget"))
        out = _parse(_run(m._handle_matrix_delete_room({})))
        assert "matrix_delete_room forget failed: forget unreachable" in out["error"]

    def test_room_action_posts_to_escaped_room_url(self, monkeypatch):
        calls = _fake_client_session(monkeypatch, 200, {})
        status, text = _run(m._matrix_room_action("https://hs", "tok", "!a:b/c", "leave", {"reason": "x"}))
        assert (status, text) == (200, "{}")
        assert calls == [{
            "url": "https://hs/_matrix/client/v3/rooms/%21a%3Ab%2Fc/leave",
            "headers": {"Authorization": "Bearer tok", "Content-Type": "application/json"},
            "body": {"reason": "x"},
        }]

    def test_room_action_sends_empty_body_by_default(self, monkeypatch):
        calls = _fake_client_session(monkeypatch, 200, {})
        _run(m._matrix_room_action("https://hs", "tok", "!a:b", "forget"))
        assert calls[0]["body"] == {}

    @pytest.mark.parametrize("exc,expected", [
        (aiohttp.ClientConnectionError("refused"), "refused"),
        (asyncio.TimeoutError(), "TimeoutError"),  # empty message -> type name, never a blank error
    ])
    def test_room_action_wraps_transport_errors(self, monkeypatch, exc, expected):
        def boom(**_kwargs):
            raise exc

        monkeypatch.setattr(aiohttp, "ClientSession", boom)
        with pytest.raises(m.MatrixRoomRequestError, match=expected):
            _run(m._matrix_room_action("https://hs", "tok", "!a:b", "leave"))

    def test_room_action_without_aiohttp(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "aiohttp", None)
        with pytest.raises(m.MatrixRoomRequestError, match="aiohttp not installed"):
            _run(m._matrix_room_action("https://hs", "tok", "!a:b", "leave"))

    def test_room_action_unexpected_errors_propagate(self, monkeypatch):
        # Only transport failures are converted; a programming error must surface, not become a tool error.
        def boom(**_kwargs):
            raise TypeError("bug")

        monkeypatch.setattr(aiohttp, "ClientSession", boom)
        with pytest.raises(TypeError):
            _run(m._matrix_room_action("https://hs", "tok", "!a:b", "leave"))

    def test_create_without_aiohttp(self, creds, monkeypatch):
        monkeypatch.setitem(sys.modules, "aiohttp", None)  # makes `import aiohttp` raise ImportError
        out = _parse(_run(m._handle_matrix_create_room({"name": "x"})))
        assert "aiohttp not installed" in out["error"]


# --------------------------------------------------------------------------
# room scoping: the current room by default, other rooms only on operator opt-in
# --------------------------------------------------------------------------
class TestRoomScope:
    @pytest.mark.parametrize("handler", ["_handle_matrix_leave_room", "_handle_matrix_delete_room"])
    def test_defaults_to_current_room(self, creds, monkeypatch, handler):
        fake = _recorder({"leave": (200, "{}"), "forget": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(getattr(m, handler)({})))
        assert out["room_id"] == CURRENT_ROOM
        assert {c["room_id"] for c in fake.calls} == {CURRENT_ROOM}

    @pytest.mark.parametrize("handler", ["_handle_matrix_leave_room", "_handle_matrix_delete_room"])
    def test_other_room_refused_without_opt_in(self, creds, monkeypatch, handler):
        fake = _recorder({"leave": (200, "{}"), "forget": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(getattr(m, handler)({"room_id": "!other:hs"})))
        assert "Refusing to act on !other:hs" in out["error"]
        assert f"the current room ({CURRENT_ROOM})" in out["error"]
        assert "MATRIX_TOOLS_ALLOW_CROSS_ROOM" in out["error"]
        assert fake.calls == []  # nothing sent to the homeserver

    @pytest.mark.parametrize("val", ["true", "1", "yes", "TRUE"])
    def test_other_room_allowed_with_opt_in(self, creds, monkeypatch, val):
        monkeypatch.setenv("MATRIX_TOOLS_ALLOW_CROSS_ROOM", val)
        fake = _recorder({"leave": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!other:hs"})))
        assert out["success"] is True
        assert fake.calls[0]["room_id"] == "!other:hs"

    @pytest.mark.parametrize("val", ["", "false", "no"])
    def test_opt_in_off_values(self, creds, monkeypatch, val):
        monkeypatch.setenv("MATRIX_TOOLS_ALLOW_CROSS_ROOM", val)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!other:hs"})))
        assert "Refusing" in out["error"]

    def test_explicit_current_room_needs_no_opt_in(self, creds, monkeypatch):
        fake = _recorder({"leave": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": f"  {CURRENT_ROOM} "})))
        assert out["room_id"] == CURRENT_ROOM

    def test_non_matrix_session_has_no_current_room(self, creds, monkeypatch):
        # A Telegram chat id must never be mistaken for a Matrix room.
        _bind_session(monkeypatch, HERMES_SESSION_PLATFORM="telegram", HERMES_SESSION_CHAT_ID=CURRENT_ROOM)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": CURRENT_ROOM})))
        assert "Refusing" in out["error"]
        assert "a Matrix conversation's own room" in out["error"]

    def test_cli_with_opt_in_may_name_any_room(self, creds, monkeypatch):
        _bind_session(monkeypatch)
        monkeypatch.setenv("MATRIX_TOOLS_ALLOW_CROSS_ROOM", "true")
        fake = _recorder({"leave": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!other:hs"})))
        assert out["success"] is True

    @pytest.mark.parametrize("platform", ["matrix", "Matrix", " MATRIX "])
    def test_current_room_platform_match_is_normalised(self, monkeypatch, platform):
        _bind_session(monkeypatch, HERMES_SESSION_PLATFORM=platform, HERMES_SESSION_CHAT_ID=" !x:hs ")
        assert m._current_matrix_room() == "!x:hs"

    def test_schemas_no_longer_require_room_id(self):
        assert m.MATRIX_LEAVE_ROOM_SCHEMA["parameters"]["required"] == []
        assert m.MATRIX_DELETE_ROOM_SCHEMA["parameters"]["required"] == []


# --------------------------------------------------------------------------
# reconciling the live adapter's membership caches after a leave
# --------------------------------------------------------------------------
class TestAdapterReconcile:
    @pytest.fixture()
    def adapter(self, creds, monkeypatch):
        fake = _FakeAdapter()
        monkeypatch.setattr(m, "_live_matrix_adapter", lambda: fake)
        return fake

    def test_leave_drops_room_from_caches(self, adapter, monkeypatch):
        monkeypatch.setattr(m, "_matrix_room_action", _recorder({"leave": (200, "{}")}))
        _run(m._handle_matrix_leave_room({}))
        assert adapter._joined_rooms == {"!other:hs"}
        assert adapter._dm_rooms == {"!other:hs": False}

    def test_failed_leave_keeps_caches(self, adapter, monkeypatch):
        monkeypatch.setattr(m, "_matrix_room_action", _recorder({"leave": (500, "boom")}))
        _run(m._handle_matrix_leave_room({}))
        assert CURRENT_ROOM in adapter._joined_rooms
        assert CURRENT_ROOM in adapter._dm_rooms

    @pytest.mark.parametrize("leave", [(200, "{}"), (403, '{"errcode":"M_FORBIDDEN"}')])
    def test_delete_drops_room_even_if_forget_fails(self, adapter, monkeypatch, leave):
        # The membership is gone once leave succeeds; a failed forget must not leave a stale cache.
        monkeypatch.setattr(m, "_matrix_room_action", _recorder({"leave": leave, "forget": (400, "x")}))
        out = _parse(_run(m._handle_matrix_delete_room({})))
        assert "Matrix forget error (400)" in out["error"]
        assert CURRENT_ROOM not in adapter._joined_rooms
        assert CURRENT_ROOM not in adapter._dm_rooms

    def test_delete_hard_leave_error_keeps_caches(self, adapter, monkeypatch):
        monkeypatch.setattr(m, "_matrix_room_action", _recorder({"leave": (500, "boom"), "forget": (200, "{}")}))
        _run(m._handle_matrix_delete_room({}))
        assert CURRENT_ROOM in adapter._joined_rooms

    def test_room_not_cached_is_fine(self, adapter):
        m._reconcile_adapter_after_leave("!never:hs")
        assert adapter._joined_rooms == {CURRENT_ROOM, "!other:hs"}

    def test_adapter_without_caches_is_fine(self, monkeypatch):
        monkeypatch.setattr(m, "_live_matrix_adapter", lambda: object())
        m._reconcile_adapter_after_leave(CURRENT_ROOM)  # must not raise

    def test_no_gateway_is_fine(self, monkeypatch):
        monkeypatch.setattr(m, "_live_matrix_adapter", lambda: None)
        m._reconcile_adapter_after_leave(CURRENT_ROOM)  # must not raise


# --------------------------------------------------------------------------
# live adapter lookup + creds
# --------------------------------------------------------------------------
class TestLiveAdapter:
    @staticmethod
    def _runner_ref(monkeypatch, ref):
        # Stub gateway.run: importing the real module opens an event loop + sockets at import time.
        stub = types.ModuleType("gateway.run")
        stub._gateway_runner_ref = ref
        monkeypatch.setitem(sys.modules, "gateway.run", stub)

    def test_returns_matrix_adapter(self, monkeypatch):
        from gateway.config import Platform
        adapter = _FakeAdapter()
        runner = types.SimpleNamespace(adapters={Platform.MATRIX: adapter})
        self._runner_ref(monkeypatch, lambda: runner)
        assert m._live_matrix_adapter() is adapter

    def test_no_runner(self, monkeypatch):
        self._runner_ref(monkeypatch, lambda: None)
        assert m._live_matrix_adapter() is None

    def test_gateway_not_importable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "gateway.run", None)  # `from gateway.run import ...` -> ImportError
        assert m._live_matrix_adapter() is None

    def test_creds_prefer_live_adapter(self, monkeypatch):
        adapter = _FakeAdapter()
        adapter._homeserver = "https://live.example/"
        adapter._access_token = "live-tok"
        monkeypatch.setattr(m, "_live_matrix_adapter", lambda: adapter)
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://env.example")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "env-tok")
        assert m._matrix_creds() == ("https://live.example", "live-tok")

    def test_creds_fall_back_to_env(self, monkeypatch):
        monkeypatch.setattr(m, "_live_matrix_adapter", lambda: None)
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://env.example/")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "env-tok")
        assert m._matrix_creds() == ("https://env.example", "env-tok")


# --------------------------------------------------------------------------
# gating
# --------------------------------------------------------------------------
class TestGate:
    @pytest.mark.parametrize("val,expected", [
        ("true", True), ("1", True), ("yes", True), ("TRUE", True), (" yes ", True),
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

    def test_only_the_matrix_bundle_exposes_them(self):
        import toolsets
        room_tools = {"matrix_create_room", "matrix_leave_room", "matrix_delete_room"}
        assert room_tools.isdisjoint(toolsets._HERMES_CORE_TOOLS)
        assert room_tools <= set(toolsets.resolve_toolset("hermes-matrix", include_registry=False))
        for name, spec in toolsets.TOOLSETS.items():
            if name == "hermes-matrix" or "hermes-matrix" in spec.get("includes", []):
                continue  # the Matrix bundle itself, or an aggregate that includes it on purpose
            exposed = room_tools & set(toolsets.resolve_toolset(name, include_registry=False))
            assert not exposed, f"{name} exposes {sorted(exposed)}"

    def test_gates_wired_as_check_fn(self):
        from tools.registry import registry
        assert registry._tools["matrix_create_room"].check_fn is m._check_matrix_create_room
        assert registry._tools["matrix_leave_room"].check_fn is m._check_matrix_room_admin
        assert registry._tools["matrix_delete_room"].check_fn is m._check_matrix_room_admin
