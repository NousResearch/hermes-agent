"""Unit tests for the Matrix room-admin agent tools (create / leave / delete).

Mocks the raw CS-API calls (_matrix_create_room_api / _matrix_room_action), the
creds source (_matrix_creds), and the live-adapter lookup (_live_adapter) — we
test OUR logic (validation, room-boundary authorization, leave/forget
sequencing, idempotency, error surfacing, adapter-cache reconciliation,
gating, toolset scoping), never a live Matrix server.
"""
import asyncio
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from gateway.config import PlatformConfig
from tools import matrix_room_tool as m


def _run(coro):
    return asyncio.run(coro)


def _parse(result):
    """Tool handlers return JSON strings."""
    assert isinstance(result, str)
    return json.loads(result)


def _assert_error(out, needle):
    """tool_error() emits {"error": ...} with no 'success' key — so an error
    payload is *error-only*. Fail the test unless the message is present and
    there is no success=True on the other side of the split."""
    assert out.get("error"), f"expected an error payload, got {out!r}"
    assert needle in out["error"], f"{needle!r} not in {out['error']!r}"
    assert out.get("success") is not True


@pytest.fixture(autouse=True)
def _isolated_session_context(monkeypatch):
    """Reset per-task session contextvars and session/env scoping vars so each
    test starts from "no current room bound, no allowlist, no escape hatch"
    (the strict CLI/standalone default) unless a test opts in explicitly."""
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("MATRIX_ALLOWED_ROOMS", raising=False)
    monkeypatch.delenv("MATRIX_TOOLS_ALLOW_ANY_ROOM", raising=False)
    monkeypatch.delenv("MATRIX_ALLOW_PUBLIC_ROOMS", raising=False)
    try:
        from gateway.session_context import reset_session_vars

        reset_session_vars()
    except Exception:
        pass


@pytest.fixture()
def creds(monkeypatch):
    monkeypatch.setattr(m, "_matrix_creds", lambda: ("https://matrix.example.org", "tok"))


@pytest.fixture()
def default_authz(monkeypatch):
    """Authorize-everything environment: a bound current room, no allowlist,
    no live adapter. Individual tests narrow this as needed."""
    monkeypatch.setattr(m, "_current_room", lambda: "!cur:hs")
    monkeypatch.setattr(m, "_allowed_room_ids", lambda: set())
    monkeypatch.setattr(m, "_joined_rooms", lambda: None)


def _recorder(responses):
    """Build an async stand-in for _matrix_room_action returning canned
    (status, text) per action, recording every call."""
    calls = []

    async def fake(homeserver, token, room_id, action, body=None):
        calls.append({"room_id": room_id, "action": action, "body": body})
        return responses[action]

    fake.calls = calls
    return fake


def _api_recorder(response):
    """Build an async stand-in for _matrix_create_room_api returning a canned
    (status, data) pair and recording every request body."""
    calls = []

    async def fake(homeserver, token, body):
        calls.append({"homeserver": homeserver, "token": token, "body": body})
        return response

    fake.calls = calls
    return fake


def _make_adapter():
    """Create a MatrixAdapter with mocked config (mirrors
    tests/gateway/test_matrix.py::_make_adapter)."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
        },
    )
    return MatrixAdapter(config)


# --------------------------------------------------------------------------
# matrix_create_room
# --------------------------------------------------------------------------
class TestCreateRoom:
    def test_create_success(self, creds, monkeypatch):
        fake = _api_recorder((200, {"room_id": "!new:hs"}))
        monkeypatch.setattr(m, "_matrix_create_room_api", fake)
        out = _parse(
            _run(
                m._handle_matrix_create_room(
                    {"name": "ops", "topic": "t", "invite": ["@a:hs"], "is_direct": True}
                )
            )
        )
        assert out["success"] is True
        assert out["room_id"] == "!new:hs"
        assert out["invited"] == ["@a:hs"]
        assert out["preset"] == "private_chat"
        assert out["encrypted"] is False
        req = fake.calls[0]
        assert req["homeserver"] == "https://matrix.example.org"
        assert req["token"] == "tok"
        body = req["body"]
        assert body["name"] == "ops"
        assert body["topic"] == "t"
        assert body["invite"] == ["@a:hs"]
        assert body["is_direct"] is True
        assert "initial_state" not in body

    def test_create_encrypted_adds_megolm_state(self, creds, monkeypatch):
        fake = _api_recorder((200, {"room_id": "!e:hs"}))
        monkeypatch.setattr(m, "_matrix_create_room_api", fake)
        _run(m._handle_matrix_create_room({"encrypted": True}))
        body = fake.calls[0]["body"]
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
        fake = _api_recorder((201, {"room_id": "!pub:hs"}))
        monkeypatch.setattr(m, "_matrix_create_room_api", fake)
        out = _parse(_run(m._handle_matrix_create_room({"preset": "public_chat"})))
        assert out["success"] is True
        assert fake.calls[0]["body"]["preset"] == "public_chat"

    def test_create_not_configured(self, monkeypatch):
        monkeypatch.setattr(m, "_matrix_creds", lambda: ("", ""))
        out = _parse(_run(m._handle_matrix_create_room({})))
        assert "Matrix not configured" in out["error"]

    def test_create_http_error(self, creds, monkeypatch):
        fake = _api_recorder((403, {"errcode": "M_FORBIDDEN"}))
        monkeypatch.setattr(m, "_matrix_create_room_api", fake)
        out = _parse(_run(m._handle_matrix_create_room({})))
        assert "Matrix createRoom error (403)" in out["error"]

    def test_create_no_room_id_in_response(self, creds, monkeypatch):
        fake = _api_recorder((200, {"oops": True}))
        monkeypatch.setattr(m, "_matrix_create_room_api", fake)
        out = _parse(_run(m._handle_matrix_create_room({})))
        assert "createRoom returned no room_id" in out["error"]

    def test_create_api_exception(self, creds, monkeypatch):
        async def boom(homeserver, token, body):
            raise RuntimeError("conn reset")

        monkeypatch.setattr(m, "_matrix_create_room_api", boom)
        out = _parse(_run(m._handle_matrix_create_room({})))
        assert "matrix_create_room request failed" in out["error"]
        assert "conn reset" in out["error"]


# --------------------------------------------------------------------------
# matrix_leave_room
# --------------------------------------------------------------------------
class TestLeaveRoom:
    def test_leave_success(self, creds, default_authz, monkeypatch):
        fake = _recorder({"leave": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!cur:hs"})))
        assert out["success"] is True
        assert out["room_id"] == "!cur:hs"
        assert out["action"] == "leave"
        assert [c["action"] for c in fake.calls] == ["leave"]

    def test_leave_passes_reason(self, creds, default_authz, monkeypatch):
        fake = _recorder({"leave": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        _run(m._handle_matrix_leave_room({"room_id": "!cur:hs", "reason": "cleanup"}))
        assert fake.calls[0]["body"] == {"reason": "cleanup"}

    def test_leave_missing_room_id(self, creds, default_authz):
        out = _parse(_run(m._handle_matrix_leave_room({})))
        assert "room_id is required" in out["error"]

    def test_leave_not_configured(self, default_authz, monkeypatch):
        monkeypatch.setattr(m, "_matrix_creds", lambda: ("", ""))
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!cur:hs"})))
        assert "Matrix not configured" in out["error"]

    def test_leave_http_error(self, creds, default_authz, monkeypatch):
        fake = _recorder({"leave": (404, '{"errcode":"M_NOT_FOUND"}')})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!cur:hs"})))
        assert "Matrix leave error (404)" in out["error"]


# --------------------------------------------------------------------------
# Room authorization (the gateway/session.py Matrix room boundary)
# --------------------------------------------------------------------------
class TestRoomAuthorization:
    def test_leave_rejects_cross_room_when_current_bound(self, creds, default_authz, monkeypatch):
        fake = _recorder({"leave": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!other:hs"})))
        _assert_error(out, "outside this turn's scope")
        assert "!other:hs" in out["error"] and "!cur:hs" in out["error"]
        assert fake.calls == []  # no API call for an unauthorized room

    def test_leave_allows_current_room(self, creds, default_authz, monkeypatch):
        fake = _recorder({"leave": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!cur:hs"})))
        assert out["success"] is True
        assert fake.calls and fake.calls[0]["room_id"] == "!cur:hs"

    def test_leave_allows_joined_room(self, creds, default_authz, monkeypatch):
        # This turn is in !cur:hs, but the bot is still a member of !other:hs —
        # a legitimate leave target.
        monkeypatch.setattr(m, "_joined_rooms", lambda: {"!other:hs"})
        fake = _recorder({"leave": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!other:hs"})))
        assert out["success"] is True
        assert [c["room_id"] for c in fake.calls] == ["!other:hs"]

    def test_leave_allows_allowlisted_room(self, creds, default_authz, monkeypatch):
        monkeypatch.setattr(m, "_allowed_room_ids", lambda: {"!allow:hs"})
        fake = _recorder({"leave": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!allow:hs"})))
        assert out["success"] is True

    def test_leave_allowlist_beats_current_room(self, creds, default_authz, monkeypatch):
        # An allowlisted room is a valid target even from another room.
        monkeypatch.setattr(m, "_allowed_room_ids", lambda: {"!allow:hs"})
        monkeypatch.setattr(m, "_joined_rooms", lambda: None)
        fake = _recorder({"leave": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!allow:hs"})))
        assert out["success"] is True
        assert out["room_id"] == "!allow:hs"

    def test_delete_rejects_cross_room_when_current_bound(self, creds, default_authz, monkeypatch):
        fake = _recorder({"leave": (200, "{}"), "forget": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_delete_room({"room_id": "!other:hs"})))
        _assert_error(out, "outside this turn's scope")
        assert fake.calls == []  # neither leave nor forget attempted

    # --- fail-closed contract: no scope established => deny, not allow -------

    def test_no_context_no_allowlist_is_denied(self, creds, monkeypatch):
        # No current room bound, no allowlist, no live adapter (cron/CLI/
        # one-shot): there is nothing to scope against, so a destructive
        # action must be DENIED (fail closed), not silently allowed.
        monkeypatch.setattr(m, "_current_room", lambda: "")
        monkeypatch.setattr(m, "_allowed_room_ids", lambda: set())
        monkeypatch.setattr(m, "_joined_rooms", lambda: None)
        fake = _recorder({"leave": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!any:hs"})))
        _assert_error(out, "MATRIX_TOOLS_ALLOW_ANY_ROOM")
        assert fake.calls == []  # no API call once authorization fails

    def test_no_context_undeclared_when_allowlist_set(self, creds, monkeypatch):
        # A NON-empty allowlist is a whitelist: it establishes scope, so a
        # room outside it is denied with the allowlist-specific message.
        monkeypatch.setattr(m, "_current_room", lambda: "")
        monkeypatch.setattr(m, "_allowed_room_ids", lambda: {"!listed:hs"})
        monkeypatch.setattr(m, "_joined_rooms", lambda: None)
        fake = _recorder({"leave": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!any:hs"})))
        _assert_error(out, "MATRIX_ALLOWED_ROOMS")
        assert fake.calls == []

    def test_unavailable_adapter_does_not_widen_scope(self, creds, monkeypatch):
        # Adapter unavailable (_joined_rooms() is None) must NOT be read as
        # "no scope, allow": with an empty allowlist and no current room the
        # deny must still stand.
        monkeypatch.setattr(m, "_current_room", lambda: "")
        monkeypatch.setattr(m, "_allowed_room_ids", lambda: set())
        monkeypatch.setattr(m, "_joined_rooms", lambda: None)
        assert m._authorize_room("!any:hs") is not None
        # ...but a positive membership match still authorizes.
        monkeypatch.setattr(m, "_joined_rooms", lambda: {"!known:hs"})
        assert m._authorize_room("!known:hs") is None

    # --- explicit opt-in escape hatch ---------------------------------------

    def test_allow_any_room_flag_permits_roomless_context(self, creds, monkeypatch):
        # The operator escape hatch: MATRIX_TOOLS_ALLOW_ANY_ROOM=true lets a
        # room-less cron/standalone run act on an arbitrary room.
        monkeypatch.delenv("MATRIX_TOOLS_ALLOW_ANY_ROOM", raising=False)
        monkeypatch.setattr(m, "_allow_any_room", lambda: True)
        monkeypatch.setattr(m, "_current_room", lambda: "")
        monkeypatch.setattr(m, "_allowed_room_ids", lambda: set())
        monkeypatch.setattr(m, "_joined_rooms", lambda: None)
        fake = _recorder({"leave": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!any:hs"})))
        assert out["success"] is True
        assert [c["room_id"] for c in fake.calls] == ["!any:hs"]

    @pytest.mark.parametrize("val,expected", [
        ("true", True), ("1", True), ("yes", True),
        ("", False), ("false", False), ("no", False),
    ])
    def test_allow_any_room_flag_truth_table(self, monkeypatch, val, expected):
        monkeypatch.setenv("MATRIX_TOOLS_ALLOW_ANY_ROOM", val)
        assert m._allow_any_room() is expected

    def test_allow_any_room_flag_unset(self, monkeypatch):
        monkeypatch.delenv("MATRIX_TOOLS_ALLOW_ANY_ROOM", raising=False)
        assert m._allow_any_room() is False

    def test_allow_any_room_flag_does_not_bypass_allowlist(self, creds, monkeypatch):
        # The flag only lifts the room-less denial; a non-empty allowlist
        # still acts as a strict whitelist.
        monkeypatch.setattr(m, "_allow_any_room", lambda: True)
        monkeypatch.setattr(m, "_current_room", lambda: "")
        monkeypatch.setattr(m, "_allowed_room_ids", lambda: {"!listed:hs"})
        monkeypatch.setattr(m, "_joined_rooms", lambda: None)
        assert m._authorize_room("!listed:hs") is None
        assert m._authorize_room("!other:hs") is not None

    def test_allowlist_set_rejects_unlisted_room(self, creds, monkeypatch):
        # An allowlist is a strict whitelist: a room not in it is rejected
        # even with no current room bound.
        monkeypatch.setattr(m, "_current_room", lambda: "")
        monkeypatch.setattr(m, "_allowed_room_ids", lambda: {"!listed:hs"})
        monkeypatch.setattr(m, "_joined_rooms", lambda: None)
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!other:hs"})))
        _assert_error(out, "MATRIX_ALLOWED_ROOMS")

    def test_current_room_from_session_contextvar(self, creds, monkeypatch):
        # _current_room must read the task-local HERMES_SESSION_CHAT_ID
        # ContextVar the gateway binds — not the process-global env.
        from gateway.session_context import set_session_vars

        tokens = set_session_vars(platform="matrix", chat_id="!bound:hs")
        try:
            assert m._current_room() == "!bound:hs"
            # And a cross-room id is rejected against it.
            monkeypatch.setattr(m, "_allowed_room_ids", lambda: set())
            monkeypatch.setattr(m, "_joined_rooms", lambda: None)
            fake = _recorder({"leave": (200, "{}")})
            monkeypatch.setattr(m, "_matrix_room_action", fake)
            out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!other:hs"})))
            _assert_error(out, "outside this turn's scope")
            assert fake.calls == []
        finally:
            from gateway.session_context import clear_session_vars

            clear_session_vars(tokens)


# --------------------------------------------------------------------------
# matrix_delete_room  (leave + forget)
# --------------------------------------------------------------------------
class TestDeleteRoom:
    def test_delete_leave_then_forget(self, creds, default_authz, monkeypatch):
        fake = _recorder({"leave": (200, "{}"), "forget": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_delete_room({"room_id": "!cur:hs"})))
        assert out["success"] is True
        assert out["action"] == "leave+forget"
        assert [c["action"] for c in fake.calls] == ["leave", "forget"]

    def test_delete_tolerates_already_left(self, creds, default_authz, monkeypatch):
        # leaving a room you're not in -> 403 M_FORBIDDEN; delete must still forget
        fake = _recorder({
            "leave": (403, '{"errcode":"M_FORBIDDEN","error":"not in room"}'),
            "forget": (200, "{}"),
        })
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_delete_room({"room_id": "!cur:hs"})))
        assert out["success"] is True
        assert [c["action"] for c in fake.calls] == ["leave", "forget"]

    def test_delete_leave_hard_error_skips_forget(self, creds, default_authz, monkeypatch):
        fake = _recorder({"leave": (500, "boom"), "forget": (200, "{}")})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_delete_room({"room_id": "!cur:hs"})))
        assert "Matrix leave (during delete) error (500)" in out["error"]
        assert [c["action"] for c in fake.calls] == ["leave"]  # forget NOT attempted

    def test_delete_forget_error(self, creds, default_authz, monkeypatch):
        fake = _recorder({"leave": (200, "{}"), "forget": (400, '{"errcode":"M_UNKNOWN"}')})
        monkeypatch.setattr(m, "_matrix_room_action", fake)
        out = _parse(_run(m._handle_matrix_delete_room({"room_id": "!cur:hs"})))
        assert "Matrix forget error (400)" in out["error"]

    def test_delete_missing_room_id(self, creds, default_authz):
        out = _parse(_run(m._handle_matrix_delete_room({})))
        assert "room_id is required" in out["error"]


# --------------------------------------------------------------------------
# Adapter cache reconciliation after a raw leave/forget
# --------------------------------------------------------------------------
class TestAdapterCacheReconciliation:
    def _seeded_adapter(self, room_id):
        adapter = _make_adapter()
        adapter._joined_rooms.add(room_id)
        adapter._room_identities[room_id] = object()
        adapter._room_identity_cached_at[room_id] = 1.0
        adapter._dm_rooms[room_id] = True
        return adapter

    def test_leave_evicts_all_caches_for_room(self, creds, default_authz, monkeypatch):
        # A raw leave must evict the room from the live adapter's membership
        # caches, otherwise _join_room_by_id later trusts the stale id and
        # skips the real join (incremental sync only ADDS ids).
        adapter = self._seeded_adapter("!cur:hs")
        monkeypatch.setattr(m, "_live_adapter", lambda: adapter)
        monkeypatch.setattr(m, "_matrix_room_action", _recorder({"leave": (200, "{}")}))
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!cur:hs"})))
        assert out["success"] is True
        assert "!cur:hs" not in adapter._joined_rooms
        assert "!cur:hs" not in adapter._room_identities
        assert "!cur:hs" not in adapter._room_identity_cached_at
        assert "!cur:hs" not in adapter._dm_rooms

    def test_leave_leaves_other_rooms_in_cache(self, creds, default_authz, monkeypatch):
        # Reconciliation must be surgical: a different, still-joined room is
        # NOT evicted.
        adapter = self._seeded_adapter("!other:hs")
        adapter._joined_rooms.add("!cur:hs")
        monkeypatch.setattr(m, "_joined_rooms", lambda: {"!cur:hs"})  # authorize
        monkeypatch.setattr(m, "_live_adapter", lambda: adapter)
        monkeypatch.setattr(m, "_matrix_room_action", _recorder({"leave": (200, "{}")}))
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!cur:hs"})))
        assert out["success"] is True
        assert adapter._joined_rooms == {"!other:hs"}
        assert "!other:hs" in adapter._room_identities

    def test_delete_evicts_after_leave_and_forget(self, creds, default_authz, monkeypatch):
        adapter = self._seeded_adapter("!cur:hs")
        monkeypatch.setattr(m, "_live_adapter", lambda: adapter)
        monkeypatch.setattr(
            m,
            "_matrix_room_action",
            _recorder({"leave": (200, "{}"), "forget": (200, "{}")}),
        )
        out = _parse(_run(m._handle_matrix_delete_room({"room_id": "!cur:hs"})))
        assert out["success"] is True
        assert "!cur:hs" not in adapter._joined_rooms
        assert "!cur:hs" not in adapter._dm_rooms
        assert "!cur:hs" not in adapter._room_identities

    def test_failed_leave_does_not_touch_cache(self, creds, default_authz, monkeypatch):
        adapter = self._seeded_adapter("!cur:hs")
        monkeypatch.setattr(m, "_live_adapter", lambda: adapter)
        monkeypatch.setattr(m, "_matrix_room_action", _recorder({"leave": (404, "gone")}))
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!cur:hs"})))
        _assert_error(out, "Matrix leave error (404)")
        assert adapter._joined_rooms == {"!cur:hs"}, "cache untouched on failed leave"
        assert adapter._room_identities == {"!cur:hs": adapter._room_identities["!cur:hs"]}

    def test_forget_error_still_reconciles(self, creds, default_authz, monkeypatch):
        # The leave succeeded, so the room is gone from membership even though
        # the subsequent forget failed. Reconciliation still runs after leave.
        adapter = self._seeded_adapter("!cur:hs")
        monkeypatch.setattr(m, "_live_adapter", lambda: adapter)
        monkeypatch.setattr(
            m,
            "_matrix_room_action",
            _recorder({"leave": (200, "{}"), "forget": (400, '{"errcode":"M_UNKNOWN"}')}),
        )
        out = _parse(_run(m._handle_matrix_delete_room({"room_id": "!cur:hs"})))
        _assert_error(out, "Matrix forget error (400)")
        assert "!cur:hs" not in adapter._joined_rooms, "leave reconciliation still applies"

    def test_no_live_adapter_is_noop(self, creds, default_authz, monkeypatch):
        # CLI / cron / a non-Matrix gateway: no live adapter, so reconciliation
        # must not raise and leave must still succeed.
        monkeypatch.setattr(m, "_live_adapter", lambda: None)
        monkeypatch.setattr(m, "_matrix_room_action", _recorder({"leave": (200, "{}")}))
        out = _parse(_run(m._handle_matrix_leave_room({"room_id": "!cur:hs"})))
        assert out["success"] is True

    def test_reconcile_cancels_pending_invite_join(self):
        adapter = _make_adapter()

        async def scenario():
            loop = asyncio.get_running_loop()

            async def noop():
                await asyncio.sleep(10)

            task = loop.create_task(noop())
            adapter._invite_join_tasks["!r:hs"] = task
            adapter.reconcile_left_room("!r:hs")
            await asyncio.sleep(0)  # let the cancellation land
            return task

        task = _run(scenario())
        assert adapter._invite_join_tasks == {}
        assert task.cancelled()

    def test_join_trusts_stale_cache_then_rejoin_after_reconcile(self):
        # Regression: _join_room_by_id trusts _joined_rooms, so a stale entry
        # short-circuits the real join. Reconciliation must clear the entry so
        # the next join re-joins.
        adapter = _make_adapter()
        client = MagicMock()
        client.join_room = AsyncMock()
        adapter._client = client
        adapter._joined_rooms.add("!stale:hs")

        assert _run(adapter._join_room_by_id("!stale:hs")) is True
        client.join_room.assert_not_called()  # the stale-cache trust gap

        adapter.reconcile_left_room("!stale:hs")
        assert _run(adapter._join_room_by_id("!stale:hs")) is True
        client.join_room.assert_awaited_once()  # real re-join happens


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
# registry wiring + toolset scoping (Matrix sessions only)
# --------------------------------------------------------------------------
class TestRegistration:
    TOOLS = ("matrix_create_room", "matrix_leave_room", "matrix_delete_room")

    def test_tools_registered(self):
        from tools.registry import registry
        for name in self.TOOLS:
            assert name in registry._tools, f"{name} not registered"
            assert registry._tools[name].toolset == "hermes-matrix"
            assert registry._tools[name].check_fn is m._check_matrix_room_admin

    def test_not_in_shared_core_tools(self):
        # Scoping contract: Matrix room-admin tools must NOT be offered in
        # non-Matrix sessions, so they must not be in the shared core list.
        from toolsets import _HERMES_CORE_TOOLS
        for name in self.TOOLS:
            assert name not in _HERMES_CORE_TOOLS, f"{name} leaked into _HERMES_CORE_TOOLS"

    def test_in_hermes_matrix_toolset(self):
        from toolsets import TOOLSETS
        for name in self.TOOLS:
            assert name in TOOLSETS["hermes-matrix"]["tools"]

    def test_not_in_other_platform_toolsets(self):
        from toolsets import TOOLSETS
        for name_ts, ts in TOOLSETS.items():
            if name_ts in {"hermes-matrix", "hermes-gateway"}:
                continue  # hermes-gateway is the union of all messaging platforms
            tools = ts.get("tools", [])
            for name in self.TOOLS:
                assert name not in tools, f"{name} found in non-Matrix toolset '{name_ts}'"

    def test_resolve_toolset_scoping(self):
        from toolsets import resolve_toolset
        matrix_tools = set(resolve_toolset("hermes-matrix"))
        for name in self.TOOLS:
            assert name in matrix_tools, f"{name} missing from resolved hermes-matrix"
        for ts in ("hermes-cli", "hermes-telegram", "hermes-discord", "hermes-slack"):
            resolved = set(resolve_toolset(ts))
            for name in self.TOOLS:
                assert name not in resolved, f"{name} leaked into resolved {ts}"
