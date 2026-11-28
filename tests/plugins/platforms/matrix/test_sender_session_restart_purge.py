"""Sender-session persistence across restarts (issue #148097).

Mautrix's OlmMachine discards a room's outbound Megolm session on any peer device-list change
(crypto/device_lists.py) and on what it *thinks* is a membership change — but the adapter starts
with an empty MemoryStateStore, so the startup full-state sync replays the room's current
m.room.member events as None -> join and the wipe fires on EVERY gateway restart. Peers then see
"Unable to decrypt" for the first post-restart replies until key forwarding recovers.
Two defences: MATRIX_DISABLE_DEVICE_CHANGE_KEY_ROTATION (bot opt-out, mirrors mautrix's own
bridge config) and a membership-replay gate that swallows member events only while the initial
sync dispatches.
"""

import asyncio

import pytest

from gateway.config import PlatformConfig


class _FakeClient:
    """Minimal add/remove_event_handler surface, keyed by event type."""

    def __init__(self):
        self.handlers = {}

    def add_event_handler(self, event_type, handler, wait_sync=False):
        self.handlers.setdefault(event_type, []).append(handler)

    def remove_event_handler(self, event_type, handler):
        handlers = self.handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)


class _FakeOlm:
    def __init__(self):
        self.disable_device_change_key_rotation = False
        self.seen = []

    async def handle_member_event(self, evt):
        self.seen.append(evt)


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    return MatrixAdapter(PlatformConfig(enabled=True, token="syt_test_token", extra={
        "homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}))


def test_membership_replay_gate_skips_only_initial_sync_dispatch(monkeypatch):
    # Stub mautrix.types.EventType so the test runs regardless of whether the matrix extra
    # (and its libolm build) is installed in the test environment.
    import sys
    import types as _types
    import enum

    class _EventType(enum.Enum):
        ROOM_MEMBER = "m.room.member"

    mautrix_mod = _types.ModuleType("mautrix")
    types_mod = _types.ModuleType("mautrix.types")
    setattr(types_mod, "EventType", _EventType)  # noqa: B010
    setattr(mautrix_mod, "types", types_mod)  # noqa: B010
    monkeypatch.setitem(sys.modules, "mautrix", mautrix_mod)
    monkeypatch.setitem(sys.modules, "mautrix.types", types_mod)

    client, olm = _FakeClient(), _FakeOlm()
    # machine.py registers exactly this: the OlmMachine's bound member handler.
    client.add_event_handler(_EventType.ROOM_MEMBER, olm.handle_member_event)

    adapter = _make_adapter()
    adapter._install_membership_replay_gate(client, olm)

    registered = client.handlers[_EventType.ROOM_MEMBER]
    assert len(registered) == 1 and registered[0] is not olm.handle_member_event
    gate = registered[0]

    # Initial sync: the replayed member event must NOT reach the machine (no session wipe).
    adapter._skip_membership_rotation = True
    asyncio.run(gate({"event_id": "$replayed"}))
    assert olm.seen == []

    # Delta sync: live membership changes still reach the machine (rotation stays meaningful).
    adapter._skip_membership_rotation = False
    asyncio.run(gate({"event_id": "$live"}))
    assert [e["event_id"] for e in olm.seen] == ["$live"]


def test_rotation_flag_defaults_to_mautrix_semantics(monkeypatch):
    monkeypatch.delenv("MATRIX_DISABLE_DEVICE_CHANGE_KEY_ROTATION", raising=False)
    olm = _FakeOlm()
    _make_adapter()._apply_bot_rotation_policy(olm)
    assert olm.disable_device_change_key_rotation is False


@pytest.mark.parametrize("value", ["true", "True", "1", "yes"])
def test_env_flag_on_disables_device_change_rotation(monkeypatch, value):
    monkeypatch.setenv("MATRIX_DISABLE_DEVICE_CHANGE_KEY_ROTATION", value)
    olm = _FakeOlm()
    _make_adapter()._apply_bot_rotation_policy(olm)
    assert olm.disable_device_change_key_rotation is True


@pytest.mark.parametrize("value", ["", "false", "0"])
def test_env_flag_off_keeps_rotation(monkeypatch, value):
    monkeypatch.setenv("MATRIX_DISABLE_DEVICE_CHANGE_KEY_ROTATION", value)
    olm = _FakeOlm()
    _make_adapter()._apply_bot_rotation_policy(olm)
    assert olm.disable_device_change_key_rotation is False


def test_config_key_disables_device_change_rotation(monkeypatch):
    monkeypatch.delenv("MATRIX_DISABLE_DEVICE_CHANGE_KEY_ROTATION", raising=False)
    from gateway.config import PlatformConfig
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="syt_test_token", extra={
        "homeserver": "https://matrix.example.org", "user_id": "@bot:example.org",
        "disable_device_change_key_rotation": True}))
    olm = _FakeOlm()
    adapter._apply_bot_rotation_policy(olm)
    assert olm.disable_device_change_key_rotation is True


def test_absorb_sync_scopes_skip_flag_to_initial_dispatch():
    adapter = _make_adapter()
    adapter._skip_membership_rotation = False
    seen = {}

    async def _fake_dispatch(sync_data):
        seen["during"] = adapter._skip_membership_rotation

    async def _fake_refresh():
        pass

    class _SyncStore:
        async def put_next_batch(self, nb):
            pass

    class _Client:
        sync_store = _SyncStore()

    adapter._dispatch_sync = _fake_dispatch
    adapter._refresh_dm_cache = _fake_refresh

    asyncio.run(adapter._absorb_sync(_Client(), {"next_batch": "s2"}, initial=True))
    assert seen["during"] is True
    assert adapter._skip_membership_rotation is False  # always cleared afterwards

    asyncio.run(adapter._absorb_sync(_Client(), {}, initial=False))
    assert seen["during"] is False  # live syncs keep the machine's real behaviour
