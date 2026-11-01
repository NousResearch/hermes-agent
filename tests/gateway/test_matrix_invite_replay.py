"""Regression tests: historic m.room.member invite events replayed from a JOINED room's
state/timeline must not be treated as invites to act on.

mautrix's ``MembershipEventDispatcher`` fans out every ``m.room.member`` event whose
``membership`` is ``invite`` as ``InternalEventType.INVITE`` -- regardless of which sync
section carried it. The adapter connects with ``MemorySyncStore``, so every (re)connect is a
full-state initial sync that re-dispatches each joined room's state and recent timeline. Any
invite event still in that response -- the bot's own historic invite, or an invite addressed
to someone else -- re-fires ``_on_invite`` on every connect:
"rejecting invite ... from unauthorized user" when the inviter is not allow-listed,
"invited to ... joining" plus a no-op join task when it is.

The only invite the bot can act on is one the homeserver delivers in ``rooms.invite``
(``SyncStream.INVITED_ROOM``). The unit tests drive ``_on_invite`` with events stamped the way
``Client.dispatch_event`` stamps them (``event.source``). CI installs no mautrix (the matrix
extra is lazy-installed), so the ``sync_stream`` fixture patches a mirror of the flag into
``mautrix.client`` when the real one is not importable. ``TestRealDispatchPath`` runs the same
payload shapes through a real mautrix ``Client`` + ``MembershipEventDispatcher`` wherever the
real package is installed, and skips otherwise.
"""

import asyncio
import enum
import logging
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _real_mautrix_available() -> bool:
    try:
        import mautrix
    except ImportError:
        return False
    return isinstance(mautrix, types.ModuleType) and hasattr(mautrix, "__file__")


needs_real_mautrix = pytest.mark.skipif(
    not _real_mautrix_available(), reason="real mautrix package not installed"
)


class _SyncStreamMirror(enum.IntFlag):
    """The ``mautrix.client.SyncStream`` members these tests and the adapter read."""

    STATE = enum.auto()
    TIMELINE = enum.auto()
    JOINED_ROOM = enum.auto()
    INVITED_ROOM = enum.auto()


@pytest.fixture
def sync_stream(monkeypatch):
    """The flag class the adapter will import: the real ``SyncStream`` when mautrix is
    installed, otherwise the mirror, patched into (a stub of) ``mautrix.client``."""
    try:
        from mautrix.client import SyncStream

        return SyncStream
    except ImportError:
        pass
    client_mod = sys.modules.get("mautrix.client")
    if client_mod is None:
        root = sys.modules.get("mautrix") or types.ModuleType("mautrix")
        client_mod = types.ModuleType("mautrix.client")
        monkeypatch.setitem(sys.modules, "mautrix", root)
        monkeypatch.setitem(sys.modules, "mautrix.client", client_mod)
        monkeypatch.setattr(root, "client", client_mod, raising=False)
    monkeypatch.setattr(client_mod, "SyncStream", _SyncStreamMirror, raising=False)
    return _SyncStreamMirror


def _replay(ss):
    return ss.JOINED_ROOM | ss.TIMELINE


def _replay_state(ss):
    return ss.JOINED_ROOM | ss.STATE


def _live(ss):
    return ss.INVITED_ROOM | ss.STATE


BOT = "@hermes:example.org"
OWNER = "@owner:example.org"
STRANGER = "@stranger:example.org"
ROOM = "!room:example.org"


def _make_adapter(user_id=BOT):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="syt_test_token",
            extra={"homeserver": "https://matrix.example.org", "user_id": user_id},
        )
    )
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    adapter._startup_ts = time.time() - 10
    adapter._allowed_user_ids = {OWNER}
    adapter._join_room_by_id = AsyncMock(return_value=True)
    return adapter


def _invite(sender, state_key, source, room_id=ROOM):
    return SimpleNamespace(
        room_id=room_id,
        sender=sender,
        state_key=state_key,
        content=SimpleNamespace(is_direct=False, membership="invite"),
        source=source,
    )


async def _drain(adapter):
    for task in list(adapter._invite_join_tasks.values()):
        await task


def _records(caplog, level):
    return [r.getMessage() for r in caplog.records if r.levelno >= level]


class TestReplayedInvitesAreIgnored:
    @pytest.mark.asyncio
    async def test_own_historic_invite_replayed_from_joined_room_is_quiet(
        self, caplog, sync_stream
    ):
        """The bot's own old invite (unauthorized inviter) re-read on boot: no warning, no join."""
        adapter = _make_adapter()
        adapter._joined_rooms = {ROOM}
        with caplog.at_level(logging.DEBUG):
            await adapter._on_invite(_invite(STRANGER, BOT, _replay(sync_stream)))
        assert adapter._invite_join_tasks == {}
        adapter._join_room_by_id.assert_not_awaited()
        assert _records(caplog, logging.WARNING) == []

    @pytest.mark.asyncio
    async def test_third_party_invite_replayed_from_joined_room_is_quiet(
        self, caplog, sync_stream
    ):
        """Someone else's invite (authorized inviter) in the timeline: no 'joining' line, no task."""
        adapter = _make_adapter()
        adapter._joined_rooms = {ROOM}
        with caplog.at_level(logging.DEBUG):
            await adapter._on_invite(
                _invite(OWNER, "@third:example.org", _replay(sync_stream))
            )
        assert adapter._invite_join_tasks == {}
        adapter._join_room_by_id.assert_not_awaited()
        assert _records(caplog, logging.INFO) == []

    @pytest.mark.asyncio
    async def test_source_alone_is_decisive(self, caplog, sync_stream):
        """Not joined, addressed to the bot, authorized inviter -- only the sync-stream stamp says
        this is history. It must still be ignored: a joined-room timeline can never carry a
        live invite for us."""
        adapter = _make_adapter()
        with caplog.at_level(logging.DEBUG):
            await adapter._on_invite(_invite(OWNER, BOT, _replay(sync_stream)))
        assert adapter._invite_join_tasks == {}
        adapter._join_room_by_id.assert_not_awaited()
        assert _records(caplog, logging.INFO) == []

    @pytest.mark.asyncio
    async def test_state_section_replay_is_quiet_too(self, caplog, sync_stream):
        adapter = _make_adapter()
        adapter._joined_rooms = {ROOM}
        with caplog.at_level(logging.DEBUG):
            await adapter._on_invite(
                _invite(OWNER, "@third:example.org", _replay_state(sync_stream))
            )
        assert adapter._invite_join_tasks == {}
        assert _records(caplog, logging.INFO) == []


class TestLiveInvitesStillWork:
    @pytest.mark.asyncio
    async def test_live_invite_from_authorized_user_joins(self, sync_stream):
        adapter = _make_adapter()
        await adapter._on_invite(_invite(OWNER, BOT, _live(sync_stream)))
        await _drain(adapter)
        adapter._join_room_by_id.assert_awaited_once_with(ROOM)

    @pytest.mark.asyncio
    async def test_live_invite_from_unauthorized_user_is_still_rejected(
        self, caplog, sync_stream
    ):
        adapter = _make_adapter()
        with caplog.at_level(logging.DEBUG):
            await adapter._on_invite(_invite(STRANGER, BOT, _live(sync_stream)))
        adapter._join_room_by_id.assert_not_awaited()
        assert any("rejecting invite" in m for m in _records(caplog, logging.WARNING))


class TestNoSourceCompat:
    """Events without a ``source`` stamp (manual dispatch, older callers) keep the old path,
    except that a room we already sit in is never re-joined. The state_key cases live in
    ``test_matrix_invite_state_key.py``."""

    @pytest.mark.asyncio
    async def test_unstamped_invite_to_self_joins(self):
        adapter = _make_adapter()
        await adapter._on_invite(
            SimpleNamespace(
                room_id=ROOM, sender=OWNER, content=SimpleNamespace(is_direct=False)
            )
        )
        await _drain(adapter)
        adapter._join_room_by_id.assert_awaited_once_with(ROOM)

    @pytest.mark.asyncio
    async def test_unstamped_invite_for_joined_room_is_quiet(self, caplog):
        adapter = _make_adapter()
        adapter._joined_rooms = {ROOM}
        with caplog.at_level(logging.DEBUG):
            await adapter._on_invite(
                SimpleNamespace(
                    room_id=ROOM,
                    sender=STRANGER,
                    content=SimpleNamespace(is_direct=False),
                )
            )
        assert adapter._invite_join_tasks == {}
        assert _records(caplog, logging.WARNING) == []


def _member_invite(sender, target, event_id=None):
    event = {
        "type": "m.room.member",
        "sender": sender,
        "state_key": target,
        "content": {"membership": "invite"},
    }
    if event_id:  # timeline events carry ids; stripped invite_state events do not
        event.update(event_id=event_id, origin_server_ts=1_700_000_000_000)
    return event


def _joined_room_sync(*events):
    return {
        "next_batch": "s1",
        "rooms": {
            "join": {ROOM: {"timeline": {"events": list(events), "limited": True}}}
        },
    }


def _invited_room_sync(sender):
    return {
        "next_batch": "s2",
        "rooms": {
            "invite": {
                ROOM: {"invite_state": {"events": [_member_invite(sender, BOT)]}}
            }
        },
    }


@needs_real_mautrix
class TestRealDispatchPath:
    """The unit tests above assume ``Client.dispatch_event`` stamps ``event.source`` with the
    sync section. These drive a sync payload through the real mautrix ``Client`` and
    ``MembershipEventDispatcher`` into ``_on_invite``, registered the way ``connect()`` does."""

    @staticmethod
    async def _feed(adapter, payload):
        from mautrix.client import Client, InternalEventType
        from mautrix.client.dispatcher import MembershipEventDispatcher
        from mautrix.client.state_store import MemoryStateStore, MemorySyncStore
        from mautrix.types import UserID

        client = Client(
            mxid=UserID(BOT),
            api=MagicMock(),
            state_store=MemoryStateStore(),
            sync_store=MemorySyncStore(),
        )
        client.add_dispatcher(MembershipEventDispatcher)
        client.add_event_handler(
            InternalEventType.INVITE, adapter._on_invite, wait_sync=True
        )
        await asyncio.gather(*client.handle_sync(payload))
        # The dispatcher runs as a background handler and fans INVITE out without awaiting
        # it; wait for everything it spawned.
        for _ in range(10):
            pending = [
                t
                for t in asyncio.all_tasks()
                if t is not asyncio.current_task() and not t.done()
            ]
            if not pending:
                break
            await asyncio.wait(pending, timeout=1)
        await _drain(adapter)

    @pytest.mark.asyncio
    async def test_joined_room_timeline_invites_are_quiet(self, caplog):
        """Not marked joined, so only the source stamp can tell the bot's own old invite
        (unauthorized inviter) from a live one."""
        adapter = _make_adapter()
        payload = _joined_room_sync(
            _member_invite(STRANGER, BOT, "$own"),
            _member_invite(OWNER, "@third:example.org", "$third"),
        )
        with caplog.at_level(logging.DEBUG):
            await self._feed(adapter, payload)
        adapter._join_room_by_id.assert_not_awaited()
        assert [m for m in _records(caplog, logging.INFO) if "invite" in m] == []

    @pytest.mark.asyncio
    async def test_live_invite_from_unauthorized_user_is_rejected(self, caplog):
        adapter = _make_adapter()
        with caplog.at_level(logging.DEBUG):
            await self._feed(adapter, _invited_room_sync(STRANGER))
        adapter._join_room_by_id.assert_not_awaited()
        assert any("rejecting invite" in m for m in _records(caplog, logging.WARNING))

    @pytest.mark.asyncio
    async def test_live_invite_from_authorized_user_joins(self):
        adapter = _make_adapter()
        await self._feed(adapter, _invited_room_sync(OWNER))
        adapter._join_room_by_id.assert_awaited_once_with(ROOM)
