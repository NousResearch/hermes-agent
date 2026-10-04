"""Self-chat mode (``MATRIX_SELF_CHAT``): messages typed from the bot's own account become user
input — the WhatsApp bridge self-chat equivalent for Matrix. The suite pins the two
echo-suppression layers that keep the agent from replying to itself forever:

1. sends through THIS adapter are recorded at send time (``_remember_event``) and dropped when the
   sync replays them back;
2. sends made by OTHER processes (``hermes send``, cron standalone) carry a ``hermes_*``
   transaction ID echoed in ``unsigned.transaction_id`` and are dropped by prefix.

Plus the scoping rule: self-chat only triggers in DM-classified rooms or ``MATRIX_HOME_ROOM`` —
never in group rooms."""

import time

import pytest

from gateway.config import PlatformConfig

_SELF = "@bot:example.org"
_DM_ROOM = "!dm:example.org"
_GROUP_ROOM = "!group:example.org"


def _make_adapter(**extra):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    return MatrixAdapter(PlatformConfig(enabled=True, token="syt_test_token", extra={
        "homeserver": "https://matrix.example.org", "user_id": _SELF, **extra}))


class _Event:
    """Minimal stand-in for a nio room-message/reaction event."""

    def __init__(self, *, sender=_SELF, room_id=_DM_ROOM, event_id="$event", content=None,
                 unsigned=None):
        self.sender, self.room_id, self.event_id = sender, room_id, event_id
        self.content = content if content is not None else {"msgtype": "m.text", "body": "hello"}
        self.unsigned = unsigned or {}
        self.origin_server_ts = int(time.time() * 1000)


def _wire(adapter, handled, *, dm_rooms=({_DM_ROOM} | set())):
    """Patch the seams: record what reaches the text handler; control room classification."""

    async def record(*args):
        handled.append(args)

    async def is_allowed(room_id):
        return True

    async def is_dm(room_id):
        return room_id in dm_rooms

    adapter._handle_text_message = record
    adapter._is_allowed_matrix_room_event = is_allowed
    adapter._is_dm_room = is_dm
    return adapter


# --- Layer 1: send-time tracking (this adapter's own sends) ---------------------------------

def test_sent_events_are_tracked_and_dropped_as_echoes():
    adapter = _make_adapter()
    assert not adapter._is_duplicate_event("$sent")
    # Every send path (_send_room_message, _send_content_event, reactions) records its event ID.
    adapter._remember_event("$sent")
    assert adapter._is_duplicate_event("$sent")


def test_remember_event_is_idempotent_null_safe_and_bounded():
    adapter = _make_adapter()
    adapter._remember_event("$a")
    adapter._remember_event("$a")  # double-record stays one entry
    adapter._remember_event(None)
    adapter._remember_event("")  # null-safe
    assert adapter._is_duplicate_event("$a")

    cap = adapter._processed_events.maxlen
    assert cap is not None  # the tracker is a bounded deque by construction
    for i in range(cap + 50):
        adapter._remember_event(f"$fill-{i}")
    assert len(adapter._processed_events) == cap  # bounded deque
    assert len(adapter._processed_events_set) == cap  # evicted IDs leave the lookup set too


# --- Layer 2: hermes_* transaction-ID echoes (other processes' sends) -----------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("unsigned", [
    {"transaction_id": "hermes_1727000000000_deadbeef"},           # dict-shaped (raw event)
    type("U", (), {"transaction_id": "hermes_1727000000000_deadbeef"})(),  # attr-shaped (nio)
], ids=["dict", "attr"])
async def test_hermes_txn_echoes_are_dropped(unsigned):
    adapter = _make_adapter(**{"self_chat": "true"})
    handled = []
    _wire(adapter, handled)
    await adapter._on_room_message(_Event(unsigned=unsigned, event_id="$echo"))
    assert handled == []


@pytest.mark.asyncio
async def test_own_message_without_txn_is_processed_as_user_input():
    adapter = _make_adapter(**{"self_chat": "true"})
    handled = []
    _wire(adapter, handled)
    await adapter._on_room_message(_Event(unsigned={}, event_id="$typed"))
    assert len(handled) == 1


@pytest.mark.asyncio
async def test_self_chat_off_keeps_upstream_behavior(monkeypatch):
    monkeypatch.delenv("MATRIX_SELF_CHAT", raising=False)
    adapter = _make_adapter()  # no self_chat key, no env → own messages stay ignored
    handled = []
    _wire(adapter, handled)
    await adapter._on_room_message(_Event(unsigned={}, event_id="$ignored"))
    assert handled == []


# --- Scoping: DM-classified rooms or MATRIX_HOME_ROOM only, never groups ---------------------

@pytest.mark.asyncio
async def test_group_rooms_never_trigger_self_chat_unless_home_room(monkeypatch):
    monkeypatch.delenv("MATRIX_HOME_ROOM", raising=False)
    adapter = _make_adapter(**{"self_chat": "true"})
    handled = []
    _wire(adapter, handled)
    await adapter._on_room_message(_Event(room_id=_GROUP_ROOM, event_id="$group"))
    assert handled == []  # a group room is never self-chat

    monkeypatch.setenv("MATRIX_HOME_ROOM", _GROUP_ROOM)
    await adapter._on_room_message(_Event(room_id=_GROUP_ROOM, event_id="$home"))
    assert len(handled) == 1  # the configured home room always qualifies


@pytest.mark.asyncio
async def test_self_chat_room_scope_fails_closed(monkeypatch):
    monkeypatch.setenv("MATRIX_HOME_ROOM", "!home:example.org")
    adapter = _make_adapter()

    async def is_dm(room_id):
        return room_id == _DM_ROOM

    adapter._is_dm_room = is_dm
    assert await adapter._is_self_chat_room("!home:example.org")  # home room, not DM
    assert await adapter._is_self_chat_room(_DM_ROOM)  # DM classification
    assert not await adapter._is_self_chat_room(_GROUP_ROOM)

    async def boom(room_id):
        raise RuntimeError("classification unavailable")

    adapter._is_dm_room = boom
    assert not await adapter._is_self_chat_room(_GROUP_ROOM)  # errors fail closed


# --- Reactions (approvals / pickers) use the same two layers --------------------------------

@pytest.mark.asyncio
async def test_reaction_echoes_drop_but_own_reactions_pass():
    adapter = _make_adapter(**{"self_chat": "true"})
    seen = []

    async def record(room_id, reacts_to, key, sender):
        seen.append((reacts_to, key))
        return True

    adapter._handle_approval_reaction = record
    content = {"m.relates_to": {"rel_type": "m.annotation", "event_id": "$prompt", "key": "✅"}}

    await adapter._on_reaction(_Event(content=content, unsigned={
        "transaction_id": "hermes_1727000000000_deadbeef"}, event_id="$rx-echo"))
    assert seen == []  # hermes_* echo from another process

    await adapter._on_reaction(_Event(content=content, unsigned={}, event_id="$rx-typed"))
    assert seen == [("$prompt", "✅")]  # the user's own reaction reaches the picker
