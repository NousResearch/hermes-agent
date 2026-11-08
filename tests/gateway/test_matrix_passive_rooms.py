"""Passive Matrix rooms: mention-gate drops persist as ambient context, never silent.

A room listed in ``MATRIX_PASSIVE_ROOMS`` (or ``matrix.passive_rooms`` in
config.yaml) records the messages the mention gate would drop as ``observed``
user rows in the room's shared session: no turn, no read receipt. A @mention
in such a room still dispatches a full turn, but on the room-scoped shared
source (user-less) plus the observe channel prompt, whose lowercase marker is
what the runner uses to withhold observed rows from replayable history.
Commands keep the sender's source; a room that is also free-response is
free-response (sender source, no ambient rows ever); DMs are never ambient.
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.session import build_session_key

PASSIVE_ROOM = "!room1:example.org"
SENDER = "@alice:example.org"
BOT = "@hermes:example.org"


@pytest.fixture(autouse=True)
def _clean_matrix_env(monkeypatch):
    """Ambient parent-shell values must not flip the adapter's defaults."""
    for name in (
        "MATRIX_REQUIRE_MENTION", "MATRIX_PASSIVE_ROOMS", "MATRIX_FREE_RESPONSE_ROOMS",
        "MATRIX_ALLOWED_ROOMS", "MATRIX_PROCESS_NOTICES", "MATRIX_AUTO_THREAD",
        "MATRIX_MAX_MEDIA_BYTES", "MATRIX_ALLOWED_USERS", "MATRIX_GROUP_ALLOWED_CHATS",
        "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(name, raising=False)


def _make_adapter(extra=None):
    """Create a MatrixAdapter with mocked config (mirrors test_matrix_mention)."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": BOT,
            **(extra or {}),
        },
    )
    adapter = MatrixAdapter(config)
    adapter._text_batch_delay_seconds = 0  # disable batching for tests
    adapter.handle_message = AsyncMock()
    adapter._startup_ts = time.time() - 10  # avoid startup grace filter
    return adapter


def _group_identity(room_id, chat_type="group"):
    from plugins.platforms.matrix.adapter import MatrixRoomIdentity

    return MatrixRoomIdentity(
        room_id=room_id, room_name="Team Room", room_topic=None, canonical_alias=None,
        server_name="example.org", joined_member_count=None, is_direct_account_data=False,
        display_name="Team Room", has_explicit_name=True, chat_type=chat_type,
    )


def _make_event(body, sender=SENDER, event_id="$evt1", room_id=PASSIVE_ROOM,
                msgtype="m.text", url=None, info=None, mention_user_ids=None):
    """Fake mautrix room event; the adapter reads room_id/sender/event_id/timestamp/content."""
    content = {"body": body, "msgtype": msgtype}
    if url:
        content["url"] = url
    if info:
        content["info"] = info
    if mention_user_ids is not None:
        content["m.mentions"] = {"user_ids": mention_user_ids}
    return SimpleNamespace(
        sender=sender,
        event_id=event_id,
        room_id=room_id,
        timestamp=int(time.time() * 1000),
        content=content,
    )


class _FakeSessionStore:
    """Records shared-session lookups and ambient transcript appends."""

    def __init__(self):
        self.sources = []
        self.entries = []
        self._session_ids = {}

    def get_or_create_session(self, source):
        self.sources.append(source)
        key = (source.platform.value, source.chat_id, source.chat_type,
               source.user_id, source.thread_id)
        if key not in self._session_ids:
            self._session_ids[key] = f"matrix-room-session-{len(self._session_ids)}"
        return SimpleNamespace(session_id=self._session_ids[key])

    def append_to_transcript(self, session_id, entry, skip_db=False):
        self.entries.append((session_id, entry))


def _passive_adapter(monkeypatch, extra=None, *, room_id=PASSIVE_ROOM,
                     free_room=False, allowed_rooms=None):
    """Matrix adapter pointed at a passive room, with homeserver-facing readers mocked."""
    monkeypatch.setenv("MATRIX_PASSIVE_ROOMS", room_id)
    if free_room:
        monkeypatch.setenv("MATRIX_FREE_RESPONSE_ROOMS", room_id)
    if allowed_rooms is not None:
        monkeypatch.setenv("MATRIX_ALLOWED_ROOMS", allowed_rooms)
    adapter = _make_adapter(extra)
    adapter._allowed_user_ids = {SENDER}  # keep the sender admitted via the allowlist snapshot
    adapter._resolve_room_identity = AsyncMock(return_value=_group_identity(room_id))
    adapter._get_display_name = AsyncMock(return_value="Alice")
    adapter._download_and_cache_media = AsyncMock(return_value="/tmp/observed-media/cached")
    adapter._background_read_receipt = MagicMock()
    store = _FakeSessionStore()
    adapter.set_session_store(store)
    return adapter, store


def _single_entry(store):
    assert len(store.entries) == 1
    return store.entries[0][1]


# ---------------------------------------------------------------------------
# Ambient rows for dropped messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_mention_text_in_passive_room_persists_ambient_row_not_turn(monkeypatch):
    adapter, store = _passive_adapter(monkeypatch)

    await adapter._on_room_message(_make_event("hello"))

    adapter.handle_message.assert_not_awaited()
    adapter._background_read_receipt.assert_not_called()
    session_id, entry = store.entries[0]
    assert entry["role"] == "user"
    assert entry["observed"] is True
    assert entry["message_id"] == "$evt1"
    assert entry["content"] == "[Alice|@alice:example.org]\nhello"
    assert isinstance(entry["timestamp"], str) and entry["timestamp"]
    assert session_id is not None
    source = store.sources[0]
    assert source.chat_id == PASSIVE_ROOM
    assert source.chat_type == "group"
    assert source.user_id is None
    assert source.user_name is None
    assert source.thread_id is None
    assert source.parent_chat_id is None
    assert source.message_id is None


@pytest.mark.asyncio
async def test_non_mention_image_in_passive_room_caches_media_and_notes_path(monkeypatch):
    adapter, store = _passive_adapter(monkeypatch)
    adapter._download_and_cache_media = AsyncMock(return_value="/tmp/observed-media/photo.jpg")
    event = _make_event("photo.jpg", event_id="$img1", msgtype="m.image",
                        url="mxc://example.org/photo",
                        info={"mimetype": "image/jpeg", "size": 12345})

    await adapter._on_room_message(event)

    adapter.handle_message.assert_not_awaited()
    adapter._download_and_cache_media.assert_awaited_once()
    args = adapter._download_and_cache_media.await_args.args
    assert args[0] == "mxc://example.org/photo"
    assert args[1] == "$img1"
    entry = _single_entry(store)
    assert entry["content"].startswith("[Alice|@alice:example.org]\nphoto.jpg")
    assert "[Matrix] Cached observed room m.image 'photo.jpg' at: /tmp/observed-media/photo.jpg" in entry["content"]


@pytest.mark.asyncio
async def test_oversized_media_in_passive_room_is_not_downloaded(monkeypatch):
    adapter, store = _passive_adapter(monkeypatch)
    event = _make_event("big.mp4", event_id="$big1", msgtype="m.video",
                        url="mxc://example.org/big",
                        info={"mimetype": "video/mp4", "size": 200 * 1024 * 1024})

    await adapter._on_room_message(event)

    adapter.handle_message.assert_not_awaited()
    adapter._download_and_cache_media.assert_not_awaited()
    entry = _single_entry(store)
    assert "[Observed Matrix attachment too large or unverifiable. Maximum: 100 MB.]" in entry["content"]


@pytest.mark.asyncio
async def test_non_mxc_media_url_in_passive_room_is_not_downloaded(monkeypatch):
    adapter, store = _passive_adapter(monkeypatch)
    event = _make_event("photo.jpg", event_id="$ext1", msgtype="m.image",
                        url="https://example.org/photo.jpg",
                        info={"mimetype": "image/jpeg"})

    await adapter._on_room_message(event)

    adapter.handle_message.assert_not_awaited()
    adapter._download_and_cache_media.assert_not_awaited()
    entry = _single_entry(store)
    assert "[Matrix] Observed room m.image could not be cached: no verifiable MXC media URL.]" in entry["content"]


@pytest.mark.asyncio
async def test_media_download_failure_notes_row_and_still_no_turn(monkeypatch):
    adapter, store = _passive_adapter(monkeypatch)
    adapter._download_and_cache_media = AsyncMock(side_effect=RuntimeError("boom"))

    await adapter._on_room_message(_make_event("photo.jpg", msgtype="m.image",
                                               url="mxc://example.org/photo",
                                               info={"mimetype": "image/jpeg"}))

    adapter.handle_message.assert_not_awaited()
    entry = _single_entry(store)
    assert "[Matrix] Observed room m.image could not be downloaded (RuntimeError).]" in entry["content"]


@pytest.mark.asyncio
async def test_media_download_returning_none_notes_row(monkeypatch):
    adapter, store = _passive_adapter(monkeypatch)
    adapter._download_and_cache_media = AsyncMock(return_value=None)

    await adapter._on_room_message(_make_event("photo.jpg", msgtype="m.image",
                                               url="mxc://example.org/photo",
                                               info={"mimetype": "image/jpeg"}))

    adapter.handle_message.assert_not_awaited()
    entry = _single_entry(store)
    assert "[Matrix] Observed room m.image could not be read, not cached.]" in entry["content"]


# ---------------------------------------------------------------------------
# Ambient authorization: ambient rows respect gateway user admission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unauthorized_sender_in_passive_room_leaves_no_ambient_row_or_session(monkeypatch):
    """A sender the gateway would not admit stays invisible: no row, not even a session lookup."""
    adapter, store = _passive_adapter(monkeypatch)
    adapter._allowed_user_ids = set()

    await adapter._on_room_message(_make_event("hello from a stranger"))

    adapter.handle_message.assert_not_awaited()
    assert store.sources == []
    assert store.entries == []


@pytest.mark.asyncio
async def test_unauthorized_sender_media_in_passive_room_is_neither_downloaded_nor_ambient(monkeypatch):
    """The admission guard fires before any media is pulled onto the host."""
    adapter, store = _passive_adapter(monkeypatch)
    adapter._allowed_user_ids = set()
    event = _make_event("photo.jpg", event_id="$img2", msgtype="m.image",
                         url="mxc://example.org/photo",
                         info={"mimetype": "image/jpeg", "size": 12345})

    await adapter._on_room_message(event)

    adapter.handle_message.assert_not_awaited()
    adapter._download_and_cache_media.assert_not_awaited()
    assert store.sources == []
    assert store.entries == []


@pytest.mark.asyncio
async def test_ambient_row_requires_live_runner_authorization_when_wired(monkeypatch):
    """With a live runner check wired, only a literal True admits the ambient row — a False
    is not rescued by the adapter's allowlist snapshot."""
    adapter, store = _passive_adapter(monkeypatch)
    adapter.set_authorization_check(MagicMock(side_effect=[True, False]))

    await adapter._on_room_message(_make_event("first", event_id="$auth1"))
    await adapter._on_room_message(_make_event("second", event_id="$auth2"))

    adapter.handle_message.assert_not_awaited()
    adapter._authorization_check.assert_any_call(SENDER, None, PASSIVE_ROOM)
    entry = _single_entry(store)
    assert entry["message_id"] == "$auth1"


@pytest.mark.asyncio
async def test_ambient_row_allowed_via_gateway_allow_all_users_snapshot(monkeypatch):
    """Unwired: the live read of the GATEWAY_ALLOW_ALL_USERS snapshot admits the sender."""
    adapter, store = _passive_adapter(monkeypatch)
    adapter._allowed_user_ids = set()
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")

    await adapter._on_room_message(_make_event("hello"))

    adapter.handle_message.assert_not_awaited()
    entry = _single_entry(store)
    assert entry["message_id"] == "$evt1"


# ---------------------------------------------------------------------------
# Mentions: full turn on the room-scoped shared source
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mention_in_passive_room_dispatches_turn_with_shared_source(monkeypatch):
    adapter, store = _passive_adapter(monkeypatch)
    # First an unmentioned message (ambient row), then the mention.
    await adapter._on_room_message(_make_event("side chatter", event_id="$ambient1"))
    await adapter._on_room_message(_make_event(f"{BOT} what did alice say?", event_id="$mention1"))

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    # The event's source is the room-scoped shared source: user scoping cleared.
    assert event.source.chat_id == PASSIVE_ROOM
    assert event.source.chat_type == "group"
    assert event.source.user_id is None
    assert event.source.user_name is None
    assert event.source.thread_id is None
    assert event.source.parent_chat_id is None
    assert event.source.message_id is None
    assert "observed group context" in event.channel_prompt
    # Session-key parity: the ambient row's shared source and the mention turn's
    # swapped source must land in the same session.
    assert build_session_key(store.sources[0]) == build_session_key(event.source)
    assert event.text == "what did alice say?"


@pytest.mark.asyncio
async def test_room_in_both_passive_and_free_lists_wins_with_full_sender_turn(monkeypatch):
    adapter, store = _passive_adapter(monkeypatch, free_room=True)

    await adapter._on_room_message(_make_event("hello"))

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.user_id == SENDER
    assert event.source.user_name == "Alice"
    assert event.source.message_id == "$evt1"
    assert "observed group context" not in (event.channel_prompt or "")
    assert store.entries == []


@pytest.mark.asyncio
async def test_slash_command_in_passive_room_keeps_sender_source(monkeypatch):
    adapter, store = _passive_adapter(monkeypatch)

    await adapter._on_room_message(_make_event("/status"))

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.user_id == SENDER
    assert event.source.user_name == "Alice"
    assert event.source.message_id == "$evt1"
    assert "observed group context" not in (event.channel_prompt or "")
    assert store.entries == []


# ---------------------------------------------------------------------------
# Free-room precedence: dual-listed rooms never ambient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notice_in_room_in_both_lists_is_neither_ambient_nor_turn(monkeypatch):
    """Free-response wins: a dual-listed room leaves NO ambient rows, even for m.notice."""
    adapter, store = _passive_adapter(monkeypatch, free_room=True)

    await adapter._on_room_message(_make_event("cron result: ok", event_id="$bothnotice", msgtype="m.notice"))

    adapter.handle_message.assert_not_awaited()
    assert store.entries == []


@pytest.mark.asyncio
async def test_non_mxc_media_in_room_in_both_lists_leaves_no_ambient_row(monkeypatch):
    """Free-response wins: rejected media in a dual-listed room leaves no ambient note row."""
    adapter, store = _passive_adapter(monkeypatch, free_room=True)
    event = _make_event("photo.jpg", event_id="$bothimg", msgtype="m.image",
                        url="https://example.org/photo.jpg",
                        info={"mimetype": "image/jpeg"})

    await adapter._on_room_message(event)

    adapter.handle_message.assert_not_awaited()
    adapter._download_and_cache_media.assert_not_awaited()
    assert store.entries == []


# ---------------------------------------------------------------------------
# Attribution is mention-only: unmentioned turns keep the sender's source
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_require_mention_false_unmentioned_turn_keeps_sender_source_no_observe_prompt(monkeypatch):
    """A full turn under MATRIX_REQUIRE_MENTION=false is never re-attributed to the shared source."""
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    adapter, store = _passive_adapter(monkeypatch)

    await adapter._on_room_message(_make_event("hello"))

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.user_id == SENDER
    assert event.source.user_name == "Alice"
    assert event.source.message_id == "$evt1"
    assert "observed group context" not in (event.channel_prompt or "")
    assert store.entries == []


@pytest.mark.asyncio
async def test_bot_thread_followup_turn_keeps_sender_source_no_observe_prompt(monkeypatch):
    """A follow-up inside a bot thread bypasses the mention gate but never gets the observe swap."""
    adapter, store = _passive_adapter(monkeypatch)
    adapter._threads.mark("$root1")  # the bot already participated in this thread
    event = _make_event("follow-up in the thread", event_id="$follow1")
    event.content["m.relates_to"] = {"rel_type": "m.thread", "event_id": "$root1"}

    await adapter._on_room_message(event)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.user_id == SENDER
    assert event.source.thread_id == "$root1"
    assert "observed group context" not in (event.channel_prompt or "")
    assert store.entries == []


# ---------------------------------------------------------------------------
# Gating: allowlist, notices, DMs, voice
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_allowlisted_passive_room_is_neither_ambient_nor_turn(monkeypatch):
    adapter, store = _passive_adapter(monkeypatch, allowed_rooms="!other:example.org")

    await adapter._on_room_message(_make_event("hello"))

    adapter.handle_message.assert_not_awaited()
    assert store.entries == []


@pytest.mark.asyncio
async def test_unmentioned_notice_in_passive_room_persists_ambient_row_not_turn(monkeypatch):
    adapter, store = _passive_adapter(monkeypatch)

    await adapter._on_room_message(_make_event("cron result: ok", event_id="$notice1", msgtype="m.notice"))

    adapter.handle_message.assert_not_awaited()
    adapter._background_read_receipt.assert_not_called()
    entry = _single_entry(store)
    assert entry["content"] == "[Alice|@alice:example.org]\ncron result: ok"
    assert entry["observed"] is True


@pytest.mark.asyncio
async def test_mentioned_notice_in_passive_room_is_dropped_without_row_or_turn(monkeypatch):
    adapter, store = _passive_adapter(monkeypatch)
    event = _make_event(f"{BOT} cron result", event_id="$notice2", msgtype="m.notice")

    await adapter._on_room_message(event)

    adapter.handle_message.assert_not_awaited()
    assert store.entries == []


@pytest.mark.asyncio
async def test_notice_in_non_passive_room_is_never_ambient(monkeypatch):
    monkeypatch.delenv("MATRIX_PASSIVE_ROOMS", raising=False)
    adapter = _make_adapter()
    adapter._resolve_room_identity = AsyncMock(return_value=_group_identity(PASSIVE_ROOM))
    adapter._get_display_name = AsyncMock(return_value="Alice")
    store = _FakeSessionStore()
    adapter.set_session_store(store)

    await adapter._on_room_message(_make_event("cron result: ok", event_id="$notice3", msgtype="m.notice"))

    adapter.handle_message.assert_not_awaited()
    assert store.entries == []


@pytest.mark.asyncio
async def test_dm_listed_in_passive_rooms_is_never_ambient(monkeypatch):
    adapter, store = _passive_adapter(monkeypatch)
    # The room is a DM: the observer resolves identity and bails on chat_type == "dm".
    adapter._resolve_room_identity = AsyncMock(return_value=_group_identity(PASSIVE_ROOM, chat_type="dm"))

    await adapter._on_room_message(_make_event("hello dm"))

    # DMs are never mention-gated: normal full DM turn, sender source, no ambient row.
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_type == "dm"
    assert event.source.user_id == SENDER
    assert "observed group context" not in (event.channel_prompt or "")
    assert store.entries == []


@pytest.mark.asyncio
async def test_observer_bails_on_dm_identity(monkeypatch):
    adapter, store = _passive_adapter(monkeypatch)
    adapter._resolve_room_identity = AsyncMock(return_value=_group_identity(PASSIVE_ROOM, chat_type="dm"))

    await adapter._observe_passive_room_message(
        PASSIVE_ROOM, SENDER, "$dmevt1", "hello dm", {"msgtype": "m.text", "body": "hello dm"})

    assert store.entries == []
    assert store.sources == []


@pytest.mark.asyncio
async def test_parkable_voice_in_passive_room_parks_without_ambient_row(monkeypatch):
    """A parked voice is claimed by the sender's bare @mention: it must park, not leave a row."""
    adapter, store = _passive_adapter(monkeypatch)
    voice = _make_event("voice message", event_id="$voice1", msgtype="m.audio",
                        url="mxc://example.org/v", info={"mimetype": "audio/ogg"})
    voice.content["org.matrix.msc3245.voice"] = {}
    voice.content["m.mentions"] = {}

    await adapter._on_room_message(voice)

    adapter.handle_message.assert_not_awaited()
    adapter._download_and_cache_media.assert_not_awaited()  # parked voice is never downloaded
    assert store.entries == []
    assert adapter._parked_voices.pending(PASSIVE_ROOM, SENDER) is True


# ---------------------------------------------------------------------------
# Runner admission: user-less group source admitted only via group allowlist
# ---------------------------------------------------------------------------


def test_matrix_group_source_admitted_only_via_matrix_group_allowed_chats(monkeypatch):
    from gateway.authz_mixin import _group_chat_env
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    assert _group_chat_env(Platform.MATRIX) == "MATRIX_GROUP_ALLOWED_CHATS"

    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.MATRIX, chat_id=PASSIVE_ROOM, chat_type="group",
        user_id=None, user_name=None,
    )

    monkeypatch.setenv("MATRIX_GROUP_ALLOWED_CHATS", PASSIVE_ROOM)
    assert runner._is_user_authorized(source) is True

    other = SessionSource(
        platform=Platform.MATRIX, chat_id="!other:example.org", chat_type="group",
        user_id=None, user_name=None,
    )
    assert runner._is_user_authorized(other) is False


# ---------------------------------------------------------------------------
# Runner replay: observed rows are withheld from the replayable history
# ---------------------------------------------------------------------------


def test_observed_matrix_rows_are_withheld_from_replayable_history():
    from gateway.run import _build_gateway_agent_history

    observed_row = {
        "role": "user", "content": "[Alice|@alice:example.org]\nside chatter",
        "observed": True, "message_id": "$ambient1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    }
    history, observed = _build_gateway_agent_history(
        [observed_row,
         {"role": "assistant", "content": "hi"},
         {"role": "user", "content": "what did alice say?"}],
        channel_prompt="observed group context (Matrix) may be provided in a context-only block",
    )

    user_contents = [m.get("content") for m in history if m.get("role") == "user"]
    assert all("side chatter" not in content for content in user_contents)
    assert "what did alice say?" in user_contents
    assert observed == "[Alice|@alice:example.org]\nside chatter"
