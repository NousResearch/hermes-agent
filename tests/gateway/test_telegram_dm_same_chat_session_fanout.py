"""Same Telegram private DM must not fan out into N concurrent sessions.

Live: one chat, same text, several inbound resolutions with different
``thread_id`` / reply-to suffixes produced distinct keys of the form
``agent:main:telegram:dm:<chat_id>:<extra>``, then N parallel turns and
N simultaneous final sends (issue #107133).

Forum topics and explicit ``force_new`` / suspend stay isolated.
"""

from datetime import datetime, timedelta

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionSource, SessionStore


CHAT_ID = "8631530276"
USER_ID = "8631530276"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Isolated SessionStore (pin DEFAULT_DB_PATH so tests never touch ~/.hermes)."""
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    return SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())


def _dm(*, chat_id=CHAT_ID, thread_id=None, message_id=None, user_id=USER_ID, profile=None):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id=user_id,
        thread_id=thread_id,
        message_id=message_id,
        profile=profile,
    )


def _forum(*, chat_id="-1002285219667", thread_id="17585", user_id="alice"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="forum",
        user_id=user_id,
        thread_id=thread_id,
    )


class TestSameChatTelegramDmDoesNotFanOut:
    def test_synthetic_thread_ids_resolve_to_one_session(self, store):
        """POSITIVE: reply-to / per-message thread suffixes must share one key and session_id."""
        first_src = _dm(thread_id="1001", message_id="2001")
        second_src = _dm(thread_id="1002", message_id="2002")

        first_key = store._generate_session_key(first_src)
        second_key = store._generate_session_key(second_src)
        assert first_key == second_key
        assert first_key == f"agent:main:telegram:dm:{CHAT_ID}"
        assert first_key.count(":") == 4

        first = store.get_or_create_session(first_src)
        second = store.get_or_create_session(second_src)
        assert first.session_key == second.session_key == first_key
        assert first.session_id == second.session_id

    def test_inflight_turn_does_not_mint_sibling_session(self, store):
        """IN-FLIGHT: an active turn on this chat must not grow a parallel session_id."""
        first = store.get_or_create_session(_dm(thread_id="1001", message_id="2001"))
        token = store.mark_turn_active(first.session_key)
        assert token

        sibling = store.get_or_create_session(_dm(thread_id="1002", message_id="2002"))
        assert sibling.session_key == first.session_key
        assert sibling.session_id == first.session_id

    def test_inflight_suffix_sibling_is_adopted_not_duplicated(self, store):
        """A live same-chat suffix key with an active turn must not mint a parallel id."""
        leftover_key = f"agent:main:telegram:dm:{CHAT_ID}:40404"
        now = datetime.now()
        leftover = SessionEntry(
            session_key=leftover_key,
            session_id="sess-leftover-fanout",
            created_at=now - timedelta(minutes=2),
            updated_at=now - timedelta(minutes=1),
            platform=Platform.TELEGRAM,
            chat_type="dm",
            origin=_dm(thread_id="40404", message_id="50505"),
            active_turn_token="turn-in-flight",
            active_turn_started_at=now,
        )
        store._ensure_loaded()
        with store._lock:
            store._entries[leftover_key] = leftover

        adopted = store.get_or_create_session(_dm(thread_id="60606", message_id="70707"))
        assert adopted.session_id == leftover.session_id
        assert adopted.session_key == f"agent:main:telegram:dm:{CHAT_ID}"

    def test_fail_open_distinct_chats_and_forum_topics(self, store):
        """CONTROL: different chat_id stay isolated; forum topics stay per-topic."""
        dm_a = _dm(chat_id="111", user_id="111")
        dm_b = _dm(chat_id="222", user_id="222")
        assert store._generate_session_key(dm_a) != store._generate_session_key(dm_b)
        entry_a = store.get_or_create_session(dm_a)
        entry_b = store.get_or_create_session(dm_b)
        assert entry_a.session_id != entry_b.session_id

        topic_a = _forum(thread_id="10")
        topic_b = _forum(thread_id="20")
        dm_root = _dm()
        key_topic_a = store._generate_session_key(topic_a)
        key_topic_b = store._generate_session_key(topic_b)
        key_dm = store._generate_session_key(dm_root)
        assert key_topic_a != key_topic_b
        assert key_topic_a != key_dm
        assert key_topic_b != key_dm
        # Forum keys still carry the topic thread_id.
        assert key_topic_a.endswith(":10")
        assert key_topic_b.endswith(":20")

        e_topic_a = store.get_or_create_session(topic_a)
        e_topic_b = store.get_or_create_session(topic_b)
        e_dm = store.get_or_create_session(dm_root)
        assert len({e_topic_a.session_id, e_topic_b.session_id, e_dm.session_id}) == 3

    def test_force_new_still_mints_a_new_session(self, store):
        src = _dm(thread_id="1001")
        live = store.get_or_create_session(src)
        nxt = store.get_or_create_session(src, force_new=True)
        assert nxt.session_id != live.session_id
        assert nxt.session_key == live.session_key

    def test_named_profile_stays_isolated(self, store):
        default_src = _dm(thread_id="11")
        named_src = _dm(thread_id="11", profile="coder")
        store.config.multiplex_profiles = True
        default_key = store._generate_session_key(default_src)
        named_key = store._generate_session_key(named_src)
        assert default_key.startswith("agent:main:")
        assert named_key.startswith("agent:coder:")
        assert default_key != named_key

    def test_topic_mode_bound_dm_topics_stay_isolated(self, store):
        """Fail-open: topic-mode + real non-General thread_id keeps per-topic sessions."""
        db = store._db
        assert db is not None
        db.enable_telegram_topic_mode(chat_id=CHAT_ID, user_id=USER_ID)
        topic_a = _dm(thread_id="17585")
        topic_b = _dm(thread_id="99999")
        lobby = _dm(thread_id=None)
        key_a = store._generate_session_key(topic_a)
        key_b = store._generate_session_key(topic_b)
        key_lobby = store._generate_session_key(lobby)
        assert key_a == f"agent:main:telegram:dm:{CHAT_ID}:17585"
        assert key_b == f"agent:main:telegram:dm:{CHAT_ID}:99999"
        assert key_lobby == f"agent:main:telegram:dm:{CHAT_ID}"
        assert key_a != key_b != key_lobby

        e_a = store.get_or_create_session(topic_a)
        e_b = store.get_or_create_session(topic_b)
        e_lobby = store.get_or_create_session(lobby)
        assert len({e_a.session_id, e_b.session_id, e_lobby.session_id}) == 3


class TestCoalescedKeySharedAcrossAdapterUndoAndHandoff:
    """Enough1122 review: adapter ``_active_sessions``, /undo eviction, and handoff
    destination keys must use the same coalesced key as SessionStore."""

    @pytest.mark.asyncio
    async def test_adapter_handle_message_guard_uses_store_root(self, store):
        from unittest.mock import AsyncMock

        from gateway.config import PlatformConfig
        from gateway.platforms.base import BasePlatformAdapter
        from gateway.platforms.event import MessageEvent, MessageType

        class _StubAdapter(BasePlatformAdapter):
            async def connect(self, *, is_reconnect: bool = False):
                pass

            async def disconnect(self):
                pass

            async def send(self, chat_id, text, **kwargs):
                pass

            async def get_chat_info(self, chat_id):
                return {}

        adapter = _StubAdapter(PlatformConfig(enabled=True, token="test-token"), Platform.TELEGRAM)
        adapter.set_session_store(store)
        adapter.set_message_handler(AsyncMock())
        started: list[str] = []

        def _start(event, session_key, **_kwargs):
            started.append(session_key)
            adapter._active_sessions[session_key] = object()
            return True

        adapter._start_session_processing = _start
        adapter._handle_message_while_active = AsyncMock()
        adapter._heal_stale_session_lock = lambda _key: None

        first = MessageEvent(
            text="one", message_type=MessageType.TEXT, source=_dm(thread_id="1001"),
        )
        second = MessageEvent(
            text="two", message_type=MessageType.TEXT, source=_dm(thread_id="1002"),
        )
        store_key = store._generate_session_key(first.source)
        assert store_key == f"agent:main:telegram:dm:{CHAT_ID}"

        await adapter.handle_message(first)
        await adapter.handle_message(second)

        assert started == [store_key]
        adapter._handle_message_while_active.assert_awaited_once()
        assert list(adapter._active_sessions) == [store_key]
        assert adapter._event_session_key(first) == adapter._event_session_key(second) == store_key

    @pytest.mark.asyncio
    async def test_undo_evicts_the_store_root_not_the_suffix(self, store):
        from gateway.platforms.event import MessageEvent, MessageType
        from gateway.run import GatewayRunner

        source = _dm(thread_id="40404")
        entry = store.get_or_create_session(source)
        store_key = store._generate_session_key(source)
        assert store_key == f"agent:main:telegram:dm:{CHAT_ID}"
        db = store._db
        assert db is not None
        if not db.get_messages_as_conversation(entry.session_id):
            db.append_message(entry.session_id, "user", "q1")
            db.append_message(entry.session_id, "assistant", "a1")

        evicted: list[str] = []
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig()
        runner.session_store = store
        runner._async_session_store = None
        runner._evict_cached_agent = evicted.append

        await GatewayRunner._handle_undo_command(
            runner,
            MessageEvent(text="/undo", message_type=MessageType.TEXT, source=source),
        )
        assert evicted == [store_key]
        assert not any(key.endswith(":40404") for key in evicted)

    def test_handoff_destination_key_matches_store(self, store):
        from types import SimpleNamespace

        from gateway.config import PlatformConfig
        from gateway.run import GatewayRunner

        dest = SimpleNamespace(
            source=_dm(thread_id="17585"),
            platform=Platform.TELEGRAM,
            handoff_config=GatewayConfig(
                platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test")},
            ),
        )
        runner = object.__new__(GatewayRunner)
        runner.config = dest.handoff_config
        runner.session_store = store
        runner._async_session_store = None

        coalesced = GatewayRunner._handoff_session_key(runner, dest, None)
        assert coalesced == store._generate_session_key(dest.source)
        assert coalesced == f"agent:main:telegram:dm:{CHAT_ID}"

        db = store._db
        assert db is not None
        db.enable_telegram_topic_mode(chat_id=CHAT_ID, user_id=USER_ID)
        topic = GatewayRunner._handoff_session_key(runner, dest, None)
        assert topic == store._generate_session_key(dest.source)
        assert topic == f"agent:main:telegram:dm:{CHAT_ID}:17585"
