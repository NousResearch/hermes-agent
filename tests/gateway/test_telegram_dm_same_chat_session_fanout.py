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
