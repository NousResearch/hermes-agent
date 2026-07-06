"""Tests for Telegram adapter shared-topic backfill -> event.channel_context.

Exercises ``_build_message_event`` / ``_compute_topic_backfill_context``:
- a SHARED-topic NEW session sets channel_context from seeded sibling DB rows
- a DM never sets it
- disabled config never sets it
- an ESTABLISHED session (non-empty transcript) never sets it

Uses a real SessionDB + sessions.json in a temp HERMES_HOME for the backfill
source, and a fake session store + monkeypatched gateway config for the gate.
"""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.config import GatewayConfig, PlatformConfig, TopicBackfillConfig  # noqa: E402
from gateway.platforms.base import MessageType  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEntry:
    def __init__(self, session_id):
        self.session_id = session_id


class _FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeSessionStore:
    """Minimal session-store stand-in matching the READ-ONLY peek contract.

    The adapter must detect a new-session window WITHOUT mutating the store
    (no get_or_create_session, which would bump updated_at and flip run.py's
    _is_new_session on the first real message). So this fake exposes the
    read-only surface the adapter actually uses: ``_generate_session_key``,
    ``_lock``, ``_ensure_loaded_locked``, ``_entries``, ``load_transcript``.

    ``entry_session_id=None`` => no entry for the key yet (brand-new window).
    ``transcript`` => what load_transcript returns for an existing entry.
    ``get_or_create_session`` is a TRAP: calling it fails the test, because
    doing so is the regression we are guarding against.
    """

    SESSION_KEY = "agent:main:telegram:group:100:5:7"

    def __init__(self, *, entry_session_id="EXIST", transcript=None):
        self._lock = _FakeLock()
        self._entries = {}
        if entry_session_id is not None:
            self._entries[self.SESSION_KEY] = _FakeEntry(entry_session_id)
        self._transcript = transcript or []
        self.get_or_create_called = False

    def _generate_session_key(self, source):
        return self.SESSION_KEY

    def _ensure_loaded_locked(self):
        return None

    def load_transcript(self, session_id):
        return self._transcript

    def get_or_create_session(self, source):  # pragma: no cover - trap
        # The adapter must NEVER call this during backfill computation; doing
        # so mutates updated_at and suppresses run.py's _is_new_session.
        self.get_or_create_called = True
        raise AssertionError(
            "adapter called get_or_create_session during backfill (mutation regression)"
        )


@pytest.fixture
def temp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    (home / "sessions").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _seed_sibling(home: Path, *, session_id, chat_id="100", thread_id="5",
                  user_id="u1", user_name="Alice", messages=None):
    index_path = home / "sessions" / "sessions.json"
    data = {}
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
    key = f"agent:main:telegram:group:{chat_id}:{thread_id or ''}:{user_id}"
    data[key] = {
        "session_key": key,
        "session_id": session_id,
        "origin": {
            "platform": "telegram", "chat_id": chat_id, "thread_id": thread_id,
            "user_id": user_id, "user_name": user_name, "chat_type": "group",
        },
    }
    index_path.write_text(json.dumps(data), encoding="utf-8")

    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id=session_id, source="telegram")
    for msg in messages or []:
        db.append_message(
            session_id=session_id, role=msg["role"], content=msg.get("content"),
            timestamp=msg.get("timestamp"),
        )


def _make_adapter(extra=None):
    return TelegramAdapter(PlatformConfig(enabled=True, token="***", extra=extra or {}))


def _make_group_message(chat_id=100, thread_id=5, user_id=7, text="hi"):
    chat = SimpleNamespace(
        id=chat_id, type="supergroup", title="Topic Chat", is_forum=True,
    )
    user = SimpleNamespace(id=user_id, full_name="Carol")
    return SimpleNamespace(
        chat=chat, from_user=user, text=text,
        message_thread_id=thread_id, is_topic_message=True,
        message_id=2001, reply_to_message=None, quote=None,
        date=None, forum_topic_created=None,
    )


def _make_dm_message(chat_id=500, user_id=7, text="hi"):
    chat = SimpleNamespace(id=chat_id, type="private", title=None, full_name="Carol")
    user = SimpleNamespace(id=user_id, full_name="Carol")
    return SimpleNamespace(
        chat=chat, from_user=user, text=text,
        message_thread_id=None, is_topic_message=False,
        message_id=2002, reply_to_message=None, quote=None,
        date=None, forum_topic_created=None,
    )


def _patch_gateway_cfg(monkeypatch, *, enabled=True, max_messages=15, max_age_hours=24):
    cfg = GatewayConfig(
        topic_backfill=TopicBackfillConfig(
            enabled=enabled, max_messages=max_messages, max_age_hours=max_age_hours,
        )
    )
    import gateway.config as gwc

    monkeypatch.setattr(gwc, "load_gateway_config", lambda: cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_shared_topic_new_session_sets_channel_context(temp_home, monkeypatch):
    now = time.time()
    _seed_sibling(
        temp_home, session_id="SIB1", chat_id="100", thread_id="5",
        user_name="Alice",
        messages=[{"role": "user", "content": "prior topic chatter", "timestamp": now - 30}],
    )
    _patch_gateway_cfg(monkeypatch, enabled=True)

    adapter = _make_adapter()
    adapter.set_session_store(_FakeSessionStore(entry_session_id="NEW", transcript=[]))

    event = adapter._build_message_event(_make_group_message(), MessageType.TEXT)

    assert event.channel_context is not None
    assert "prior topic chatter" in event.channel_context
    assert "Alice" in event.channel_context


def test_dm_never_sets_channel_context(temp_home, monkeypatch):
    now = time.time()
    # Even with sibling data present in the (irrelevant) topic, a DM must skip.
    _seed_sibling(
        temp_home, session_id="SIB1", chat_id="100", thread_id="5",
        messages=[{"role": "user", "content": "topic chatter", "timestamp": now - 30}],
    )
    _patch_gateway_cfg(monkeypatch, enabled=True)

    adapter = _make_adapter()
    adapter.set_session_store(_FakeSessionStore(entry_session_id="NEW", transcript=[]))

    event = adapter._build_message_event(_make_dm_message(), MessageType.TEXT)

    assert event.channel_context is None


def test_disabled_config_does_not_set_channel_context(temp_home, monkeypatch):
    now = time.time()
    _seed_sibling(
        temp_home, session_id="SIB1", chat_id="100", thread_id="5",
        messages=[{"role": "user", "content": "topic chatter", "timestamp": now - 30}],
    )
    _patch_gateway_cfg(monkeypatch, enabled=False)

    adapter = _make_adapter()
    adapter.set_session_store(_FakeSessionStore(entry_session_id="NEW", transcript=[]))

    event = adapter._build_message_event(_make_group_message(), MessageType.TEXT)

    assert event.channel_context is None


def test_established_session_does_not_set_channel_context(temp_home, monkeypatch):
    now = time.time()
    _seed_sibling(
        temp_home, session_id="SIB1", chat_id="100", thread_id="5",
        messages=[{"role": "user", "content": "topic chatter", "timestamp": now - 30}],
    )
    _patch_gateway_cfg(monkeypatch, enabled=True)

    adapter = _make_adapter()
    # Non-empty transcript with a REAL user turn -> established -> skip.
    established = _FakeSessionStore(
        entry_session_id="EXIST",
        transcript=[{"role": "user", "content": "already talking here"}],
    )
    adapter.set_session_store(established)

    event = adapter._build_message_event(_make_group_message(), MessageType.TEXT)

    assert event.channel_context is None


def test_observed_only_transcript_still_treated_as_new(temp_home, monkeypatch):
    """A transcript with only OBSERVED rows (no real turns) is still new."""
    now = time.time()
    _seed_sibling(
        temp_home, session_id="SIB1", chat_id="100", thread_id="5",
        user_name="Alice",
        messages=[{"role": "user", "content": "prior topic chatter", "timestamp": now - 30}],
    )
    _patch_gateway_cfg(monkeypatch, enabled=True)

    adapter = _make_adapter()
    observed_only = _FakeSessionStore(
        entry_session_id="NEW",
        transcript=[{"role": "user", "content": "observed line", "observed": True}],
    )
    adapter.set_session_store(observed_only)

    event = adapter._build_message_event(_make_group_message(), MessageType.TEXT)

    assert event.channel_context is not None
    assert "prior topic chatter" in event.channel_context


def test_no_session_store_does_not_set_channel_context(temp_home, monkeypatch):
    now = time.time()
    _seed_sibling(
        temp_home, session_id="SIB1", chat_id="100", thread_id="5",
        messages=[{"role": "user", "content": "topic chatter", "timestamp": now - 30}],
    )
    _patch_gateway_cfg(monkeypatch, enabled=True)

    adapter = _make_adapter()
    # No session store set -> cannot detect new window -> skip.

    event = adapter._build_message_event(_make_group_message(), MessageType.TEXT)

    assert event.channel_context is None


def test_backfill_computation_does_not_mutate_session_store(temp_home, monkeypatch):
    """Regression: computing backfill must NOT call get_or_create_session.

    Calling it would bump the entry's updated_at, so run.py's later
    get_or_create_session reuse path sets updated_at > created_at and
    _is_new_session (created_at == updated_at) flips to False on the very
    first message of a shared topic — suppressing the session:start hook and
    first-turn skill/channel-prompt injection. The fake store raises if
    get_or_create_session is touched; this test asserts the read-only path.
    """
    now = time.time()
    _seed_sibling(
        temp_home, session_id="SIB1", chat_id="100", thread_id="5",
        user_name="Alice",
        messages=[{"role": "user", "content": "prior topic chatter", "timestamp": now - 30}],
    )
    _patch_gateway_cfg(monkeypatch, enabled=True)

    adapter = _make_adapter()
    store = _FakeSessionStore(entry_session_id="NEW", transcript=[])
    adapter.set_session_store(store)

    # Must not raise (the trap fires inside get_or_create_session) and must
    # still produce the backfill via the read-only peek path.
    event = adapter._build_message_event(_make_group_message(), MessageType.TEXT)

    assert store.get_or_create_called is False
    assert event.channel_context is not None
    assert "prior topic chatter" in event.channel_context


def test_brand_new_key_with_no_entry_is_treated_as_new(temp_home, monkeypatch):
    """No entry yet for the session key (truly first message) => new window."""
    now = time.time()
    _seed_sibling(
        temp_home, session_id="SIB1", chat_id="100", thread_id="5",
        user_name="Alice",
        messages=[{"role": "user", "content": "prior topic chatter", "timestamp": now - 30}],
    )
    _patch_gateway_cfg(monkeypatch, enabled=True)

    adapter = _make_adapter()
    # entry_session_id=None => _entries is empty => no entry for the key.
    adapter.set_session_store(_FakeSessionStore(entry_session_id=None, transcript=[]))

    event = adapter._build_message_event(_make_group_message(), MessageType.TEXT)

    assert event.channel_context is not None
    assert "prior topic chatter" in event.channel_context
