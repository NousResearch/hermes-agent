"""Tests for the Telegram reaction-capture feedback loop.

Covers three layers:
  * SessionDB storage (outbound index + reaction events + review marking)
  * The TelegramAdapter reaction handler (diff + correlation + opt-in gate)
  * The weekly reaction-review digest builder
"""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


# ── Storage layer ───────────────────────────────────────────────────────


class TestReactionStorage:
    def test_outbound_index_roundtrip_and_snippet_normalisation(self, db):
        db.record_telegram_outbound_message(
            chat_id="100",
            message_id="5",
            thread_id="2",
            session_id="sess1",
            snippet="line one\n\n  multiple   spaces  ",
        )
        got = db.lookup_telegram_outbound_message(chat_id="100", message_id="5")
        assert got is not None
        assert got["session_id"] == "sess1"
        assert got["thread_id"] == "2"
        # whitespace collapsed to single spaces
        assert got["snippet"] == "line one multiple spaces"

    def test_outbound_index_upsert_preserves_session_when_resent(self, db):
        db.record_telegram_outbound_message(
            chat_id="100", message_id="5", session_id="sess1", snippet="first"
        )
        # A later upsert without a session_id must not clobber the known one.
        db.record_telegram_outbound_message(
            chat_id="100", message_id="5", session_id=None, snippet="second"
        )
        got = db.lookup_telegram_outbound_message(chat_id="100", message_id="5")
        assert got["session_id"] == "sess1"
        assert got["snippet"] == "second"

    def test_reaction_autoenriches_from_index(self, db):
        db.record_telegram_outbound_message(
            chat_id="100", message_id="5", session_id="sess1", snippet="useful answer"
        )
        rid = db.record_telegram_reaction(
            chat_id="100", message_id="5", emoji="👍", user_id="u1", user_name="sasha"
        )
        assert isinstance(rid, int) and rid > 0
        rows = db.list_telegram_reactions()
        assert len(rows) == 1
        assert rows[0]["emoji"] == "👍"
        assert rows[0]["session_id"] == "sess1"
        assert rows[0]["snippet"] == "useful answer"
        assert rows[0]["action"] == "add"
        assert rows[0]["user_name"] == "sasha"

    def test_reaction_without_index_still_records(self, db):
        rid = db.record_telegram_reaction(
            chat_id="100", message_id="999", emoji="❤", user_id="u1"
        )
        assert rid > 0
        rows = db.list_telegram_reactions()
        assert len(rows) == 1
        assert rows[0]["session_id"] is None
        assert rows[0]["snippet"] is None

    def test_action_filter_add_vs_remove(self, db):
        db.record_telegram_reaction(chat_id="1", message_id="1", emoji="👍", action="add")
        db.record_telegram_reaction(chat_id="1", message_id="1", emoji="👍", action="remove")
        assert len(db.list_telegram_reactions(actions=("add",))) == 1
        assert len(db.list_telegram_reactions(actions=("add", "remove"))) == 2

    def test_mark_reviewed_excludes_from_unreviewed(self, db):
        db.record_telegram_reaction(chat_id="1", message_id="1", emoji="👍")
        db.record_telegram_reaction(chat_id="1", message_id="2", emoji="💩")
        unrev = db.list_telegram_reactions(only_unreviewed=True)
        assert len(unrev) == 2
        n = db.mark_telegram_reactions_reviewed([r["id"] for r in unrev])
        assert n == 2
        assert db.list_telegram_reactions(only_unreviewed=True) == []

    def test_read_helpers_are_safe_before_migration(self, db):
        """Read paths must not raise (or trigger migration) on a fresh DB."""
        # Fresh DB: reactions tables don't exist yet.
        assert db.lookup_telegram_outbound_message(chat_id="1", message_id="1") is None
        assert db.list_telegram_reactions() == []
        assert db.mark_telegram_reactions_reviewed([]) == 0


# ── Telegram adapter handler ─────────────────────────────────────────────


def _make_adapter(db, *, capture="1", monkeypatch=None):
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from gateway.config import PlatformConfig
    from gateway.platforms.base import Platform

    if monkeypatch is not None:
        monkeypatch.setenv("HERMES_TELEGRAM_REACTION_CAPTURE", capture)

    adapter = object.__new__(TelegramAdapter)
    adapter._platform = Platform.TELEGRAM
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="x")
    adapter._session_store = SimpleNamespace(
        _db=db, _entries={}, _ensure_loaded=lambda: None
    )
    return adapter


def _emoji_reaction(emoji):
    return SimpleNamespace(emoji=emoji, custom_emoji_id=None, type="emoji")


class TestReactionDiff:
    def test_add_swap_remove(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        add, rem = TelegramAdapter._diff_reactions([], [_emoji_reaction("❤")])
        assert add == [("emoji", "❤")] and rem == []

        add, rem = TelegramAdapter._diff_reactions(
            [_emoji_reaction("❤")], [_emoji_reaction("💩")]
        )
        assert add == [("emoji", "💩")] and rem == [("emoji", "❤")]

        add, rem = TelegramAdapter._diff_reactions([_emoji_reaction("👍")], [])
        assert add == [] and rem == [("emoji", "👍")]

    def test_custom_and_paid(self):
        from plugins.platforms.telegram.adapter import TelegramAdapter

        custom = SimpleNamespace(emoji=None, custom_emoji_id="555", type="custom_emoji")
        paid = SimpleNamespace(emoji=None, custom_emoji_id=None, type="paid")
        add, _ = TelegramAdapter._diff_reactions([], [custom])
        assert add == [("custom_emoji", "555")]
        add, _ = TelegramAdapter._diff_reactions([], [paid])
        assert add == [("paid", "paid")]


class TestReactionHandler:
    def test_capture_correlates_to_indexed_message(self, db, monkeypatch):
        adapter = _make_adapter(db, monkeypatch=monkeypatch)
        # Index an outbound bot message first.
        event = SimpleNamespace(
            source=SimpleNamespace(chat_id="123", thread_id="2")
        )
        result = SimpleNamespace(success=True, message_id="77")
        adapter._on_final_response_sent(event, "telegram:123:2", result, "concise answer")
        assert db.lookup_telegram_outbound_message(chat_id="123", message_id="77")

        update = SimpleNamespace(
            message_reaction=SimpleNamespace(
                chat=SimpleNamespace(id=123),
                message_id=77,
                user=SimpleNamespace(id=42, username="sasha", full_name="Sasha", first_name="Sasha"),
                old_reaction=[],
                new_reaction=[_emoji_reaction("💩")],
            )
        )
        asyncio.run(adapter._handle_message_reaction(update, None))
        rows = db.list_telegram_reactions()
        assert len(rows) == 1
        assert rows[0]["emoji"] == "💩"
        assert rows[0]["snippet"] == "concise answer"
        assert rows[0]["user_name"] == "sasha"

    def test_reaction_on_non_bot_message_is_skipped(self, db, monkeypatch):
        adapter = _make_adapter(db, monkeypatch=monkeypatch)
        update = SimpleNamespace(
            message_reaction=SimpleNamespace(
                chat=SimpleNamespace(id=123),
                message_id=999,  # never indexed
                user=SimpleNamespace(id=42, username="sasha"),
                old_reaction=[],
                new_reaction=[_emoji_reaction("👍")],
            )
        )
        asyncio.run(adapter._handle_message_reaction(update, None))
        assert db.list_telegram_reactions() == []

    def test_disabled_capture_is_noop(self, db, monkeypatch):
        adapter = _make_adapter(db, capture="false", monkeypatch=monkeypatch)
        event = SimpleNamespace(source=SimpleNamespace(chat_id="123", thread_id=None))
        result = SimpleNamespace(success=True, message_id="77")
        adapter._on_final_response_sent(event, "k", result, "answer")
        # Nothing indexed because capture is off.
        assert db.lookup_telegram_outbound_message(chat_id="123", message_id="77") is None
        update = SimpleNamespace(
            message_reaction=SimpleNamespace(
                chat=SimpleNamespace(id=123), message_id=77,
                user=SimpleNamespace(id=1), old_reaction=[], new_reaction=[_emoji_reaction("👍")],
            )
        )
        asyncio.run(adapter._handle_message_reaction(update, None))
        assert db.list_telegram_reactions() == []


# ── Weekly digest ────────────────────────────────────────────────────────


class TestReactionDigest:
    def _seed(self, db):
        db.record_telegram_outbound_message(chat_id="1", message_id="10", session_id="s1", snippet="useful concise answer")
        db.record_telegram_reaction(chat_id="1", message_id="10", emoji="❤", user_name="sasha")
        db.record_telegram_outbound_message(chat_id="1", message_id="20", session_id="s2", snippet="bad rambling answer")
        db.record_telegram_reaction(chat_id="1", message_id="20", emoji="💩", user_name="sasha")
        db.record_telegram_outbound_message(chat_id="1", message_id="30", session_id="s3", snippet="ambiguous thing")
        db.record_telegram_reaction(chat_id="1", message_id="30", emoji="🤔", user_name="sasha")

    def test_classify(self):
        from gateway.reaction_review import classify_emoji

        assert classify_emoji("💩") == "negative"
        assert classify_emoji("❤") == "positive"
        assert classify_emoji("🤔") == "neutral"
        assert classify_emoji("🦄") == "other"

    def test_build_digest_buckets_and_followups(self, db):
        from gateway.reaction_review import build_digest

        self._seed(db)
        d = build_digest(db, since_days=7, mark_reviewed=False)
        assert d["total"] == 3
        assert d["empty"] is False
        assert len(d["buckets"]["negative"]) == 1
        assert len(d["buckets"]["positive"]) == 1
        assert len(d["buckets"]["neutral"]) == 1
        # Negatives surface first in the follow-up list.
        assert d["followups"][0].startswith("You put 💩")
        assert "what went wrong" in d["followups"][0]
        assert "🗳" in d["text"]

    def test_mark_reviewed_makes_next_run_empty(self, db):
        from gateway.reaction_review import build_digest

        self._seed(db)
        d1 = build_digest(db, since_days=7, mark_reviewed=True)
        assert d1["total"] == 3 and len(d1["reviewed_ids"]) == 3
        d2 = build_digest(db, since_days=7)
        assert d2["empty"] is True
        assert d2["total"] == 0

    def test_empty_window_message(self, db):
        from gateway.reaction_review import build_digest

        d = build_digest(db, since_days=7)
        assert d["empty"] is True
        assert "Nothing to review" in d["text"]
