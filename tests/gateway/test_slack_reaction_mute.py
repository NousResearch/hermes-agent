"""Per-thread response mute toggled by a principal's emoji reaction (Slack).

Proves the behaviour end to end at the adapter boundary, not just that the code path exists:

* a mute reaction from a principal on a *reply* keys the mute on the thread parent;
* under ``enforce`` a later message in that thread never reaches the gateway runner but IS
  appended to the thread session's transcript (ingest-only);
* under ``log-only`` the runner is still called and a ``would_suppress`` audit row is written;
* two principals → one removal keeps the thread muted, the last removal unmutes;
* re-adding / removing an unknown reactor is a no-op in both directions;
* non-principals are ignored (and audited);
* the mute survives a new adapter instance on the same db file (restart);
* slash commands bypass the mute.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from tests.gateway.test_slack_approval_buttons import _ensure_slack_mock  # noqa: E402

_ensure_slack_mock()

from gateway.config import GatewayConfig, PlatformConfig  # noqa: E402
from gateway.session import SessionStore  # noqa: E402
from hermes_state import SessionDB  # noqa: E402
from plugins.platforms.slack.adapter import SlackAdapter, _apply_yaml_config  # noqa: E402
from plugins.platforms.slack.reaction_mute import (  # noqa: E402
    MODE_ENFORCE, MODE_LOG_ONLY, MODE_OFF, ReactionMuteStore, normalize_emoji, normalize_mode,
)

CHANNEL = "C1"
PARENT_TS = "1000.000"
REPLY_TS = "1001.000"
ROB = "U_ROB"
SEAN = "U_SEAN"
STRANGER = "U_STRANGER"


def run(coro):
    return asyncio.run(coro)


def react(adapter, event):
    """Drive the handler the way the Bolt listener closure does (``removed`` from the event type)."""
    return run(adapter._handle_slack_reaction(event, removed=event["type"] == "reaction_removed"))


class _FakeEntry:
    def __init__(self, session_id):
        self.session_id = session_id


class _FakeSessionStore:
    """Just enough of SessionStore for the ingest-only path."""

    def __init__(self):
        self.rows = []
        self.sources = []

    def get_or_create_session(self, source, force_new=False, touch_activity=True):
        self.sources.append(source)
        return _FakeEntry(f"sess-{source.chat_id}-{source.thread_id}")

    def has_platform_message_id(self, session_id, platform_message_id):
        return any(r.get("message_id") == platform_message_id for _, r in self.rows)

    def append_to_transcript(self, session_id, message, skip_db=False):
        self.rows.append((session_id, message))


def _make_adapter(tmp_path, mode=MODE_ENFORCE, users=(ROB, SEAN), emoji=None, monkeypatch=None):
    if monkeypatch is not None:
        for var in ("SLACK_REACTION_MUTE_MODE", "SLACK_REACTION_MUTE_EMOJI", "SLACK_REACTION_MUTE_USERS",
                    "SLACK_REACTION_MUTE_DB", "SLACK_ALLOWED_USERS", "SLACK_REQUIRE_MENTION",
                    "SLACK_FREE_RESPONSE_CHANNELS", "SLACK_REACTION_TRIGGERS"):
            monkeypatch.delenv(var, raising=False)
    extra = {
        "reaction_mute_mode": mode,
        "reaction_mute_users": list(users),
        "reaction_mute_db": str(tmp_path / "mute.db"),
        "require_mention": False,  # free-response channel: every message would start a turn
    }
    if emoji is not None:
        extra["reaction_mute_emoji"] = emoji
    config = PlatformConfig(enabled=True, token="xoxb-test-token", extra=extra)
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._bot_user_id = "U_BOT"
    adapter._team_clients = {"T1": AsyncMock()}
    adapter._team_bot_user_ids = {"T1": "U_BOT"}
    adapter._channel_team = {CHANNEL: "T1"}
    adapter._running = True
    # The reacted-to message is a reply whose parent is PARENT_TS.
    adapter._team_clients["T1"].conversations_replies = AsyncMock(return_value={
        "messages": [{"ts": REPLY_TS, "thread_ts": PARENT_TS, "user": "U_HUMAN", "text": "hi"}]})
    # Runner handoff and the enrichment that would hit Slack.
    adapter.handle_message = AsyncMock()
    adapter._fetch_thread_context = AsyncMock(return_value="")
    adapter._fetch_thread_parent_text = AsyncMock(return_value="")
    adapter._has_active_session_for_thread = MagicMock(return_value=False)
    adapter._resolve_user_name = AsyncMock(return_value="Test User")
    adapter._resolve_channel_name = AsyncMock(return_value="general")
    adapter._session_store = _FakeSessionStore()
    return adapter


def _reaction(user, *, removed=False, ts=REPLY_TS, reaction="mute", event_ts="3000.0"):
    return {
        "type": "reaction_removed" if removed else "reaction_added",
        "user": user,
        "reaction": reaction,
        "item": {"type": "message", "channel": CHANNEL, "ts": ts},
        "item_user": "U_HUMAN",
        "event_ts": event_ts,
    }


def _message(text, *, ts, thread_ts=None, user="U_HUMAN"):
    event = {
        "type": "message", "channel": CHANNEL, "channel_type": "channel", "team": "T1",
        "user": user, "text": text, "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


# ---------------------------------------------------------------------------
# Store semantics (pure, no Slack)
# ---------------------------------------------------------------------------


def test_store_reactor_set_and_idempotency(tmp_path):
    store = ReactionMuteStore(tmp_path / "m.db")
    assert not store.is_muted(CHANNEL, PARENT_TS)
    assert store.add_reactor(CHANNEL, PARENT_TS, ROB) == (True, True)
    assert store.add_reactor(CHANNEL, PARENT_TS, ROB) == (True, False)  # idempotent re-add
    assert store.add_reactor(CHANNEL, PARENT_TS, SEAN) == (True, True)
    assert store.reactors(CHANNEL, PARENT_TS) == {ROB, SEAN}
    assert store.remove_reactor(CHANNEL, PARENT_TS, ROB) == (True, True)  # still muted
    assert store.is_muted(CHANNEL, PARENT_TS)
    assert store.remove_reactor(CHANNEL, PARENT_TS, ROB) == (True, False)  # unknown reactor: no-op
    assert store.remove_reactor(CHANNEL, PARENT_TS, SEAN) == (False, True)  # last one out
    assert not store.is_muted(CHANNEL, PARENT_TS)
    assert store.remove_reactor(CHANNEL, PARENT_TS, SEAN) == (False, False)
    assert store.list_mutes() == []


def test_store_unique_constraint_is_real(tmp_path):
    store = ReactionMuteStore(tmp_path / "m.db")
    store.add_reactor(CHANNEL, PARENT_TS, ROB)
    conn = store._connection()
    ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'thread_mutes'").fetchone()[0]
    assert "UNIQUE (channel_id, thread_ts)" in ddl
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO thread_mutes (channel_id, thread_ts, created_at, updated_at) VALUES (?, ?, 0, 0)",
                     (CHANNEL, PARENT_TS))
    assert conn.execute("SELECT COUNT(*) FROM thread_mutes").fetchone()[0] == 1


def test_store_survives_reopen(tmp_path):
    ReactionMuteStore(tmp_path / "m.db").add_reactor(CHANNEL, PARENT_TS, ROB)
    fresh = ReactionMuteStore(tmp_path / "m.db")
    assert fresh.is_muted(CHANNEL, PARENT_TS)
    assert fresh.reactors(CHANNEL, PARENT_TS) == {ROB}


def test_mode_and_emoji_normalization():
    assert normalize_mode(None) == MODE_OFF
    assert normalize_mode("garbage") == MODE_OFF  # fail closed
    assert normalize_mode("log_only") == MODE_LOG_ONLY
    assert normalize_mode("LOG-ONLY") == MODE_LOG_ONLY
    assert normalize_mode("enforce") == MODE_ENFORCE
    assert normalize_mode(True) == MODE_ENFORCE
    assert normalize_emoji(":Mute:") == "mute"
    assert normalize_emoji("") == "mute"
    assert normalize_emoji("no_bell") == "no_bell"


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_yaml_bridge_sets_env(monkeypatch):
    for var in ("SLACK_REACTION_MUTE_MODE", "SLACK_REACTION_MUTE_EMOJI", "SLACK_REACTION_MUTE_USERS"):
        monkeypatch.delenv(var, raising=False)
    seeded = _apply_yaml_config({}, {
        "reaction_mute_mode": "log-only", "reaction_mute_emoji": "no_bell",
        "reaction_mute_users": [ROB, SEAN]})
    import os
    assert os.environ["SLACK_REACTION_MUTE_MODE"] == "log-only"
    assert os.environ["SLACK_REACTION_MUTE_EMOJI"] == "no_bell"
    assert os.environ["SLACK_REACTION_MUTE_USERS"] == f"{ROB},{SEAN}"
    assert seeded["reaction_mute_users"] == [ROB, SEAN]
    for var in ("SLACK_REACTION_MUTE_MODE", "SLACK_REACTION_MUTE_EMOJI", "SLACK_REACTION_MUTE_USERS"):
        monkeypatch.delenv(var, raising=False)


def test_principals_fall_back_to_allowed_users(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, users=(), monkeypatch=monkeypatch)
    adapter.config.extra.pop("reaction_mute_users")
    assert adapter._slack_reaction_mute_users() == set()  # nothing configured: nobody
    monkeypatch.setenv("SLACK_ALLOWED_USERS", f"{ROB}, {SEAN}")
    assert adapter._slack_reaction_mute_users() == {ROB, SEAN}
    adapter.config.extra["reaction_mute_users"] = "U_ONLY"
    assert adapter._slack_reaction_mute_users() == {"U_ONLY"}  # explicit list wins


# ---------------------------------------------------------------------------
# Toggle via reactions (adapter boundary)
# ---------------------------------------------------------------------------


def test_reaction_on_reply_mutes_parent_thread(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, monkeypatch=monkeypatch)
    react(adapter, _reaction(ROB))
    store = adapter._reaction_mute_store()
    assert store.is_muted(CHANNEL, PARENT_TS)
    assert not store.is_muted(CHANNEL, REPLY_TS)  # keyed on the resolved parent, not item.ts
    assert store.reactors(CHANNEL, PARENT_TS) == {ROB}
    assert store.events(kind="muted")[0]["thread_ts"] == PARENT_TS
    # The mute emoji never becomes a synthetic agent message (reaction_triggers stays off).
    adapter.handle_message.assert_not_called()


def test_reaction_on_parent_keys_on_itself(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, monkeypatch=monkeypatch)
    adapter._team_clients["T1"].conversations_replies = AsyncMock(return_value={
        "messages": [{"ts": PARENT_TS, "thread_ts": PARENT_TS, "user": "U_HUMAN", "text": "root"}]})
    react(adapter, _reaction(ROB, ts=PARENT_TS))
    assert adapter._reaction_mute_store().is_muted(CHANNEL, PARENT_TS)


def test_non_principal_reaction_is_ignored_and_audited(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, monkeypatch=monkeypatch)
    react(adapter, _reaction(STRANGER))
    store = adapter._reaction_mute_store()
    assert not store.is_muted(CHANNEL, PARENT_TS)
    ignored = store.events(kind="ignored_non_principal")
    assert len(ignored) == 1 and ignored[0]["user_id"] == STRANGER
    # ...and a stranger cannot unmute either.
    react(adapter, _reaction(ROB))
    react(adapter, _reaction(STRANGER, removed=True))
    assert store.is_muted(CHANNEL, PARENT_TS)


def test_other_emoji_does_nothing(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, monkeypatch=monkeypatch)
    react(adapter, _reaction(ROB, reaction="thumbsup"))
    assert not adapter._reaction_mute_store().is_muted(CHANNEL, PARENT_TS)
    assert adapter._reaction_mute_store().events() == []


def test_configured_emoji_is_honoured(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, emoji=":no_bell:", monkeypatch=monkeypatch)
    react(adapter, _reaction(ROB, reaction="mute"))
    assert not adapter._reaction_mute_store().is_muted(CHANNEL, PARENT_TS)
    react(adapter, _reaction(ROB, reaction="no_bell"))
    assert adapter._reaction_mute_store().is_muted(CHANNEL, PARENT_TS)


def test_two_principals_last_removal_unmutes(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, monkeypatch=monkeypatch)
    store = adapter._reaction_mute_store()
    react(adapter, _reaction(ROB))
    react(adapter, _reaction(SEAN))
    react(adapter, _reaction(ROB))  # duplicate add: idempotent
    assert store.reactors(CHANNEL, PARENT_TS) == {ROB, SEAN}
    react(adapter, _reaction(ROB, removed=True))
    assert store.is_muted(CHANNEL, PARENT_TS)  # Sean's reaction still holds it
    react(adapter, _reaction(ROB, removed=True))  # duplicate remove: no-op
    assert store.is_muted(CHANNEL, PARENT_TS)
    react(adapter, _reaction(SEAN, removed=True))
    assert not store.is_muted(CHANNEL, PARENT_TS)
    kinds = [e["kind"] for e in reversed(store.events())]
    assert kinds == ["muted", "reactor_added", "noop_add", "reactor_removed", "noop_remove", "unmuted"]


def test_mute_emoji_never_routes_to_agent_even_with_reaction_triggers(tmp_path, monkeypatch):
    """With reaction_triggers allowing the mute emoji, the toggle consumes it: no synthetic turn."""
    adapter = _make_adapter(tmp_path, monkeypatch=monkeypatch)
    adapter.config.extra["reaction_triggers"] = ["mute", "thumbsup"]
    forwarded = []

    async def _capture(event):
        forwarded.append(event)

    adapter._handle_slack_message = _capture
    react(adapter, _reaction(ROB))
    assert adapter._reaction_mute_store().is_muted(CHANNEL, PARENT_TS)
    assert forwarded == []  # consumed by the mute, never a reaction:added turn
    react(adapter, _reaction(ROB, reaction="thumbsup"))
    assert len(forwarded) == 1  # other emojis still route as before


def test_mode_off_tracks_nothing(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, mode=MODE_OFF, monkeypatch=monkeypatch)
    react(adapter, _reaction(ROB))
    assert not (tmp_path / "mute.db").exists()


# ---------------------------------------------------------------------------
# Responder gate (full inbound path through _handle_slack_message)
# ---------------------------------------------------------------------------


def test_enforce_suppresses_turn_but_ingests_transcript_row(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, mode=MODE_ENFORCE, monkeypatch=monkeypatch)
    # Control: before the mute, a thread reply reaches the runner.
    run(adapter._handle_slack_message(_message("before mute", ts="1002.000", thread_ts=PARENT_TS)))
    assert adapter.handle_message.await_count == 1
    adapter.handle_message.reset_mock()

    react(adapter, _reaction(ROB))
    run(adapter._handle_slack_message(_message("humans only please", ts="1003.000", thread_ts=PARENT_TS)))

    adapter.handle_message.assert_not_called()  # suppression fired
    store = adapter._reaction_mute_store()
    suppressed = store.events(kind="suppressed")
    assert len(suppressed) == 1
    assert suppressed[0]["thread_ts"] == PARENT_TS and suppressed[0]["message_ts"] == "1003.000"
    # Ingest-only: the message landed in the thread session's transcript as a user row.
    rows = adapter._session_store.rows
    assert len(rows) == 1
    session_id, row = rows[0]
    assert session_id == f"sess-{CHANNEL}-{PARENT_TS}"
    assert row["role"] == "user" and row["message_id"] == "1003.000"
    assert "humans only please" in row["content"]
    assert "Test User | Slack user <@U_HUMAN>" in row["content"]
    # Re-delivery of the same platform message does not stack a second row.
    run(adapter._handle_slack_message(_message("humans only please", ts="1003.000", thread_ts=PARENT_TS)))
    assert len(adapter._session_store.rows) == 1

    # Other threads are untouched.
    run(adapter._handle_slack_message(_message("elsewhere", ts="2002.000", thread_ts="2000.000")))
    assert adapter.handle_message.await_count == 1


def test_enforce_top_level_message_in_muted_thread_root(tmp_path, monkeypatch):
    """A mute on a top-level message (no replies yet) keys on its ts; the first reply carries
    thread_ts == that ts and is suppressed."""
    adapter = _make_adapter(tmp_path, mode=MODE_ENFORCE, monkeypatch=monkeypatch)
    adapter._team_clients["T1"].conversations_replies = AsyncMock(return_value={
        "messages": [{"ts": "5000.000", "user": "U_HUMAN", "text": "root without thread"}]})
    react(adapter, _reaction(ROB, ts="5000.000"))
    run(adapter._handle_slack_message(_message("first reply", ts="5001.000", thread_ts="5000.000")))
    adapter.handle_message.assert_not_called()


def test_unmute_restores_responses(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, mode=MODE_ENFORCE, monkeypatch=monkeypatch)
    react(adapter, _reaction(ROB))
    react(adapter, _reaction(ROB, removed=True))
    run(adapter._handle_slack_message(_message("back", ts="1004.000", thread_ts=PARENT_TS)))
    assert adapter.handle_message.await_count == 1


def test_log_only_records_but_still_responds(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, mode=MODE_LOG_ONLY, monkeypatch=monkeypatch)
    react(adapter, _reaction(ROB))
    run(adapter._handle_slack_message(_message("still answered", ts="1005.000", thread_ts=PARENT_TS)))
    assert adapter.handle_message.await_count == 1  # responds anyway
    store = adapter._reaction_mute_store()
    would = store.events(kind="would_suppress")
    assert len(would) == 1 and would[0]["message_ts"] == "1005.000" and would[0]["mode"] == "log-only"
    assert store.events(kind="suppressed") == []
    assert adapter._session_store.rows == []  # the normal turn persists it, not the mute path


def test_slash_command_bypasses_mute(tmp_path, monkeypatch):
    adapter = _make_adapter(tmp_path, mode=MODE_ENFORCE, monkeypatch=monkeypatch)
    react(adapter, _reaction(ROB))
    run(adapter._handle_slack_message(_message("/status", ts="1006.000", thread_ts=PARENT_TS)))
    assert adapter.handle_message.await_count == 1
    assert adapter._reaction_mute_store().events(kind="command_bypass")


def test_mute_survives_restart(tmp_path, monkeypatch):
    first = _make_adapter(tmp_path, mode=MODE_ENFORCE, monkeypatch=monkeypatch)
    react(first, _reaction(ROB))
    first._reaction_mute_store().close()

    second = _make_adapter(tmp_path, mode=MODE_ENFORCE, monkeypatch=monkeypatch)  # same db file
    run(second._handle_slack_message(_message("after restart", ts="1007.000", thread_ts=PARENT_TS)))
    second.handle_message.assert_not_called()
    assert second._reaction_mute_store().reactors(CHANNEL, PARENT_TS) == {ROB}


def test_mode_off_never_suppresses_even_with_stale_db(tmp_path, monkeypatch):
    ReactionMuteStore(tmp_path / "mute.db").add_reactor(CHANNEL, PARENT_TS, ROB)
    adapter = _make_adapter(tmp_path, mode=MODE_OFF, monkeypatch=monkeypatch)
    run(adapter._handle_slack_message(_message("off", ts="1008.000", thread_ts=PARENT_TS)))
    assert adapter.handle_message.await_count == 1


def test_enforce_ingest_writes_a_real_transcript_row(tmp_path, monkeypatch):
    """Same as the enforce test but against the real SessionStore + SessionDB: the suppressed
    message must be readable back through load_transcript for the thread's session, and a
    redelivery of the same platform ts must not add a second row."""
    adapter = _make_adapter(tmp_path, mode=MODE_ENFORCE, monkeypatch=monkeypatch)
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / "state.db")  # isolated state.db, never the real one
    adapter._session_store = store

    react(adapter, _reaction(ROB))
    run(adapter._handle_slack_message(_message("persist me", ts="1009.000", thread_ts=PARENT_TS)))
    run(adapter._handle_slack_message(_message("persist me", ts="1009.000", thread_ts=PARENT_TS)))
    adapter.handle_message.assert_not_called()

    entries = [e for e in store._entries.values() if e.platform.value == "slack"]
    assert len(entries) == 1
    rows = [r for r in store.load_transcript(entries[0].session_id) if r.get("role") == "user"]
    assert len(rows) == 1
    assert "persist me" in rows[0]["content"]
    assert "Test User | Slack user <@U_HUMAN>" in rows[0]["content"]
    assert store.has_platform_message_id(entries[0].session_id, "1009.000")
