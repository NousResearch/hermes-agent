"""Contract tests for the opt-in channel-context continuity block (Feature A).

Covers the SessionDB ``recent_channel_messages`` query, the ``build_channel_context_block`` helper,
and the turn-path config gate. Behaviour contracts only (relationships between data), never
snapshots of counts or hardcoded platform lists.
"""

import time

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource, build_channel_context_block
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _source(thread_id=None, platform=Platform.SLACK):
    return SessionSource(
        platform=platform,
        chat_id="C123",
        chat_type="thread" if thread_id else "channel",
        user_id="U1",
        thread_id=thread_id,
    )


def _seed(db, *, session_id="sess", source="slack", chat_id="C123", chat_type="channel",
          thread_id=None, messages=()):
    """Create one session and append ``(role, content, ts)`` messages with explicit timestamps."""
    db.create_session(session_id, source=source, chat_id=chat_id, chat_type=chat_type, thread_id=thread_id)
    for role, content, ts in messages:
        db.append_message(session_id, role=role, content=content, timestamp=ts)


NOW = time.time()


# ---------------------------------------------------------------------------
# SessionDB.recent_channel_messages
# ---------------------------------------------------------------------------

class TestRecentChannelMessages:
    def test_only_requested_roles_returned(self, db):
        _seed(db, messages=[
            ("assistant", "agent said hello", NOW - 100),
            ("user", "human said hi", NOW - 90),
        ])
        rows = db.recent_channel_messages("slack", "C123", "channel", None, roles=["assistant"], limit=20)
        assert all(r["role"] == "assistant" for r in rows)
        assert [r["content"] for r in rows] == ["agent said hello"]

    def test_newest_first_ordering(self, db):
        _seed(db, messages=[
            ("assistant", "oldest", NOW - 300),
            ("assistant", "middle", NOW - 200),
            ("assistant", "newest", NOW - 100),
        ])
        rows = db.recent_channel_messages("slack", "C123", "channel", None, roles=["assistant"], limit=20)
        assert [r["content"] for r in rows] == ["newest", "middle", "oldest"]

    def test_scoped_to_channel(self, db):
        _seed(db, session_id="a", chat_id="C1", messages=[("assistant", "in channel one", NOW - 100)])
        _seed(db, session_id="b", chat_id="C2", messages=[("assistant", "in channel two", NOW - 100)])
        rows = db.recent_channel_messages("slack", "C1", "channel", None, roles=["assistant"], limit=20)
        assert [r["content"] for r in rows] == ["in channel one"]

    def test_thread_isolated_from_channel(self, db):
        _seed(db, session_id="chan", chat_id="C1", chat_type="channel", thread_id=None,
              messages=[("assistant", "channel msg", NOW - 100)])
        _seed(db, session_id="thread", chat_id="C1", chat_type="thread", thread_id="T9",
              messages=[("assistant", "thread msg", NOW - 100)])
        thread_rows = db.recent_channel_messages("slack", "C1", "thread", "T9", roles=["assistant"], limit=20)
        assert [r["content"] for r in thread_rows] == ["thread msg"]

    def test_since_ts_bounds_window(self, db):
        _seed(db, messages=[
            ("assistant", "recent", NOW - 100),
            ("assistant", "ancient", NOW - 10 * 86400),
        ])
        rows = db.recent_channel_messages("slack", "C123", "channel", None,
                                          roles=["assistant"], limit=20, since_ts=NOW - 86400)
        assert [r["content"] for r in rows] == ["recent"]


# ---------------------------------------------------------------------------
# build_channel_context_block
# ---------------------------------------------------------------------------

class TestBuildChannelContextBlock:
    def test_none_outside_slack_discord(self, db):
        assert build_channel_context_block(db, _source(platform=Platform.TELEGRAM)) is None

    def test_none_for_fresh_channel(self, db):
        assert build_channel_context_block(db, _source()) is None

    def test_assistant_only_by_default(self, db):
        _seed(db, messages=[
            ("assistant", "agent reply", NOW - 100),
            ("user", "human question", NOW - 90),
        ])
        block = build_channel_context_block(db, _source())
        assert "agent reply" in block
        assert "human question" not in block

    def test_include_other_users_adds_humans(self, db):
        _seed(db, messages=[
            ("assistant", "agent reply", NOW - 100),
            ("user", "human question", NOW - 90),
        ])
        block = build_channel_context_block(db, _source(), include_other_users=True)
        assert "agent reply" in block
        assert "human question" in block

    def test_newest_first_in_block(self, db):
        _seed(db, messages=[
            ("assistant", "oldest", NOW - 300),
            ("assistant", "newest", NOW - 100),
        ])
        block = build_channel_context_block(db, _source())
        assert block.index("newest") < block.index("oldest")

    def test_db_failure_returns_none(self, db):
        class _Boom:
            def recent_channel_messages(self, *a, **kw):
                raise RuntimeError("boom")
        assert build_channel_context_block(_Boom(), _source()) is None

    def test_thread_header(self, db):
        _seed(db, chat_type="thread", thread_id="T9",
              messages=[("assistant", "thread reply", NOW - 100)])
        block = build_channel_context_block(db, _source(thread_id="T9"))
        assert "thread" in block.splitlines()[0]


# ---------------------------------------------------------------------------
# Turn-path config gate
# ---------------------------------------------------------------------------

class TestChannelContextNote:
    @pytest.mark.asyncio
    async def test_disabled_appends_nothing(self, monkeypatch, db):
        monkeypatch.setattr(gateway_run, "_load_gateway_config",
                            lambda: {"continuity": {"channel_context": False}})
        monkeypatch.setattr(gateway_run, "_gateway_session_db_inner", lambda g: db)
        notes = []
        await GatewayTurnMixin()._hmwa_channel_context_note(_source(), notes)
        assert notes == []

    @pytest.mark.asyncio
    async def test_enabled_fresh_channel_appends_nothing(self, monkeypatch, db):
        monkeypatch.setattr(gateway_run, "_load_gateway_config",
                            lambda: {"continuity": {"channel_context": True}})
        monkeypatch.setattr(gateway_run, "_gateway_session_db_inner", lambda g: db)
        notes = []
        await GatewayTurnMixin()._hmwa_channel_context_note(_source(), notes)
        assert notes == []

    @pytest.mark.asyncio
    async def test_enabled_injects_block(self, monkeypatch, db):
        _seed(db, messages=[("assistant", "agent reply", NOW - 100)])
        monkeypatch.setattr(gateway_run, "_load_gateway_config",
                            lambda: {"continuity": {"channel_context": True}})
        monkeypatch.setattr(gateway_run, "_gateway_session_db_inner", lambda g: db)
        notes = []
        await GatewayTurnMixin()._hmwa_channel_context_note(_source(), notes)
        assert len(notes) == 1
        assert "Your recent messages in this channel" in notes[0]
        assert "agent reply" in notes[0]

    @pytest.mark.asyncio
    async def test_db_error_appends_nothing(self, monkeypatch):
        monkeypatch.setattr(gateway_run, "_load_gateway_config",
                            lambda: {"continuity": {"channel_context": True}})
        monkeypatch.setattr(gateway_run, "_gateway_session_db_inner", lambda g: None)
        notes = []
        await GatewayTurnMixin()._hmwa_channel_context_note(_source(), notes)
        assert notes == []
