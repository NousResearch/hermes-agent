"""Gateway /sessions and /resume must number the same sessions in the same order.

Regression coverage for #146798: ``/resume <number>`` used a different sort
order (and different filters and pagination) than ``/sessions``, so the number
shown by ``/sessions`` could resume a different session. The contract is one
numbered listing shared by both commands: same query, same lineage-activity
sort, same titled-or-current filter, same over-fetch-then-filter pagination,
same visibility. A number shown by one command resolves to the same session
in the other.

Scope boundaries documented here, not bugs: ``/sessions full`` is a wider view
whose extra unnamed rows are not number-addressable from ``/resume`` (which
has no ``full`` flag); admin widenings keep their existing per-command scope
(``/resume --all`` is platform-scoped, ``/sessions all`` is global); and
numbers bind at pick time, so activity between listing and pick moves the
pick (a stated residual race, not a snapshot).
"""

import re
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, build_session_key


def _make_event(text="/resume", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _session_key_for_event(event):
    return build_session_key(event.source)


def _make_runner(session_db=None, current_session_id="current_session_001",
                 event=None):
    """Bare GatewayRunner with a mock session_store and optional session_db."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = SimpleNamespace(platforms={})
    runner._voice_mode = {}
    if session_db is not None:
        from hermes_state import AsyncSessionDB
        session_db = AsyncSessionDB(session_db)
    runner._session_db = session_db
    runner._running_agents = {}
    runner._is_user_authorized = lambda _source: True
    session_key = build_session_key(event.source) if event else "agent:main:telegram:dm"
    mock_session_entry = MagicMock()
    mock_session_entry.session_id = current_session_id
    mock_session_entry.session_key = session_key
    mock_store = MagicMock()
    mock_store.get_or_create_session.return_value = mock_session_entry
    mock_store.load_transcript.return_value = []
    mock_store.switch_session.return_value = mock_session_entry
    runner.session_store = mock_store
    return runner


def _seed(db, lane_key, specs):
    """Create titled sessions with fixed started/active timestamps.

    ``specs`` is a list of ``(sid, title, started, active)``; ``title=None``
    leaves the session untitled. ``user_id``/``chat_id`` default to the lane
    caller so the rows pass the IDOR visibility proof; pass ``user_id`` via
    a 5-tuple to build a row the caller must NOT see.
    """
    for spec in specs:
        sid, title, started, active = spec[:4]
        user_id = spec[4] if len(spec) > 4 else "12345"
        chat_id = spec[5] if len(spec) > 5 else "67890"
        source = spec[6] if len(spec) > 6 else "telegram"
        db.create_session(sid, source, session_key=lane_key,
                          user_id=user_id, chat_id=chat_id)
        if title is not None:
            db.set_session_title(sid, title)
        db._conn.execute(
            "UPDATE sessions SET started_at=?, last_activity_at=? WHERE id=?",
            (started, active, sid))
    db._conn.commit()


def _sessions_ids(result):
    """Session ids in display order from a /sessions reply."""
    return re.findall(r"— `([^`]+)`", result)


def _numbered_lines(result):
    """Numbered listing lines (``N. ...``) in order."""
    return [line for line in result.splitlines()
            if re.match(r"^\d+\. ", line.strip())]


def _current_position(result):
    """Index of the ``(current)`` row among the numbered lines."""
    for idx, line in enumerate(_numbered_lines(result)):
        if "(current)" in line:
            return idx
    return None


async def _resume_target(runner, event):
    """Session id ``/resume N`` switches to, or None when it does not switch."""
    await runner._handle_resume_command(event)
    calls = runner.session_store.switch_session.call_args_list
    if not calls:
        return None
    return calls[-1][0][1]


T0 = 1_700_000_000.0
DAY = 86400.0


class TestNumberingAgreement:
    """The primary #146798 contract: /sessions row N == /resume N target."""

    @pytest.mark.asyncio
    async def test_crossed_timestamps_agree(self, tmp_path):
        """Started-earlier/active-later vs started-later/active-earlier."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions")
        lane_key = _session_key_for_event(event)
        _seed(db, lane_key, [
            ("old_active", "Old But Active", T0, T0 + 4 * DAY),
            ("new_idle", "Newer But Idle", T0 + DAY, T0 + DAY + 60),
            ("current_session_001", "Current Work", T0 - DAY, T0 - DAY),
        ])
        runner = _make_runner(session_db=db, event=event)
        sessions = await runner._handle_sessions_command(event)
        assert _sessions_ids(sessions) == ["old_active", "new_idle", "current_session_001"]
        target = await _resume_target(runner, _make_event(text="/resume 1"))
        assert target == "old_active"
        assert target == _sessions_ids(sessions)[0]
        db.close()

    @pytest.mark.asyncio
    async def test_every_row_agrees(self, tmp_path):
        """Five sessions whose started order is the reverse of activity order."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions")
        lane_key = _session_key_for_event(event)
        specs = [
            (f"s{i}", f"Work {i}", T0 + (5 - i) * DAY, T0 + i * DAY)
            for i in range(5)
        ]
        _seed(db, lane_key, specs + [
            ("current_session_001", "Current Work", T0 - DAY, T0 - DAY),
        ])
        runner = _make_runner(session_db=db, event=event)
        sessions = await runner._handle_sessions_command(event)
        expected = [f"s{i}" for i in reversed(range(5))] + ["current_session_001"]
        assert _sessions_ids(sessions) == expected
        for position, sid in enumerate(expected[:5], start=1):
            runner.session_store.switch_session.reset_mock()
            target = await _resume_target(runner, _make_event(text=f"/resume {position}"))
            assert target == sid, f"position {position} resolved to {target}, want {sid}"
        # The current row resolves to itself, so picking it is already-on, not a switch.
        runner.session_store.switch_session.reset_mock()
        await runner._handle_resume_command(_make_event(text="/resume 6"))
        assert runner.session_store.switch_session.call_args_list == []
        db.close()

    @pytest.mark.asyncio
    async def test_untitled_current_numbered_in_both(self, tmp_path):
        """An untitled current session is a numbered row in both listings."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions")
        lane_key = _session_key_for_event(event)
        _seed(db, lane_key, [
            ("titled_a", "Titled A", T0, T0),
            ("current_session_001", None, T0 + 2 * DAY, T0 + 2 * DAY),
        ])
        runner = _make_runner(session_db=db, event=event)
        sessions = await runner._handle_sessions_command(event)
        resume_list = await runner._handle_resume_command(_make_event(text="/resume"))
        assert _current_position(sessions) == 0
        assert _current_position(resume_list) == 0
        assert "— (current)" in resume_list
        runner.session_store.switch_session.reset_mock()
        await runner._handle_resume_command(_make_event(text="/resume 1"))
        assert runner.session_store.switch_session.call_args_list == []
        db.close()

    @pytest.mark.asyncio
    async def test_titled_current_marked_in_both(self, tmp_path):
        """A titled current session carries the marker in both listings."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions")
        lane_key = _session_key_for_event(event)
        _seed(db, lane_key, [
            ("titled_a", "Titled A", T0, T0),
            ("current_session_001", "Current Work", T0 + 2 * DAY, T0 + 2 * DAY),
        ])
        runner = _make_runner(session_db=db, event=event)
        sessions = await runner._handle_sessions_command(event)
        resume_list = await runner._handle_resume_command(_make_event(text="/resume"))
        assert _current_position(sessions) == 0
        assert _current_position(resume_list) == 0
        assert "**Current Work** (current)" in sessions
        db.close()


class TestPaginationAndFiltering:
    """Over-fetch-then-filter pagination: non-visible rows must not eat page slots."""

    @pytest.mark.asyncio
    async def test_untitled_rows_do_not_consume_resume_page(self, tmp_path):
        """Five newest untitled rows must not shrink the /resume page to five."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions")
        lane_key = _session_key_for_event(event)
        titled = [(f"t{i}", f"Titled {i}", T0 + i, T0 + i) for i in range(12)]
        untitled = [(f"u{i}", None, T0 + 100 + i, T0 + 100 + i) for i in range(5)]
        _seed(db, lane_key, titled + untitled + [
            ("current_session_001", "Current Work", T0 - DAY, T0 - DAY),
        ])
        runner = _make_runner(session_db=db, event=event)
        sessions = await runner._handle_sessions_command(event)
        resume_list = await runner._handle_resume_command(_make_event(text="/resume"))
        assert len(_numbered_lines(resume_list)) == 10
        assert _sessions_ids(sessions) == [f"t{i}" for i in reversed(range(2, 12))]
        shown_titles = set(re.findall(r"\*\*(.+?)\*\*", resume_list))
        assert "Titled 0" not in shown_titles
        assert "Titled 1" not in shown_titles
        db.close()

    @pytest.mark.asyncio
    async def test_invisible_rows_do_not_consume_resume_page(self, tmp_path):
        """Rows another user owns (same lane key, failing the IDOR proof) must
        not shrink the page either."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions")
        lane_key = _session_key_for_event(event)
        titled = [(f"t{i}", f"Titled {i}", T0 + i, T0 + i) for i in range(12)]
        hidden = [(f"h{i}", f"Hidden {i}", T0 + 100 + i, T0 + 100 + i, "other-user")
                  for i in range(5)]
        _seed(db, lane_key, titled + hidden + [
            ("current_session_001", "Current Work", T0 - DAY, T0 - DAY),
        ])
        runner = _make_runner(session_db=db, event=event)
        sessions = await runner._handle_sessions_command(event)
        resume_list = await runner._handle_resume_command(_make_event(text="/resume"))
        assert len(_numbered_lines(resume_list)) == 10
        assert "Hidden" not in resume_list
        assert "Hidden" not in sessions
        db.close()

    @pytest.mark.asyncio
    async def test_truncation_beyond_ten(self, tmp_path):
        """Twelve titled rows: both listings show the same first ten, the
        tenth resolves, and the eleventh is out of range (numbering is a
        10-row page, not a window into the whole store)."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions")
        lane_key = _session_key_for_event(event)
        _seed(db, lane_key, [
            (f"t{i}", f"Titled {i}", T0 + i, T0 + i) for i in range(12)
        ] + [("current_session_001", "Current Work", T0 - DAY, T0 - DAY)])
        runner = _make_runner(session_db=db, event=event)
        sessions = await runner._handle_sessions_command(event)
        assert _sessions_ids(sessions) == [f"t{i}" for i in reversed(range(2, 12))]
        target = await _resume_target(runner, _make_event(text="/resume 10"))
        assert target == "t2"
        runner.session_store.switch_session.reset_mock()
        await runner._handle_resume_command(_make_event(text="/resume 11"))
        assert runner.session_store.switch_session.call_args_list == []
        db.close()

    @pytest.mark.asyncio
    async def test_numbered_candidates_not_truncated_before_visibility(self, tmp_path):
        """The shared helper returns every candidate, so the visibility filter
        (not the page limit) decides what the caller sees."""
        from hermes_cli import session_listing
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        lane_key = "agent:main:telegram:dm"
        _seed(db, lane_key, [
            (f"t{i}", f"Titled {i}", T0 + i, T0 + i) for i in range(12)
        ])
        rows = session_listing.query_numbered_session_listing(
            db, source="telegram", session_key=lane_key, limit=10)
        assert len(rows) == 12
        assert [row["id"] for row in rows] == [f"t{i}" for i in reversed(range(12))]
        db.close()


class TestPreservedBehaviour:
    """Inverse probes: what the fix must NOT change."""

    @pytest.mark.asyncio
    async def test_search_mode_ordering_unchanged(self, tmp_path):
        """Search still orders by activity and still matches ids of unnamed rows."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions search Work")
        lane_key = _session_key_for_event(event)
        _seed(db, lane_key, [
            ("alpha", "Alpha Work", T0, T0 + 4 * DAY),
            ("beta", "Beta Work", T0 + DAY, T0 + DAY + 60),
            ("current_session_001", "Current Work", T0 - DAY, T0 - DAY),
        ])
        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_sessions_command(event)
        assert _sessions_ids(result)[:2] == ["alpha", "beta"]
        db.close()

    @pytest.mark.asyncio
    async def test_full_mode_superset_is_explicit(self, tmp_path):
        """/sessions full shows unnamed rows; /resume does not number them."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions full")
        lane_key = _session_key_for_event(event)
        _seed(db, lane_key, [
            ("titled_a", "Titled A", T0, T0),
            ("untitled_a", None, T0 + DAY, T0 + DAY),
            ("current_session_001", "Current Work", T0 - DAY, T0 - DAY),
        ])
        runner = _make_runner(session_db=db, event=event)
        full = await runner._handle_sessions_command(event)
        assert "`untitled_a`" in full
        resume_list = await runner._handle_resume_command(_make_event(text="/resume"))
        assert "Titled A" in resume_list
        assert len(_numbered_lines(resume_list)) == 2
        db.close()

    @pytest.mark.asyncio
    async def test_admin_widening_scopes_preserved(self, tmp_path):
        """/resume --all stays platform-scoped; /sessions all stays global."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/resume --all")
        lane_key = _session_key_for_event(event)
        _seed(db, lane_key, [("tg_named", "Telegram Work", T0, T0)])
        db.create_session("discord_named", "discord",
                          session_key="agent:main:discord:dm:other",
                          user_id="other-user", chat_id="other")
        db.set_session_title("discord_named", "Discord Work")
        db._conn.commit()
        runner = _make_runner(session_db=db, event=event)
        runner._resume_caller_is_admin = lambda _source: True
        resume_list = await runner._handle_resume_command(event)
        assert "Telegram Work" in resume_list
        assert "Discord Work" not in resume_list
        sessions_all = await runner._handle_sessions_command(
            _make_event(text="/sessions all"))
        assert "Telegram Work" in sessions_all
        assert "Discord Work" in sessions_all
        db.close()

    @pytest.mark.asyncio
    async def test_lineage_projection_in_both(self, tmp_path):
        """A compressed root projects onto its tip in both listings, ranked by
        tip activity, and the root itself stays out of both."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions")
        lane_key = _session_key_for_event(event)
        _seed(db, lane_key, [
            ("plain", "Plain Work", T0 + DAY, T0 + DAY),
            ("current_session_001", "Current Work", T0 - DAY, T0 - DAY),
        ])
        db.create_session("old_root", "telegram", session_key=lane_key,
                          user_id="12345", chat_id="67890")
        db.set_session_title("old_root", "Root Work")
        db.end_session("old_root", "compression")
        db.create_session("tip", "telegram", session_key=lane_key,
                          user_id="12345", chat_id="67890",
                          parent_session_id="old_root")
        db.set_session_title("tip", "Tip Work")
        db._conn.execute(
            "UPDATE sessions SET started_at=?, last_activity_at=? WHERE id=?",
            (T0 - 2 * DAY, T0 - 2 * DAY, "old_root"))
        db._conn.execute(
            "UPDATE sessions SET started_at=?, last_activity_at=? WHERE id=?",
            (T0, T0 + 4 * DAY, "tip"))
        db._conn.commit()
        runner = _make_runner(session_db=db, event=event)
        sessions = await runner._handle_sessions_command(event)
        resume_list = await runner._handle_resume_command(_make_event(text="/resume"))
        assert "Tip Work" in sessions and "Root Work" not in sessions
        assert "Tip Work" in resume_list and "Root Work" not in resume_list
        assert sessions.index("Tip Work") < sessions.index("Plain Work")
        assert resume_list.index("Tip Work") < resume_list.index("Plain Work")
        db.close()

    @pytest.mark.asyncio
    async def test_background_mutation_moves_pick(self, tmp_path):
        """Numbers bind at pick time: activity between listing and pick moves
        the pick. This documents the stated residual race, it is not a
        snapshot: a snapshot would answer from a list the user no longer sees."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions")
        lane_key = _session_key_for_event(event)
        _seed(db, lane_key, [
            ("first", "First Work", T0, T0 + 2 * DAY),
            ("second", "Second Work", T0 + DAY, T0 + DAY),
            ("current_session_001", "Current Work", T0 - DAY, T0 - DAY),
        ])
        runner = _make_runner(session_db=db, event=event)
        sessions = await runner._handle_sessions_command(event)
        assert _sessions_ids(sessions)[0] == "first"
        db._conn.execute(
            "UPDATE sessions SET last_activity_at=? WHERE id=?",
            (T0 + 9 * DAY, "second"))
        db._conn.commit()
        target = await _resume_target(runner, _make_event(text="/resume 1"))
        assert target == "second"
        db.close()

    @pytest.mark.asyncio
    async def test_out_of_range_and_id_title_paths(self, tmp_path):
        """Out-of-range stays an error; id and title resolution are untouched."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions")
        lane_key = _session_key_for_event(event)
        _seed(db, lane_key, [
            ("titled_a", "Titled A", T0, T0),
            ("titled_b", "Titled B", T0 + DAY, T0 + DAY),
            ("current_session_001", "Current Work", T0 - DAY, T0 - DAY),
        ])
        runner = _make_runner(session_db=db, event=event)
        runner.session_store.switch_session.reset_mock()
        await runner._handle_resume_command(_make_event(text="/resume 99"))
        assert runner.session_store.switch_session.call_args_list == []
        target = await _resume_target(runner, _make_event(text="/resume titled_a"))
        assert target == "titled_a"
        target = await _resume_target(runner, _make_event(text="/resume Titled B"))
        assert target == "titled_b"
        db.close()

    @pytest.mark.asyncio
    async def test_tool_source_excluded_from_global_sessions(self, tmp_path):
        """Tool-source rows never reach the global /sessions all listing."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        event = _make_event(text="/sessions all")
        lane_key = _session_key_for_event(event)
        _seed(db, lane_key, [
            ("titled_a", "Titled A", T0, T0),
            ("tool_row", "Tool Work", T0 + DAY, T0 + DAY, "12345", "67890", "tool"),
            ("current_session_001", "Current Work", T0 - DAY, T0 - DAY),
        ])
        runner = _make_runner(session_db=db, event=event)
        runner._resume_caller_is_admin = lambda _source: True
        result = await runner._handle_sessions_command(event)
        assert "Tool Work" not in result
        assert "Titled A" in result
        db.close()
