"""Tests for the interactive session browser (`hermes sessions browse`).

Covers:
- _session_browse_picker logic (curses mocked, fallback tested)
- cmd_sessions 'browse' action integration
- Argument parser registration
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


from hermes_cli.sessions_cmd_browse import _content_hits, _format_row, _session_browse_picker


# ─── Sample session data ──────────────────────────────────────────────────────

def _make_sessions(n=5):
    """Generate a list of fake rich-session dicts."""
    now = time.time()
    sessions = []
    for i in range(n):
        sessions.append({
            "id": f"20260308_{i:06d}_abcdef",
            "source": "cli" if i % 2 == 0 else "telegram",
            "model": "test/model",
            "title": f"Session {i}" if i % 3 != 0 else None,
            "preview": f"Hello from session {i}",
            "last_active": now - i * 3600,
            "started_at": now - i * 3600 - 60,
            "message_count": (i + 1) * 5,
        })
    return sessions


SAMPLE_SESSIONS = _make_sessions(5)


# ─── _session_browse_picker ──────────────────────────────────────────────────

class TestSessionBrowsePicker:
    """Tests for the _session_browse_picker function."""

    def test_empty_sessions_returns_none(self, capsys):
        result = _session_browse_picker([])
        assert result is None
        assert "No sessions found" in capsys.readouterr().out


    def test_fallback_mode_valid_selection(self):
        """When curses is unavailable, fallback numbered list should work."""
        sessions = _make_sessions(3)

        # Mock curses import to fail, forcing fallback
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "curses":
                raise ImportError("no curses")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with patch("builtins.input", return_value="2"):
                result = _session_browse_picker(sessions)

        assert result == sessions[1]["id"]


    def test_fallback_shows_preview_when_no_title(self, capsys):
        """When no title, show preview."""
        sessions = [{
            "id": "test_002",
            "source": "cli",
            "title": None,
            "preview": "Hello world test message",
            "last_active": time.time(),
        }]

        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "curses":
                raise ImportError("no curses")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with patch("builtins.input", return_value="q"):
                _session_browse_picker(sessions)

        output = capsys.readouterr().out
        assert "Hello world test message" in output


# ─── Curses-based picker (mocked curses) ────────────────────────────────────

@pytest.mark.platforms("linux")
class TestCursesBrowse:
    """Tests for the curses-based interactive picker via simulated key sequences."""

    def _run_with_keys(self, sessions, key_sequence):
        """Simulate running the curses picker with a given key sequence."""

        # Build a mock stdscr that returns keys from the sequence
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (30, 120)
        mock_stdscr.getch.side_effect = key_sequence

        # Capture what curses.wrapper receives and call it with our mock
        with patch("curses.wrapper") as mock_wrapper:
            # When wrapper is called, invoke the function with our mock stdscr
            def run_inner(func):
                try:
                    func(mock_stdscr)
                except StopIteration:
                    pass  # key sequence exhausted

            mock_wrapper.side_effect = run_inner
            with patch("curses.curs_set"):
                with patch("curses.has_colors", return_value=False):
                    return _session_browse_picker(sessions)



    def test_escape_cancels(self):
        sessions = _make_sessions(3)
        result = self._run_with_keys(sessions, [27])  # Esc
        assert result is None


    def test_type_to_filter_then_enter(self):
        """Typing characters filters the list, Enter selects from filtered."""
        sessions = [
            {"id": "s1", "source": "cli", "title": "Alpha project", "preview": "", "last_active": time.time()},
            {"id": "s2", "source": "cli", "title": "Beta project", "preview": "", "last_active": time.time()},
            {"id": "s3", "source": "cli", "title": "Gamma project", "preview": "", "last_active": time.time()},
        ]
        # Type "Beta" then Enter — should select s2
        keys = [ord(c) for c in "Beta"] + [10]
        result = self._run_with_keys(sessions, keys)
        assert result == "s2"








# ─── Argument parser registration ──────────────────────────────────────────



# ─── Integration: cmd_sessions browse action ────────────────────────────────



# ─── Message-content search ─────────────────────────────────────────────────

def _hit(session_id=None, role=None, snippet=None):
    """One ``SearchHit``-shaped row (the picker reads attributes, not keys)."""
    return SimpleNamespace(message_id="m1", session_id=session_id, role=role, snippet=snippet)


class TestContentHits:
    """`_content_hits` — FTS5 over message text, the half `_match` cannot cover."""

    def _store(self, rows):
        store = MagicMock()
        store.search.search_messages.return_value = tuple(rows)
        return store

    def test_prefixes_tokens_and_dedupes_sessions(self):
        store = self._store([
            _hit("s9", "assistant", "tool json mentioning >>>brave<<<"),
            _hit("s9", "user", "please fix >>>brave<<< autocomplete"),
            _hit("s4", "assistant", "unrelated"),
        ])

        hits = _content_hits(store, "Brave autocomplete")

        # user excerpt wins over the higher-ranked assistant one; one entry per session
        assert hits == {"s9": "please fix brave autocomplete", "s4": "unrelated"}
        kwargs = store.search.search_messages.call_args.kwargs
        assert kwargs["query"] == "Brave* autocomplete*"
        assert kwargs["role_filter"] == ["user", "assistant"]
        assert "role" in kwargs["fields"]

    def test_quoted_phrase_keeps_its_shape(self):
        store = self._store([])

        _content_hits(store, '"session browse"')

        assert store.search.search_messages.call_args.kwargs["query"] == '"session browse"'

    def test_boolean_operators_are_not_prefix_wildcarded(self):
        store = self._store([])

        _content_hits(store, "brave OR firefox NOT edge")

        assert store.search.search_messages.call_args.kwargs["query"] == "brave* OR firefox* NOT edge*"

    def test_short_query_skips_the_index(self):
        store = self._store([])
        assert _content_hits(store, "b") == {}
        store.search.search_messages.assert_not_called()

    def test_no_store_or_store_error_degrades_to_empty(self):
        assert _content_hits(None, "brave") == {}
        store = MagicMock()
        store.search.search_messages.side_effect = RuntimeError("search index unavailable")
        assert _content_hits(store, "brave") == {}


class TestContentSearchInPicker:
    """A session whose TITLE says nothing about the topic must still be findable by typing."""

    def _run_with_keys(self, sessions, key_sequence, store):
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (30, 120)
        mock_stdscr.getch.side_effect = key_sequence
        with patch("curses.wrapper") as mock_wrapper:
            def run_inner(func):
                try:
                    func(mock_stdscr)
                except StopIteration:
                    pass  # key sequence exhausted
            mock_wrapper.side_effect = run_inner
            with patch("curses.curs_set"):
                with patch("curses.has_colors", return_value=False):
                    return _session_browse_picker(sessions, store=store)

    def _store(self, rows, records=None):
        store = MagicMock()
        store.sessions.lifecycle_statuses.return_value = {}
        store.search.search_messages.return_value = tuple(rows)
        records = records or {}
        store.sessions.get.side_effect = lambda sid: records.get(sid)
        return store

    @staticmethod
    def _record(session_id, title, started_at, last_activity_at=None):
        return SimpleNamespace(
            session_id=session_id, source="cli", title=title, message_count=12,
            started_at=started_at, ended_at=None, last_activity_at=last_activity_at)

    def test_message_only_match_is_listed_and_resumable(self):
        sessions = [
            {"id": "s1", "source": "cli", "title": "Weekly prep", "preview": "", "last_active": time.time()},
        ]
        now = time.time()
        store = self._store(
            [_hit("s7", "user", "the >>>kubernetes<<< ingress broke again")],
            {"s7": self._record("s7", "Notes", now - 7200, now - 3600)},
        )

        result = self._run_with_keys(sessions, [ord(c) for c in "kubernetes"] + [10], store)

        assert result == "s7"

    def test_content_rows_are_marked_and_snippet_shown(self):
        sessions = [{"id": "s1", "source": "cli", "title": "Weekly prep", "preview": "", "last_active": time.time()}]
        store = self._store([_hit("s1", "user", "kubernetes ingress broke")])
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (30, 120)
        mock_stdscr.getch.side_effect = [ord(c) for c in "kubernetes"] + [27, 27]
        with patch("curses.wrapper") as mock_wrapper:
            def run_inner(func):
                try:
                    func(mock_stdscr)
                except StopIteration:
                    pass
            mock_wrapper.side_effect = run_inner
            with patch("curses.curs_set"):
                with patch("curses.has_colors", return_value=False):
                    _session_browse_picker(sessions, store=store)

        drawn = " ".join(str(c.args[2]) for c in mock_stdscr.addnstr.call_args_list)
        assert "* Weekly prep" in drawn
        assert "match: kubernetes ingress broke" in drawn

    def test_title_match_is_not_starred(self):
        """A row the title already explains stays unmarked, even when the text matches too."""
        sessions = [{"id": "s1", "source": "cli", "title": "Weekly prep", "preview": "", "last_active": time.time()}]
        store = self._store([_hit("s1", "user", "weekly prep agenda")])
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (30, 120)
        mock_stdscr.getch.side_effect = [ord(c) for c in "weekly"] + [27, 27]
        with patch("curses.wrapper") as mock_wrapper:
            def run_inner(func):
                try:
                    func(mock_stdscr)
                except StopIteration:
                    pass
            mock_wrapper.side_effect = run_inner
            with patch("curses.curs_set"):
                with patch("curses.has_colors", return_value=False):
                    _session_browse_picker(sessions, store=store)

        drawn = " ".join(str(c.args[2]) for c in mock_stdscr.addnstr.call_args_list)
        assert "Weekly prep" in drawn
        assert "* Weekly prep" not in drawn


def test_format_row_marks_message_matches():
    row = {"id": "abcdef", "source": "cli", "title": "Weekly prep", "preview": "",
           "message_count": 3, "last_active": time.time()}
    plain = _format_row(dict(row), 120)
    marked = _format_row({**row, "_hit": "excerpt"}, 120)
    assert "Weekly prep" in plain and not plain.startswith("*")
    assert marked.lstrip().startswith("* Weekly prep")


# ─── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case handling for the session browser."""


    def test_relative_time_formatting(self, capsys):
        """Verify various time deltas format correctly."""
        now = time.time()
        sessions = [
            {"id": "recent", "source": "cli", "title": None, "preview": "just now test", "last_active": now},
            {"id": "hour_ago", "source": "cli", "title": None, "preview": "hour ago test", "last_active": now - 7200},
            {"id": "days_ago", "source": "cli", "title": None, "preview": "days ago test", "last_active": now - 259200},
        ]

        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "curses":
                raise ImportError("no curses")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with patch("builtins.input", return_value="q"):
                _session_browse_picker(sessions)

        output = capsys.readouterr().out
        assert "just now" in output
        assert "2h ago" in output
        assert "3d ago" in output
