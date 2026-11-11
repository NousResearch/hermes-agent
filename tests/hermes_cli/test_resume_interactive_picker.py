"""Interactive ``--resume`` picker: optional flag argument, picker triggers, /sessions completions."""

import contextlib
import os
import sys
from types import SimpleNamespace

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from hermes_cli.commands_completion import SlashCommandCompleter
from hermes_cli.main_resume_args import resolve_interactive_resume, resolve_session_name_or_search


def _parse(argv):
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat = build_top_level_parser()
    return parser.parse_args(argv)


@pytest.mark.parametrize("argv", [["-r"], ["--resume"], ["chat", "--resume"], ["chat", "-r"]])
def test_bare_resume_flag_parses_to_true(argv):
    assert _parse(argv).resume is True


@pytest.mark.parametrize("argv", [["-r", "abc123"], ["chat", "--resume", "abc123"]])
def test_resume_flag_keeps_its_value(argv):
    assert _parse(argv).resume == "abc123"


@pytest.mark.parametrize("argv", [[], ["chat"]])
def test_no_resume_flag_stays_unset(argv):
    assert getattr(_parse(argv), "resume", None) is None


@pytest.fixture
def fake_session_db(monkeypatch):
    db = object()

    @contextlib.contextmanager
    def _session_db():
        yield db

    # A stub module keeps the real hermes_cli.main (and its startup side effects) out of the test.
    monkeypatch.setitem(sys.modules, "hermes_cli.main", SimpleNamespace(_session_db=_session_db))
    return db


@pytest.fixture
def picker(monkeypatch):
    """Replace the picker; ``picker.answer`` is what it returns, ``picker.calls`` records kwargs."""
    state = SimpleNamespace(answer=None, calls=[])

    def _pick(session_db, **kwargs):
        state.calls.append((session_db, kwargs))
        return state.answer

    monkeypatch.setattr("hermes_cli.sessions_cmd_browse.pick_interactive_session", _pick)
    return state


def _tty(monkeypatch, is_tty):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: is_tty))


class TestResolveInteractiveResume:
    def test_tty_picks_a_session(self, monkeypatch, fake_session_db, picker):
        _tty(monkeypatch, True)
        picker.answer = "sess-1"
        args = SimpleNamespace(resume=True)

        resolve_interactive_resume(args, use_tui=False)

        assert args.resume == "sess-1"
        assert picker.calls == [(fake_session_db, {"exclude_sources": ["tool"]})]

    def test_cancel_exits_zero(self, monkeypatch, fake_session_db, picker):
        _tty(monkeypatch, True)
        picker.answer = None

        with pytest.raises(SystemExit) as exc:
            resolve_interactive_resume(SimpleNamespace(resume=True), use_tui=False)

        assert exc.value.code == 0

    def test_non_tty_errors_without_opening_picker(self, monkeypatch, capsys, fake_session_db, picker):
        _tty(monkeypatch, False)

        with pytest.raises(SystemExit) as exc:
            resolve_interactive_resume(SimpleNamespace(resume=True), use_tui=False)

        assert exc.value.code == 1
        assert "requires a TTY" in capsys.readouterr().err
        assert picker.calls == []

    def test_tui_requests_session_overlay_and_clears_resume(self, monkeypatch, picker):
        monkeypatch.delenv("HERMES_TUI_OPEN_SESSIONS", raising=False)
        args = SimpleNamespace(resume=True)

        resolve_interactive_resume(args, use_tui=True)

        assert os.environ["HERMES_TUI_OPEN_SESSIONS"] == "1"
        assert args.resume is None
        assert picker.calls == []

    @pytest.mark.parametrize("value", [None, "abc123"])
    def test_explicit_or_absent_resume_is_untouched(self, picker, value):
        args = SimpleNamespace(resume=value)

        resolve_interactive_resume(args, use_tui=False)

        assert args.resume == value
        assert picker.calls == []


class TestResolveSessionNameOrSearch:
    def test_exact_match_skips_picker(self, monkeypatch, fake_session_db, picker):
        _tty(monkeypatch, True)
        args = SimpleNamespace(resume="my title")

        resolve_session_name_or_search(args, "my title", use_tui=False, resolve_fn=lambda v: "sess-9")

        assert args.resume == "sess-9"
        assert picker.calls == []

    def test_no_match_on_tty_opens_picker_with_query(self, monkeypatch, fake_session_db, picker):
        _tty(monkeypatch, True)
        picker.answer = "sess-2"
        args = SimpleNamespace(resume="disc")

        resolve_session_name_or_search(args, "disc", use_tui=False, resolve_fn=lambda v: None)

        assert args.resume == "sess-2"
        assert picker.calls == [(fake_session_db, {"exclude_sources": ["tool"], "initial_search": "disc"})]

    def test_no_match_picker_cancel_exits_zero(self, monkeypatch, fake_session_db, picker):
        _tty(monkeypatch, True)
        picker.answer = None

        with pytest.raises(SystemExit) as exc:
            resolve_session_name_or_search(SimpleNamespace(resume="x"), "x", use_tui=False, resolve_fn=lambda v: None)

        assert exc.value.code == 0

    @pytest.mark.parametrize("is_tty,use_tui", [(False, False), (True, True)])
    def test_no_match_without_picker_passes_raw_value(self, monkeypatch, fake_session_db, picker, is_tty, use_tui):
        _tty(monkeypatch, is_tty)
        args = SimpleNamespace(resume="whatever")

        resolve_session_name_or_search(args, "whatever", use_tui=use_tui, resolve_fn=lambda v: None)

        assert args.resume == "whatever"
        assert picker.calls == []


class TestPickInteractiveSession:
    def _stub(self, monkeypatch, sessions):
        seen = {}

        class Reader:
            def list_sessions_rich(self, **kwargs):
                seen["list_kwargs"] = kwargs
                return sessions

        def browse(listed, session_db=None, initial_search=""):
            seen["browse"] = (listed, session_db, initial_search)
            return "picked"

        monkeypatch.setattr("hermes_cli.sessions_cmd._surface_reader", lambda db: Reader())
        monkeypatch.setattr("hermes_cli.sessions_cmd_browse._session_browse_picker", browse)
        return seen

    def test_forwards_initial_search_to_browser(self, monkeypatch):
        from hermes_cli.sessions_cmd_browse import pick_interactive_session

        db = object()
        sessions = [{"id": "s1"}]
        seen = self._stub(monkeypatch, sessions)

        result = pick_interactive_session(db, exclude_sources=["tool"], initial_search="bot")

        assert result == "picked"
        assert seen["browse"] == (sessions, db, "bot")
        assert seen["list_kwargs"] == {"exclude_sources": ["tool"], "limit": 500}

    def test_no_sessions_returns_none_without_browser(self, monkeypatch):
        from hermes_cli.sessions_cmd_browse import pick_interactive_session

        seen = self._stub(monkeypatch, [])

        assert pick_interactive_session(object()) is None
        assert "browse" not in seen

    def test_no_db_returns_none(self):
        from hermes_cli.sessions_cmd_browse import pick_interactive_session

        assert pick_interactive_session(None) is None


class TestSessionsCompletions:
    @pytest.fixture(autouse=True)
    def fake_store(self, monkeypatch):
        sessions = [
            {"id": "s1", "title": "Build Discord Bot", "last_active": 100, "message_count": 5},
            {"id": "s2", "title": "Fix", "last_active": 200, "message_count": 2},
            {"id": "s3", "title": "", "preview": "hello world", "last_active": 300, "message_count": 1},
        ]

        class FakeDB:
            def session_lifecycle_statuses(self, ids):
                return {}

            def close(self):
                pass

        class FakeReader:
            def list_sessions_rich(self, limit=25):
                return sessions

        path = type("P", (), {"exists": lambda self: True, "__truediv__": lambda self, x: self})()
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: path)
        monkeypatch.setattr("hermes_state.SessionDB", lambda db_path, read_only=True: FakeDB())
        monkeypatch.setattr("hermes_cli.sessions_cmd._surface_reader", lambda db: FakeReader())

    @staticmethod
    def _complete(text):
        return list(SlashCommandCompleter().get_completions(Document(text=text), CompleteEvent(completion_requested=True)))

    def test_sessions_lists_the_same_as_resume(self):
        sessions = [(c.text, c.display_text) for c in self._complete("/sessions ")]
        resume = [(c.text, c.display_text) for c in self._complete("/resume ")]

        assert sessions == resume
        assert len(sessions) == 3

    def test_title_with_spaces_is_quoted_on_insert(self):
        texts = {c.text for c in self._complete("/sessions ")}

        assert '"Build Discord Bot"' in texts
        assert "Fix" in texts

    def test_untitled_session_inserts_its_id(self):
        by_display = {c.display_text: c.text for c in self._complete("/sessions ")}

        assert by_display["hello world"] == "s3"

    def test_filter_matches_title_and_id(self):
        assert [c.text for c in self._complete("/sessions discord")] == ['"Build Discord Bot"']
        assert [c.text for c in self._complete("/sessions s2")] == ["Fix"]

    def test_already_quoted_query_does_not_double_quote(self):
        assert [c.text for c in self._complete('/sessions "Build')] == ["Build Discord Bot"]
