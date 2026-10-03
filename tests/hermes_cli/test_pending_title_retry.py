"""Regression for #124033: a transient DB-row race after ``/new <title>`` kept
``_pending_title`` forever — the comments claimed it was "kept for retry" but no
code ever retried it, the status bar kept displaying the unsaved title, and the
auto-titler wrote its own over the untitled row. The pending title is now
retried (per turn via ``chat()``) until the row exists."""

from cli import HermesCLI


class _FakeDB:
    def __init__(self):
        self.saved = []

    def set_session_title(self, sid, title):
        self.saved.append((sid, title))


class _TitleAgent:
    def __init__(self, fail_times=0):
        self.session_id = "sess-1"
        self.fail_times = fail_times
        self.ensure_calls = 0
        self._session_db_created = False

    def _ensure_db_session(self):
        self.ensure_calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            self._session_db_created = False
            raise OSError("database is busy")
        self._session_db_created = True


def _cli(fail_times=0):
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = _TitleAgent(fail_times)
    cli.session_id = "sess-1"
    cli._pending_title = "my title"
    cli._session_db = _FakeDB()
    return cli


def test_transient_row_failure_is_retried_on_the_next_turn():
    """The first attempt fails transiently; the next turn applies the title."""
    cli = _cli(fail_times=1)

    cli._apply_pending_title()
    assert cli._pending_title == "my title"

    cli._apply_pending_title()
    assert cli._pending_title is None
    assert cli._session_db.saved == [("sess-1", "my title")]


def test_no_pending_title_is_a_noop():
    """No pending intent: no DB round-trip at all."""
    cli = _cli(fail_times=0)
    cli._pending_title = None

    cli._apply_pending_title()

    assert cli.agent.ensure_calls == 0
    assert cli._session_db.saved == []


def test_absent_session_db_is_a_noop():
    """No session store: nothing to retry against, no crash."""
    cli = _cli(fail_times=0)
    cli._session_db = None

    cli._apply_pending_title()

    assert cli.agent.ensure_calls == 0
    assert cli._pending_title == "my title"
