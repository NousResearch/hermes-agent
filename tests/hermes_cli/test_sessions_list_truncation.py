"""Regression coverage for the ``hermes sessions list`` result cap."""

from argparse import Namespace

from hermes_cli import sessions_cmd


class _SessionDB:
    def __init__(self, sessions):
        self.sessions = sessions
        self.calls = []

    def list_sessions_rich(self, **kwargs):
        self.calls.append(kwargs)
        return self.sessions


def _session(session_id):
    return {
        "id": session_id,
        "source": "cli",
        "title": session_id,
        "preview": "",
        "last_active": 0,
    }


def test_list_does_not_print_footer_when_limit_is_unbounded(monkeypatch, capsys):
    db = _SessionDB([_session("one"), _session("two")])
    monkeypatch.setattr(sessions_cmd, "_relative_time", lambda *_args, **_kwargs: "now")

    sessions_cmd._cmd_list(db, Namespace(limit=0, source="cli", workspace=None))

    assert db.calls == [{"source": "cli", "exclude_sources": None, "limit": 0}]
    assert "more (use --limit" not in capsys.readouterr().out


def test_list_reports_a_truncated_result_set(monkeypatch, capsys):
    db = _SessionDB([_session("one"), _session("two"), _session("three")])
    monkeypatch.setattr(sessions_cmd, "_relative_time", lambda *_args, **_kwargs: "now")

    sessions_cmd._cmd_list(db, Namespace(limit=2, source="cli", workspace=None))

    assert db.calls == [{"source": "cli", "exclude_sources": None, "limit": 3}]
    output = capsys.readouterr().out
    assert "one" in output and "two" in output
    assert "three" not in output
    assert "… at least 1 more (use --limit 0 or --query to filter)" in output
