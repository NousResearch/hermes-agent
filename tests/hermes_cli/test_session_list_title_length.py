"""Behavioral test: sessions list renders full session titles (#14082).

Valid titles can be up to SessionDB.MAX_TITLE_LENGTH (100) characters, and
``hermes --resume <title>`` requires an exact title match.  Truncating titles
in the list display means copy-pasted titles fail to resolve.  The list must
render the complete title, not a truncated prefix.
"""
import sys
from types import SimpleNamespace


def _make_args():
    return SimpleNamespace(
        sessions_action="list",
        source=None,
        limit=20,
        workspace=None,
    )


class _FakeDB:
    """Minimal SessionDB stub for the list path."""

    def __init__(self, sessions):
        self._sessions = sessions

    def list_sessions_rich(self, **kw):
        return list(self._sessions)

    def close(self):
        pass


def test_sessions_list_renders_full_long_title(monkeypatch, capsys):
    """A title longer than 50 chars must appear untruncated in the output."""
    import hermes_cli.main as main_mod
    import hermes_state

    long_title = "A" * 80  # 80 chars — well above the old 30/50 caps
    sessions = [
        {
            "id": "20260706_123456_abcd1234",
            "title": long_title,
            "preview": "hello",
            "source": "cli",
            "last_active": 1_800_000_000,
        }
    ]

    monkeypatch.setattr(
        hermes_state, "SessionDB", lambda: _FakeDB(sessions)
    )
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "list"])

    main_mod.main()

    out = capsys.readouterr().out
    assert long_title in out, "Full 80-char title must appear untruncated"
