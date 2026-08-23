from types import SimpleNamespace
from unittest.mock import MagicMock

from hermes_cli.oneshot import _archive_oneshot_session_if_requested


def test_oneshot_session_archival_is_opt_in(monkeypatch):
    monkeypatch.delenv("HERMES_ONESHOT_ARCHIVE_SESSION", raising=False)
    db = MagicMock()

    assert _archive_oneshot_session_if_requested(SimpleNamespace(session_id="s1"), db) is False
    db.set_session_archived.assert_not_called()


def test_oneshot_session_archival_hides_the_exact_session(monkeypatch):
    monkeypatch.setenv("HERMES_ONESHOT_ARCHIVE_SESSION", "1")
    db = MagicMock()
    db.set_session_archived.return_value = True

    assert _archive_oneshot_session_if_requested(SimpleNamespace(session_id="panel-s1"), db) is True
    db.set_session_archived.assert_called_once_with("panel-s1", True)


def test_oneshot_session_archival_fails_closed_without_identity(monkeypatch):
    monkeypatch.setenv("HERMES_ONESHOT_ARCHIVE_SESSION", "1")
    db = MagicMock()

    assert _archive_oneshot_session_if_requested(SimpleNamespace(session_id=""), db) is False
    db.set_session_archived.assert_not_called()
