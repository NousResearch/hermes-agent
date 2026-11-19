"""Session title refusals on SQLite keep the legacy store's validation order and messages."""

import pytest

from persistence.conversation.sessions import NewSession, TitleSource


def test_automatic_title_reports_invalid_source(backend, scope_a):
    sessions = backend.open(scope_a).sessions
    sessions.create(NewSession("title", "cli"))
    with pytest.raises(ValueError, match="^invalid automatic title source: 'user'$"):
        sessions.set_auto_title("title", "automatic", source=TitleSource.USER)
    assert sessions.get_title("title") is None
