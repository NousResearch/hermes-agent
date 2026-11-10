"""A title transferred to a compression continuation on SQLite leaves the ancestor's stored provenance intact."""

from persistence.conversation.sessions import NewSession, TitleSource


def test_transferred_title_record_keeps_provenance_but_getter_is_unset(backend, scope_a):
    sessions = backend.open(scope_a).sessions
    sessions.create(NewSession("parent", "cli"))
    sessions.set_title("parent", "shared")
    sessions.end("parent", "compression")
    sessions.create(NewSession("child", "cli", parent_session_id="parent"))
    sessions.set_title("child", "shared")
    record = sessions.get("parent")
    assert record.title is None
    assert record.title_source is TitleSource.USER
    assert sessions.get_title_source("parent") is None
    assert sessions.get("child").title == "shared"
