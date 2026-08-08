"""Auto-title must write into the session's OWN profile store.

Regression for the phantom-session bug: a desktop session running under a
non-launch profile (e.g. ``keo``) was titled through the launch-profile
handle. Because ``record_auxiliary_usage`` ensures a session row exists
(``ensure_session(..., source="unknown")``) before recording the
``title_generation`` usage, that write MATERIALISED a 0-message clone of the
session in ``~/.hermes/state.db`` — surfacing in the usage dashboard as a
duplicate "Default (default) · unknown" session — while the real row in the
profile's own store stayed untitled.

These tests exercise the real ``SessionDB`` against temp files (no mocks of
the storage layer) so the cross-DB isolation is proven end to end.
"""

import threading

import pytest

from agent.title_generator import auto_title_session
from hermes_state import SessionDB


@pytest.fixture
def launch_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "launch" / "state.db")
    yield db
    db.close()


@pytest.fixture
def profile_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "profiles" / "keo" / "state.db")
    yield db
    db.close()


def _seed_session(db, session_id: str) -> None:
    db.create_session(session_id, "desktop")
    db.append_message(session_id, "user", "hello")
    db.append_message(session_id, "assistant", "hi there")


def test_db_factory_titles_profile_store_and_leaves_launch_db_untouched(
    tmp_path, launch_db, profile_db, monkeypatch
):
    """The whole bug in one assertion: no phantom row in the launch store."""
    session_id = "20260807_114321_cf3ed7"
    _seed_session(profile_db, session_id)

    monkeypatch.setattr(
        "agent.title_generator.generate_title",
        lambda *a, **k: "Claude Model Confirmation",
    )

    opened = []

    def factory():
        db = SessionDB(db_path=tmp_path / "profiles" / "keo" / "state.db")
        opened.append(db)
        return db

    auto_title_session(
        None,
        session_id,
        "hello",
        "hi there",
        db_factory=factory,
    )

    # Title landed in the profile store...
    assert profile_db.get_session_title(session_id) == "Claude Model Confirmation"
    # ...and the launch store never learned this session exists.
    assert launch_db.get_session(session_id) is None


def test_db_factory_handle_is_closed_by_the_titler_thread(tmp_path, profile_db, monkeypatch):
    """A fire-and-forget thread must not leak the handle it opened."""
    session_id = "sess-close"
    _seed_session(profile_db, session_id)
    monkeypatch.setattr("agent.title_generator.generate_title", lambda *a, **k: "Some Title")

    opened = []

    def factory():
        db = SessionDB(db_path=tmp_path / "profiles" / "keo" / "state.db")
        opened.append(db)
        return db

    auto_title_session(None, session_id, "hello", "hi there", db_factory=factory)

    assert len(opened) == 1
    with pytest.raises(Exception):
        # Closed handles reject further use.
        opened[0].get_session_title(session_id)


def test_db_factory_handle_closed_even_when_generation_raises(
    tmp_path, profile_db, monkeypatch
):
    """The close is in ``finally`` — a generation failure must not leak it."""
    session_id = "sess-boom"
    _seed_session(profile_db, session_id)

    def _boom(*a, **k):
        raise RuntimeError("title backend down")

    monkeypatch.setattr("agent.title_generator.generate_title", _boom)

    opened = []

    def factory():
        db = SessionDB(db_path=tmp_path / "profiles" / "keo" / "state.db")
        opened.append(db)
        return db

    # Never raises: daemon-thread target swallows.
    auto_title_session(None, session_id, "hello", "hi there", db_factory=factory)

    assert len(opened) == 1
    with pytest.raises(Exception):
        opened[0].get_session_title(session_id)


def test_borrowed_session_db_is_not_closed(launch_db, monkeypatch):
    """Launch-profile sessions still pass a borrowed handle — keep it open."""
    session_id = "sess-borrowed"
    _seed_session(launch_db, session_id)
    monkeypatch.setattr("agent.title_generator.generate_title", lambda *a, **k: "Borrowed Title")

    auto_title_session(launch_db, session_id, "hello", "hi there")

    # Still usable => not closed, and the title was written here.
    assert launch_db.get_session_title(session_id) == "Borrowed Title"


def test_auxiliary_usage_lands_in_the_profile_store_not_the_launch_store(
    tmp_path, launch_db, profile_db, monkeypatch
):
    """The `title_generation` usage row is what created the phantom.

    ``record_auxiliary_usage`` calls ``ensure_session(..., source="unknown")``
    to satisfy the FK, so recording against the wrong handle is what actually
    conjured the row. Pin it to the correct store.
    """
    session_id = "sess-usage"
    _seed_session(profile_db, session_id)

    def _generate(*a, **k):
        # Simulate the aux client recording usage mid-generation, which is
        # what the real accounting context does.
        db = SessionDB(db_path=tmp_path / "profiles" / "keo" / "state.db")
        try:
            db.record_auxiliary_usage(
                session_id,
                "title_generation",
                model="deepseek-v4-flash",
                input_tokens=126,
                output_tokens=310,
            )
        finally:
            db.close()
        return "Usage Routed Title"

    monkeypatch.setattr("agent.title_generator.generate_title", _generate)

    auto_title_session(
        None,
        session_id,
        "hello",
        "hi there",
        db_factory=lambda: SessionDB(
            db_path=tmp_path / "profiles" / "keo" / "state.db"
        ),
    )

    # The launch store has no row and therefore no phantom.
    assert launch_db.get_session(session_id) is None
    assert profile_db.get_session_title(session_id) == "Usage Routed Title"


def test_no_db_and_no_factory_is_a_silent_noop():
    """Guard still short-circuits when neither handle nor factory is given."""
    auto_title_session(None, "sess-x", "hello", "hi there")  # must not raise


def test_titler_runs_off_thread_without_escaping_exceptions(tmp_path, profile_db, monkeypatch):
    """Factory blowing up must not spray a traceback from the daemon thread."""
    session_id = "sess-factory-boom"
    _seed_session(profile_db, session_id)
    monkeypatch.setattr("agent.title_generator.generate_title", lambda *a, **k: "T")

    def _bad_factory():
        raise OSError("disk gone")

    errors = []
    t = threading.Thread(
        target=lambda: auto_title_session(
            None, session_id, "hello", "hi", db_factory=_bad_factory
        )
    )
    t.start()
    t.join(timeout=5)

    assert not t.is_alive()
    assert not errors
