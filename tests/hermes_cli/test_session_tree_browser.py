"""The classic CLI's full-screen ``/tree`` browser (`hermes_cli/session_tree_browser.py`).

Key handling and view state are tested without a terminal: every view change must go back to
``SessionDB.session_family`` (the shared fold rules), Enter returns the picked id, and the keys
match the Ink overlay's.
"""

from __future__ import annotations

import pytest

from hermes_cli.session_tree_browser import _TreeBrowser, browse_tree
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.create_session("root", source="cli")
    session_db.append_message("root", "user", "plan it")
    session_db.end_session("root", "compression")
    session_db.create_session("cont", source="cli", parent_session_id="root")
    session_db.create_session("a", source="cli", parent_session_id="cont", model_config={"_branched_from": "cont"})
    session_db.set_session_title("a", "idea-a")
    session_db.create_session("a1", source="cli", parent_session_id="a", model_config={"_branched_from": "a"})
    session_db.set_session_title("a1", "idea-a1")
    session_db.create_session("b", source="cli", parent_session_id="cont", model_config={"_branched_from": "cont"})
    session_db.set_session_title("b", "idea-b")
    try:
        yield session_db
    finally:
        session_db.close()


def _browser(db, session_id="a1", show_all=False):
    return _TreeBrowser(object(), db, session_id, show_all)


def _ids(browser):
    return [row["session_id"] for row in browser.rows]


def test_it_opens_on_the_current_session_with_compressions_folded(db) -> None:
    browser = _browser(db)

    assert _ids(browser) == ["root", "a", "a1", "b"]
    assert browser.rows[browser.cursor]["session_id"] == "a1"


def test_enter_switches_to_the_highlighted_row_but_not_to_where_you_are(db) -> None:
    browser = _browser(db)

    assert browser.handle("enter") is False  # already here: nothing to switch to
    browser.handle("down")
    assert browser.handle("enter") is True
    assert browser.result == "b"


def test_left_folds_a_parent_and_right_unfolds_it(db) -> None:
    browser = _browser(db)
    browser.handle("up")  # idea-a, which has a child

    browser.handle("left")
    assert _ids(browser) == ["root", "a", "b"]
    assert browser.rows[1]["folded"] is True

    browser.handle("right")
    assert _ids(browser) == ["root", "a", "a1", "b"]


def test_left_on_a_leaf_jumps_to_its_parent(db) -> None:
    browser = _browser(db)

    browser.handle("left")

    assert browser.rows[browser.cursor]["session_id"] == "a"
    assert _ids(browser) == ["root", "a", "a1", "b"]


def test_c_and_s_ask_the_store_for_the_wider_view(db) -> None:
    browser = _browser(db)

    browser.handle("c")
    assert "cont" in _ids(browser)
    browser.handle("c")
    assert "cont" not in _ids(browser)


def test_search_filters_titles_and_esc_clears_it(db) -> None:
    browser = _browser(db)

    browser.handle("/")
    for char in "idea-b":
        browser.handle(char)
    assert _ids(browser) == ["b"]
    browser.handle("esc")
    assert browser.searching is False and _ids(browser) == ["root", "a", "a1", "b"]


def test_esc_and_q_close_without_a_pick(db) -> None:
    for key in ("esc", "q"):
        browser = _browser(db)
        assert browser.handle(key) is True
        assert browser.result is None


def test_a_family_of_one_never_opens_the_browser(db, monkeypatch) -> None:
    import curses

    monkeypatch.setattr(curses, "wrapper", lambda *_a, **_k: pytest.fail("nothing to browse"))
    db.create_session("lonely", source="cli")

    assert browse_tree(db, "lonely") is None
