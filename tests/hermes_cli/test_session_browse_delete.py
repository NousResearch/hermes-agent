"""Behavior of ``d`` delete in the real ``hermes sessions browse`` picker against a real SessionDB.

Drives the picker-created delete callback (curses.wrapper only bypasses terminal rendering) and
pins the contract that a live turn lease / compression lock protects a session from the browser.
"""

import os
from pathlib import Path

import pytest

from hermes_cli import sessions_cmd_browse
from hermes_state import SessionDB

TARGET, OTHER = "browse-target", "browse-other"
KEYS = {"d": ord("d"), "y": ord("y"), "Y": ord("Y"), "n": ord("n"), "enter": 10}


def _acquire(db, kind, sid, holder):
    acquire = db.try_acquire_session_turn_lease if kind == "turn" else db.try_acquire_compression_lock
    assert acquire(sid, holder, ttl_seconds=300.0) is True


def _release(db, kind, sid, holder):
    (db.release_session_turn_lease if kind == "turn" else db.release_compression_lock)(sid, holder)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("guard", ["turn", "compression"])
def test_browser_delete_respects_confirmation_and_live_guards(tmp_path, monkeypatch, guard):
    import curses

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # An AssertionError inside the wrapper would be swallowed into the fallback picker; never let it.
    monkeypatch.setattr(sessions_cmd_browse, "_fallback_picker", lambda *_: pytest.fail("fell back from curses"))

    db = SessionDB(tmp_path / "state.db")
    holder = f"pid={os.getpid()}:{guard}=1"
    held = False
    try:
        for sid in (TARGET, OTHER):
            db.create_session(sid, source="cli")
            db.append_message(sid, "user", f"hello from {sid}")
            db.append_message(sid, "assistant", f"reply for {sid}")
        sessions = [{"id": sid, "source": "cli", "title": sid} for sid in (TARGET, OTHER)]
        observed = {}

        def survivors():
            return (db.get_session(TARGET) is not None, db.message_count(TARGET), db.message_count(OTHER))

        def wrapper(run):
            browser = run.__self__  # the real browser, holding the picker's own delete callback

            def press(*names):
                for name in names:
                    browser._handle_key(KEYS[name])

            press("d", "n")  # cancelled confirmation: nothing happens
            observed["cancel"] = (survivors(), [s["id"] for s in browser.sessions], browser.confirm_delete)

            _acquire(db, guard, TARGET, holder)
            nonlocal held
            held = True
            press("d", "y")  # live guard: refused, with an active-session notice
            observed["refused"] = (survivors(), [s["id"] for s in browser.sessions], browser.flash)

            _release(db, guard, TARGET, holder)
            held = False
            press("d", "Y")
            observed["deleted"] = (survivors(), [s["id"] for s in browser.sessions], browser.flash)
            press("enter")  # the surviving session is still selectable

        monkeypatch.setattr(curses, "wrapper", wrapper)
        selected = sessions_cmd_browse._session_browse_picker(sessions, db)

        assert observed["cancel"] == ((True, 2, 2), [TARGET, OTHER], None)
        survived, ids, flash = observed["refused"]
        assert survived == (True, 2, 2) and ids == [TARGET, OTHER]
        assert "active" in flash.lower() and flash != "Deleted."
        survived, ids, flash = observed["deleted"]
        assert survived == (False, 0, 2) and ids == [OTHER]
        assert flash == "Deleted."
        assert selected == OTHER
        assert db.get_session(OTHER) is not None
    finally:
        if held:
            _release(db, guard, TARGET, holder)
        db.close()
