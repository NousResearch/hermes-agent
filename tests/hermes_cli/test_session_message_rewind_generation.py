"""REST message pages carry the durable rewind generation (#119819).

A refreshed page that omits the client's newest rendered rows is ambiguous: a
stale read (retain the rendered rows) or an intentional rewind (the page is
the authority; the undone rows must stay gone). ``sessions.rewind_count`` is the
generation signal that disambiguates, so every message page must carry it and
bump it when a rewind lands.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.mark.parametrize("serving_profile", ["work", None])
def test_message_pages_carry_the_rewind_generation(tmp_path, monkeypatch, serving_profile):
    from hermes_state import SessionDB

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    work_home = default_home / "profiles" / "work" if serving_profile else default_home / "custom-home"
    default_home.mkdir(parents=True)
    work_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(work_home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", work_home / "state.db")

    for home in (default_home, work_home):
        db = SessionDB(db_path=home / "state.db")
        try:
            db.create_session(session_id="same-id", source="desktop")
            db.append_messages_batch("same-id", [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second question"},
                {"role": "assistant", "content": "second answer"},
            ])
        finally:
            db.close()

    from hermes_cli.web_routers.sessions import manage_router

    app = FastAPI()
    app.include_router(manage_router)
    with TestClient(app) as client:
        query = "limit=120&order=latest&include_compacted=true"
        before = client.get(f"/api/sessions/same-id/messages?{query}").json()
        assert before["rewind_generation"] == 0
        assert len(before["messages"]) == 4

        # A rewind (what /undo drives server-side) bumps the generation BEFORE
        # the soft-deleted rows vanish from the page. Rewind targets a
        # user-originated row and soft-deletes it and everything after it.
        db = SessionDB(db_path=work_home / "state.db")
        try:
            target = [
                row["id"] for row in db.get_messages("same-id") if row["role"] == "user"
            ][-1]
            db.rewind_to_message("same-id", target)
        finally:
            db.close()

        after = client.get(f"/api/sessions/same-id/messages?{query}").json()
        assert after["rewind_generation"] == 1
        # The undone rows are gone from the page, and the generation is what
        # tells the client this omission is authoritative, not a stale read.
        assert [row["content"] for row in after["messages"]] == ["first question", "first answer"]


def test_message_pages_generation_survives_profile_pinning(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(session_id="s", source="desktop")
        db.append_messages_batch("s", [{"role": "user", "content": "a"}])
        target = db.get_messages("s")[-1]["id"]
        db.rewind_to_message("s", target)
        db.rewind_to_message  # second rewind attempt below uses a fresh target
        db.append_messages_batch("s", [{"role": "user", "content": "b"}])
        target2 = db.get_messages("s")[-1]["id"]
        db.rewind_to_message("s", target2)
    finally:
        db.close()

    from hermes_cli.web_routers.sessions import manage_router

    app = FastAPI()
    app.include_router(manage_router)
    with TestClient(app) as client:
        page = client.get("/api/sessions/s/messages?limit=120&order=latest").json()

    assert page["rewind_generation"] == 2
