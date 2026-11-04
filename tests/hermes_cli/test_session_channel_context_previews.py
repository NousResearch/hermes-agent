"""Session previews must show what the user typed, not the adapter's ``channel_context``.

The gateway persists a turn that carries ``MessageEvent.channel_context`` as
``{channel_context}\\n\\n[New message]\\n{text}`` (``gateway/run_inbound.py``). Every
preview surface (session lists, the timeline, the /rewind and /undo picker) used the
head of that stored content, so an adapter that adds context on every turn showed
the same context line for every session.

These drive the real SQL/shaping path through SessionDB, like the skill-preview tests.
"""

from pathlib import Path

import pytest

from agent import skill_commands
from hermes_state import SessionDB
from hermes_state_common import CHANNEL_CONTEXT_MESSAGE_MARKER, strip_channel_context
from tools import skills_tool

CONTEXT = "[Chat room_id: R1]\n[Recent chat messages - context only, not requests]\n[10:00] bob: hi"


def _turn(text: str) -> str:
    return f"{CONTEXT}{CHANNEL_CONTEXT_MESSAGE_MARKER}{text}"


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def _seed(db, session_id, content):
    db.create_session(session_id=session_id, source="rocketchat", model="m")
    db.append_message(session_id, role="user", content=content)
    db.append_message(session_id, role="assistant", content="On it.")


def test_session_list_preview_is_the_typed_message(db):
    _seed(db, "s1", _turn("summarize the release notes"))
    _seed(db, "s2", _turn("what changed in the deploy?"))
    previews = {row["id"]: row["preview"] for row in db.list_sessions_rich(limit=10)}
    assert previews == {"s1": "summarize the release notes", "s2": "what changed in the deploy?"}


def test_sender_prefix_after_the_marker_stays(db):
    # A shared multi-user session gets the ``[name]`` prefix before the context is joined.
    _seed(db, "s1", _turn("[alice] summarize the release notes"))
    (row,) = db.list_sessions_rich(limit=10)
    assert row["preview"] == "[alice] summarize the release notes"


def test_message_without_context_is_unchanged(db):
    _seed(db, "s1", "plain question\nwith two lines")
    (row,) = db.list_sessions_rich(limit=10)
    assert row["preview"] == "plain question with two lines"


def test_auto_loaded_skill_behind_context_shows_the_request(db, tmp_path, monkeypatch):
    # A channel-bound skill is prepended to event.text before the context is joined.
    skills_dir = tmp_path / "skills"
    (skills_dir / "work").mkdir(parents=True)
    (skills_dir / "work" / "SKILL.md").write_text(
        "---\nname: work\ndescription: Work\n---\n\n# work\n\n" + "Do the work carefully. " * 60 + "\n")
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_commands, "_skill_commands_by_key", {})
    skill_commands.scan_skill_commands()
    loaded = skill_commands._load_skill_payload("work")
    payload = skill_commands._build_skill_message(
        loaded[0], loaded[1],
        '[IMPORTANT: The "work" skill is auto-loaded. Follow its instructions for this session.]',
    )
    _seed(db, "s1", _turn(f"{payload}\n\nwhy is my deploy timing out?"))
    (row,) = db.list_sessions_rich(limit=10)
    assert row["preview"] == "why is my deploy timing out?"


def test_rewind_picker_preview_is_the_typed_message(db):
    _seed(db, "s1", _turn("summarize the release notes"))
    (entry,) = db.list_recent_user_messages("s1")
    assert entry["preview"] == "summarize the release notes"


def test_timeline_preview_is_the_typed_message(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    store = SessionDB(db_path=home / "state.db")
    try:
        store.create_session(session_id="t1", source="rocketchat")
        store.append_messages_batch("t1", [{"role": "user", "content": _turn("summarize the notes"), "timestamp": 100}])
        from hermes_cli.web_routers.sessions import manage_router

        app = FastAPI()
        app.include_router(manage_router)
        with TestClient(app) as client:
            entries = client.get("/api/sessions/t1/timeline").json()["entries"]
        assert [e["preview"] for e in entries] == ["summarize the notes"]
    finally:
        store.close()


def test_strip_channel_context_uses_the_first_marker():
    assert strip_channel_context(_turn("a")) == "a"
    assert strip_channel_context("no marker") == "no marker"
    assert strip_channel_context(_turn(f"a{CHANNEL_CONTEXT_MESSAGE_MARKER}b")) == f"a{CHANNEL_CONTEXT_MESSAGE_MARKER}b"
