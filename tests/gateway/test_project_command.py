"""Tests for /project gateway slash command (messaging-side 'Move to project')."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, build_session_key


def _make_event(text="/project", platform=Platform.SLACK,
                user_id="U123", chat_id="D456"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner(tmp_path, current_session_id="sess_001", event=None, *, row_cwd=""):
    from gateway.run import GatewayRunner
    from hermes_state import SessionDB, AsyncSessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session(current_session_id, source="slack", cwd=row_cwd or None)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = SimpleNamespace(platforms={})
    runner._session_db = AsyncSessionDB(db)
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None

    session_key = build_session_key(event.source) if event else "agent:main:slack:dm"

    entry = MagicMock()
    entry.session_id = current_session_id
    entry.session_key = session_key
    store = MagicMock()
    store.get_or_create_session.return_value = entry
    store._generate_session_key = lambda src: build_session_key(src)
    runner.session_store = store

    evicted = []
    runner._evict_cached_agent = lambda session_key: evicted.append(session_key)
    runner._evictions = evicted

    async_store = MagicMock()
    async_store.get_or_create_session.return_value = entry
    runner._async_session_store = async_store

    runner._session_key_for_source = lambda source: build_session_key(source)
    runner._normalize_source_for_session_key = lambda source: source
    runner._resolve_profile_home_for_source = lambda source: tmp_path
    return runner, db, session_key


def _patch_projects_db(monkeypatch, tmp_path, projects):
    """Point projects_db.connect_closing at a temp db holding `projects` (list of (name, folder_path))."""
    from hermes_cli import projects_db as pdb
    real_connect = pdb.connect
    db_path = tmp_path / "projects.db"

    def _connect(db_path=None):
        return real_connect(db_path=tmp_db_path)

    tmp_db_path = db_path

    monkeypatch.setattr(pdb, "connect", _connect)
    with pdb.connect_closing() as conn:
        for name, folder in projects:
            pdb.create_project(conn, name=name, folders=[folder])
    return db_path


@pytest.mark.asyncio
async def test_project_lists_projects(tmp_path, monkeypatch):
    folder = tmp_path / "repo"
    folder.mkdir()
    _patch_projects_db(monkeypatch, tmp_path, [("abv", str(folder))])

    event = _make_event("/project")
    runner, _db, _ = _make_runner(tmp_path, event=event)
    reply = await runner._handle_project_command(event)
    assert "abv" in reply
    assert "/project <name>" in reply


@pytest.mark.asyncio
async def test_project_unknown_name(tmp_path, monkeypatch):
    _patch_projects_db(monkeypatch, tmp_path, [])

    event = _make_event("/project nope")
    runner, _db, _ = _make_runner(tmp_path, event=event)
    reply = await runner._handle_project_command(event)
    assert "No project named" in reply


@pytest.mark.asyncio
async def test_project_moves_session(tmp_path, monkeypatch):
    folder = tmp_path / "repo"
    folder.mkdir()
    _patch_projects_db(monkeypatch, tmp_path, [("abv", str(folder))])

    event = _make_event("/project abv")
    runner, db, session_key = _make_runner(tmp_path, event=event, row_cwd=str(tmp_path))
    reply = await runner._handle_project_command(event)

    assert "Session moved" in reply
    row = db.get_session("sess_001")
    assert row["cwd"] == str(folder)
    assert session_key in runner._evictions


@pytest.mark.asyncio
async def test_project_bad_path(tmp_path, monkeypatch):
    folder = tmp_path / "repo"
    folder.mkdir()
    _patch_projects_db(monkeypatch, tmp_path, [("abv", str(folder))])
    folder.rmdir()  # folder no longer exists

    event = _make_event("/project abv")
    runner, db, _ = _make_runner(tmp_path, event=event)
    reply = await runner._handle_project_command(event)
    assert "does not exist" in reply
    row = db.get_session("sess_001")
    assert row["cwd"] != str(folder)
