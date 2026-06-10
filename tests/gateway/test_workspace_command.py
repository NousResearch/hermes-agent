"""Tests for gateway /workspace session cwd command."""

from datetime import datetime
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_cli.commands import resolve_command


class _FakeSessionDB:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    def get_session(self, session_id: str):
        return self.rows.get(session_id)

    def update_session_cwd(self, session_id: str, cwd: str):
        self.rows.setdefault(session_id, {"id": session_id})["cwd"] = cwd or None


class _Store:
    def __init__(self, entry):
        self.entry = entry

    def get_or_create_session(self, _source):
        return self.entry


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_source(), message_id="m1")


def _runner(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    source = _source()
    entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    db = _FakeSessionDB()
    db.rows[entry.session_id] = {"id": entry.session_id, "cwd": None}

    runner = object.__new__(GatewayRunner)
    runner._session_db = db
    runner.session_store = _Store(entry)
    evicted: list[str] = []
    runner._evict_cached_agent = evicted.append
    runner._global_gateway_cwd = lambda: str(tmp_path / "global")

    registered: list[tuple[str, dict]] = []
    cleared: list[str] = []
    cleaned: list[str] = []
    monkeypatch.setattr(
        "tools.terminal_tool.register_task_env_overrides",
        lambda task_id, overrides: registered.append((task_id, overrides)),
    )
    monkeypatch.setattr(
        "tools.terminal_tool.cleanup_vm",
        lambda task_id: cleaned.append(task_id),
    )
    monkeypatch.setattr(
        "tools.terminal_tool.clear_task_env_overrides",
        lambda task_id: cleared.append(task_id),
    )
    runner._workspace_probe = SimpleNamespace(
        registered=registered,
        cleared=cleared,
        cleaned=cleaned,
        evicted=evicted,
    )
    return runner, entry, db


def test_workspace_command_registered_with_cwd_alias():
    cmd = resolve_command("workspace")
    alias = resolve_command("cwd")

    assert cmd is not None
    assert cmd.name == "workspace"
    assert alias is cmd


@pytest.mark.asyncio
async def test_workspace_set_and_clear_persist_session_cwd(tmp_path, monkeypatch):
    runner, entry, db = _runner(tmp_path, monkeypatch)
    project = tmp_path / "project"
    project.mkdir()

    set_result = await runner._handle_workspace_command(_event(f"/workspace {project}"))

    assert "Workspace set" in set_result
    assert db.rows[entry.session_id]["cwd"] == str(project)
    assert (entry.session_id, {"cwd": str(project)}) in runner._workspace_probe.registered
    assert entry.session_id in runner._workspace_probe.cleaned
    assert entry.session_key in runner._workspace_probe.evicted

    clear_result = await runner._handle_workspace_command(_event("/workspace clear"))

    assert "cleared" in clear_result
    assert db.rows[entry.session_id]["cwd"] is None
    assert entry.session_id in runner._workspace_probe.cleared


def test_workspace_cwd_carries_across_session_id_rotation(tmp_path, monkeypatch):
    runner, entry, db = _runner(tmp_path, monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    db.rows[entry.session_id]["cwd"] = str(project)

    runner._carry_gateway_session_cwd(entry.session_id, "sess-2")

    assert db.rows["sess-2"]["cwd"] == str(project)
    assert ("sess-2", {"cwd": str(project)}) in runner._workspace_probe.registered
    assert entry.session_id in runner._workspace_probe.cleared


@pytest.mark.asyncio
async def test_workspace_rejects_relative_and_missing_paths(tmp_path, monkeypatch):
    runner, _entry, _db = _runner(tmp_path, monkeypatch)

    relative = await runner._handle_workspace_command(_event("/workspace project"))
    missing = await runner._handle_workspace_command(_event(f"/workspace {tmp_path / 'missing'}"))

    assert "absolute path" in relative
    assert "not an existing directory" in missing
