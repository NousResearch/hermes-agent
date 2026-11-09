"""Behaviour-contract tests for the generic plugin state surface:
``ctx.record_delegation_delivery`` / ``ctx.record_completion_note`` and the
read-only ``ctx.read_session*`` / ``ctx.read_kanban_*`` facade.

The contracts pinned here: writes are recorded once (idempotent by stable
id), reads are versioned plain dicts, a missing capability fails closed
without writing or reading anything, and a read-only kanban access never
creates or migrates a board.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import hermes_yaml as yaml
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.plugins import PluginManager

PLUGIN_NAME = "state-ctx-fixture"

PLUGIN_INIT = '''"""Test plugin capturing its ctx for state-surface tests."""

CTX = None


def register(ctx):
    global CTX
    CTX = ctx
'''

EXPECTED_TASK_DTO_FIELDS = {
    "id",
    "title",
    "status",
    "assignee",
    "priority",
    "session_id",
    "board",
    "created_at",
    "started_at",
    "completed_at",
    "result",
    "block_kind",
    "block_recurrences",
    "completion_contract",
    "current_step_key",
}


def _load_fixture_plugin(tmp_path, monkeypatch, *, granted: bool):
    """Load the fixture plugin in an isolated home; return (home, ctx)."""
    home = tmp_path / "hermes-home"
    plugins_dir = home / "plugins"
    plug = plugins_dir / PLUGIN_NAME
    plug.mkdir(parents=True)
    (plug / "plugin.yaml").write_text(
        yaml.safe_dump({
            "name": PLUGIN_NAME,
            "version": "0.1.0",
            "description": "Test plugin for the ctx state surface",
            "capabilities": ["state.detached_result", "state.read"],
        }),
        encoding="utf-8",
    )
    (plug / "__init__.py").write_text(PLUGIN_INIT, encoding="utf-8")
    entries: dict = {PLUGIN_NAME: {}}
    if granted:
        entries[PLUGIN_NAME] = {
            "granted_capabilities": ["state.detached_result", "state.read"]
        }
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": [PLUGIN_NAME], "entries": entries}}),
        encoding="utf-8",
    )
    empty_bundled = tmp_path / "bundled-plugins"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty_bundled))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    manager = PluginManager()
    manager.discover_and_load()
    loaded = manager._plugins[PLUGIN_NAME]
    assert loaded.enabled is True
    assert loaded.error is None
    ctx = loaded.module.CTX
    assert ctx is not None
    return home, ctx


@pytest.fixture
def granted(tmp_path, monkeypatch):
    return _load_fixture_plugin(tmp_path, monkeypatch, granted=True)


@pytest.fixture
def denied(tmp_path, monkeypatch):
    return _load_fixture_plugin(tmp_path, monkeypatch, granted=False)


def _ensure_session(home: Path, session_id: str) -> str:
    from hermes_state_registry import acquire, release

    db = acquire(home / "state.db")
    try:
        return db.ensure_session(session_id, source="test")
    finally:
        release(db)


def _session_messages(home: Path, session_id: str) -> list:
    from hermes_state_registry import acquire, release

    db = acquire(home / "state.db")
    try:
        return db.get_messages(session_id)
    finally:
        release(db)


def _make_task(home: Path, **kwargs) -> str:
    kb.init_db()
    conn = kbc.connect()
    try:
        return kb.create_task(conn, **kwargs)
    finally:
        conn.close()


# ── Detached results ────────────────────────────────────────────────────────


def test_record_delegation_delivery_roundtrip_and_idempotent(granted):
    home, ctx = granted
    _ensure_session(home, "s-origin")

    first = ctx.record_delegation_delivery(
        "s-origin", "research done", delegation_id="d-1"
    )
    assert isinstance(first, int)
    second = ctx.record_delegation_delivery(
        "s-origin", "research done", delegation_id="d-1"
    )
    assert second == first  # recorded once: same id, no duplicate row

    rows = _session_messages(home, "s-origin")
    assert len(rows) == 1
    assert rows[0]["content"] == "research done"
    assert rows[0]["display_kind"] == "async_delegation_complete"


def test_record_delegation_delivery_requires_delegation_id(granted):
    _, ctx = granted
    with pytest.raises(ValueError):
        ctx.record_delegation_delivery("s-origin", "x", delegation_id="")


def test_record_completion_note_roundtrip_and_idempotent(granted):
    home, ctx = granted
    _ensure_session(home, "s-origin")

    first = ctx.record_completion_note(
        "s-origin", "task finished while you were away", admission_id="a-1"
    )
    assert isinstance(first, int)
    assert (
        ctx.record_completion_note(
            "s-origin", "task finished while you were away", admission_id="a-1"
        )
        == first
    )
    assert len(_session_messages(home, "s-origin")) == 1


def test_record_rejects_empty_session_id(granted):
    _, ctx = granted
    with pytest.raises(ValueError):
        ctx.record_delegation_delivery("", "x", delegation_id="d-1")
    with pytest.raises(ValueError):
        ctx.record_completion_note("", "x", admission_id="a-1")


def test_detached_result_denied_without_capability(denied):
    home, ctx = denied
    _ensure_session(home, "s-origin")

    assert ctx.record_delegation_delivery("s-origin", "x", delegation_id="d-1") is None
    assert ctx.record_completion_note("s-origin", "x", admission_id="a-1") is None
    assert _session_messages(home, "s-origin") == []


# ── Read-only state facade ──────────────────────────────────────────────────


def test_read_session_and_messages(granted):
    home, ctx = granted
    _ensure_session(home, "s-origin")
    ctx.record_delegation_delivery("s-origin", "hello", delegation_id="d-1")

    session = ctx.read_session("s-origin")
    assert session is not None
    assert session["id"] == "s-origin"

    page = ctx.read_session_messages("s-origin")
    assert page["session_id"] == "s-origin"
    assert page["schema_version"] is not None
    assert [m["content"] for m in page["messages"]] == ["hello"]
    assert page["next_after_id"] is None


def test_read_session_messages_keyset_paging(granted):
    home, ctx = granted
    _ensure_session(home, "s-origin")
    for i in range(3):
        ctx.record_delegation_delivery("s-origin", f"note {i}", delegation_id=f"d-{i}")

    first = ctx.read_session_messages("s-origin", limit=2)
    assert [m["content"] for m in first["messages"]] == ["note 0", "note 1"]
    assert first["next_after_id"] is not None

    second = ctx.read_session_messages(
        "s-origin", limit=2, after_id=first["next_after_id"]
    )
    assert [m["content"] for m in second["messages"]] == ["note 2"]
    assert second["next_after_id"] is None


def test_read_session_messages_rejects_after_id_with_compacted(granted):
    _, ctx = granted
    with pytest.raises(ValueError):
        ctx.read_session_messages("s-x", after_id=5, include_compacted=True)


def test_resolve_session_returns_live_tip(granted):
    home, ctx = granted
    _ensure_session(home, "s-origin")
    assert ctx.resolve_session("s-origin") == "s-origin"


def test_state_read_denied_without_capability(denied):
    home, ctx = denied
    _ensure_session(home, "s-origin")

    assert ctx.read_session("s-origin") is None
    page = ctx.read_session_messages("s-origin")
    assert page["messages"] == []
    assert ctx.resolve_session("s-origin") == "s-origin"


def test_read_kanban_task_closed_dto(granted):
    home, ctx = granted
    task_id = _make_task(
        home,
        title="write the summary",
        assignee="default",
        session_id="s-origin",
        initial_status="running",
    )

    dto = ctx.read_kanban_task(task_id)
    assert dto is not None
    assert set(dto) == EXPECTED_TASK_DTO_FIELDS
    assert dto["id"] == task_id
    assert dto["title"] == "write the summary"
    # create_task maps initial_status="running" to the stored "ready" status.
    assert dto["status"] == "ready"
    assert dto["session_id"] == "s-origin"
    assert dto["board"] == "default"
    assert "claim_lock" not in dto and "workspace_path" not in dto


def test_list_kanban_tasks_filters(granted):
    home, ctx = granted
    _make_task(
        home,
        title="one",
        assignee="default",
        session_id="s-a",
        initial_status="running",
    )
    _make_task(
        home, title="two", assignee="other", session_id="s-a", initial_status="blocked"
    )
    _make_task(
        home,
        title="three",
        assignee="default",
        session_id="s-b",
        initial_status="running",
    )

    page = ctx.list_kanban_tasks(session_id="s-a")
    assert page["api_version"] == 1
    assert sorted(t["title"] for t in page["tasks"]) == ["one", "two"]

    blocked = ctx.list_kanban_tasks(status="blocked")
    assert [t["title"] for t in blocked["tasks"]] == ["two"]

    with pytest.raises(ValueError):
        ctx.list_kanban_tasks(status="not-a-status")


def test_read_kanban_task_summary(granted):
    home, ctx = granted
    task_id = _make_task(home, title="summarised", assignee="default")
    assert ctx.read_kanban_task_summary(task_id) is None

    conn = sqlite3.connect(str(home / "kanban.db"))
    try:
        conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at, ended_at, summary)"
            " VALUES (?, 'done', 1, 2, 'did the thing')",
            (task_id,),
        )
        conn.commit()
    finally:
        conn.close()
    assert ctx.read_kanban_task_summary(task_id) == "did the thing"


def test_read_kanban_never_creates_board(granted):
    home, ctx = granted
    assert not (home / "kanban.db").exists()

    assert ctx.read_kanban_task("nope") is None
    assert ctx.list_kanban_tasks() == {"api_version": 1, "tasks": []}
    assert ctx.read_kanban_task_summary("nope") is None
    assert not (home / "kanban.db").exists()


def test_kanban_read_denied_without_capability(denied):
    home, ctx = denied
    task_id = _make_task(home, title="secret", assignee="default")

    assert ctx.read_kanban_task(task_id) is None
    assert ctx.list_kanban_tasks() == {"api_version": 1, "tasks": []}
    assert ctx.read_kanban_task_summary(task_id) is None
