"""Integration tests for the opt-in Kanban recovery supervisor plugin."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.plugins import PluginManager, get_plugin_manager


PLUGIN_NAME = "kanban-recovery-supervisor"


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Give each test a board and an installed supervisor profile."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "profiles" / "oink").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def enable_plugin(home: Path, **settings):
    config = {
        "plugins": {"enabled": [PLUGIN_NAME]},
        "kanban_recovery_supervisor": settings,
    }
    import yaml

    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def test_registers_post_commit_block_hook(isolated_home):
    enable_plugin(isolated_home)

    manager = PluginManager()
    manager.discover_and_load()

    assert PLUGIN_NAME in manager._plugins
    assert manager._plugins[PLUGIN_NAME].enabled is True
    assert len(manager._hooks["kanban_task_blocked"]) == 1


def test_safe_eligible_block_creates_one_independent_recovery_card(isolated_home):
    enable_plugin(
        isolated_home,
        enabled_boards=["default"],
        mode="safe_recovery",
        supervisor_profile="oink",
    )
    manager = PluginManager()
    manager.discover_and_load()
    callback = manager._hooks["kanban_task_blocked"][0]

    lifecycle_manager = get_plugin_manager()
    saved_hooks = {name: list(hooks) for name, hooks in lifecycle_manager._hooks.items()}
    lifecycle_manager._hooks["kanban_task_blocked"] = [callback]
    try:
        conn = kb.connect()
        try:
            source_id = kb.create_task(
                conn,
                title="Source task",
                assignee="froink",
            )
            kb.add_notify_sub(
                conn,
                task_id=source_id,
                platform="telegram",
                chat_id="1234",
                thread_id="5678",
                notifier_profile="froink",
            )
            assert kb.claim_task(conn, source_id) is not None
            assert kb.block_task(
                conn,
                source_id,
                reason="Provider returned HTTP 429 rate limit",
                kind="transient",
            ) is True
        finally:
            conn.close()
    finally:
        lifecycle_manager._hooks = saved_hooks

    conn = kb.connect()
    try:
        recoveries = [
            task for task in kb.list_tasks(conn, include_archived=True)
            if task.created_by == PLUGIN_NAME
        ]
        source = kb.get_task(conn, source_id)
        comments = kb.list_comments(conn, source_id)
        subscriptions = kb.list_notify_subs(conn)
    finally:
        conn.close()

    assert len(recoveries) == 1
    recovery = recoveries[0]
    assert recovery.assignee == "oink"
    assert recovery.status == "ready"
    assert source_id in (recovery.body or "")
    assert "Prohibited" in (recovery.body or "")
    assert source is not None and source.status == "blocked"
    assert any(recovery.id in comment.body for comment in comments)
    assert any(sub["task_id"] == recovery.id for sub in subscriptions)


def test_duplicate_failure_event_creates_only_one_card_from_durable_state(isolated_home):
    enable_plugin(
        isolated_home,
        enabled_boards=["default"],
        mode="safe_recovery",
        supervisor_profile="oink",
        cooldown_seconds=0,
    )
    manager = PluginManager()
    manager.discover_and_load()
    callback = manager._hooks["kanban_task_blocked"][0]

    conn = kb.connect()
    try:
        source_id = kb.create_task(conn, title="Source", assignee="froink")
        kb.claim_task(conn, source_id)
        kb.block_task(conn, source_id, reason="stale worker claim", kind="transient")
    finally:
        conn.close()

    event = {
        "task_id": source_id,
        "board": "default",
        "run_id": 41,
        "reason": "stale worker claim",
    }
    callback(**event)
    callback(**event)
    callback(**{**event, "run_id": 42})

    conn = kb.connect()
    try:
        recoveries = [
            task for task in kb.list_tasks(conn, include_archived=True)
            if task.created_by == PLUGIN_NAME
        ]
    finally:
        conn.close()
    assert len(recoveries) == 1

    state_path = kb.board_dir("default") / "plugin-state" / f"{PLUGIN_NAME}.sqlite3"
    with sqlite3.connect(state_path) as state:
        events = state.execute(
            "SELECT action FROM recovery_events ORDER BY created_at, event_signature"
        ).fetchall()
    assert len(events) == 2
    assert ("cap_reached",) in events


def test_cooldown_blocks_a_different_safe_signature_for_the_same_source(isolated_home):
    enable_plugin(
        isolated_home,
        enabled_boards=["default"],
        mode="safe_recovery",
        supervisor_profile="oink",
        cooldown_seconds=900,
    )
    manager = PluginManager()
    manager.discover_and_load()
    callback = manager._hooks["kanban_task_blocked"][0]

    conn = kb.connect()
    try:
        source_id = kb.create_task(conn, title="Source", assignee="froink")
        kb.claim_task(conn, source_id)
        kb.block_task(conn, source_id, reason="provider HTTP 503", kind="transient")
    finally:
        conn.close()

    callback(task_id=source_id, board="default", run_id=1, reason="provider HTTP 503")
    callback(task_id=source_id, board="default", run_id=2, reason="stale worker claim")

    conn = kb.connect()
    try:
        recoveries = [
            task for task in kb.list_tasks(conn, include_archived=True)
            if task.created_by == PLUGIN_NAME
        ]
    finally:
        conn.close()
    assert len(recoveries) == 1

    state_path = kb.board_dir("default") / "plugin-state" / f"{PLUGIN_NAME}.sqlite3"
    with sqlite3.connect(state_path) as state:
        actions = {row[0] for row in state.execute("SELECT action FROM recovery_events")}
    assert "cooldown" in actions


def test_failed_card_creation_is_reconciled_on_the_next_same_event(
    isolated_home, monkeypatch
):
    enable_plugin(
        isolated_home,
        enabled_boards=["default"],
        mode="safe_recovery",
        supervisor_profile="oink",
        cooldown_seconds=0,
    )
    manager = PluginManager()
    manager.discover_and_load()
    callback = manager._hooks["kanban_task_blocked"][0]

    conn = kb.connect()
    try:
        source_id = kb.create_task(conn, title="Source", assignee="froink")
        kb.claim_task(conn, source_id)
        kb.block_task(conn, source_id, reason="provider HTTP 503", kind="transient")
    finally:
        conn.close()

    original_create_task = kb.create_task

    def fail_recovery_creation(conn, **kwargs):
        if kwargs.get("created_by") == PLUGIN_NAME:
            raise sqlite3.OperationalError("simulated board write failure")
        return original_create_task(conn, **kwargs)

    event = {
        "task_id": source_id,
        "board": "default",
        "run_id": 50,
        "reason": "provider HTTP 503",
    }
    monkeypatch.setattr(kb, "create_task", fail_recovery_creation)
    callback(**event)
    monkeypatch.setattr(kb, "create_task", original_create_task)
    callback(**event)

    conn = kb.connect()
    try:
        recoveries = [
            task for task in kb.list_tasks(conn, include_archived=True)
            if task.created_by == PLUGIN_NAME
        ]
    finally:
        conn.close()
    assert len(recoveries) == 1

    state_path = kb.board_dir("default") / "plugin-state" / f"{PLUGIN_NAME}.sqlite3"
    with sqlite3.connect(state_path) as state:
        action, recovery_id = state.execute(
            "SELECT action, recovery_task_id FROM recovery_events"
        ).fetchone()
    assert action == "created"
    assert recovery_id == recoveries[0].id


def test_notify_only_audits_an_eligible_event_without_creating_a_card(isolated_home):
    enable_plugin(
        isolated_home,
        enabled_boards=["default"],
        mode="notify_only",
        supervisor_profile="oink",
    )
    manager = PluginManager()
    manager.discover_and_load()
    callback = manager._hooks["kanban_task_blocked"][0]

    conn = kb.connect()
    try:
        source_id = kb.create_task(conn, title="Source", assignee="froink")
        kb.claim_task(conn, source_id)
        kb.block_task(conn, source_id, reason="missing local upload transport", kind="transient")
    finally:
        conn.close()

    callback(
        task_id=source_id,
        board="default",
        run_id=42,
        reason="missing local upload transport",
    )

    conn = kb.connect()
    try:
        recoveries = [
            task for task in kb.list_tasks(conn, include_archived=True)
            if task.created_by == PLUGIN_NAME
        ]
    finally:
        conn.close()
    assert recoveries == []

    state_path = kb.board_dir("default") / "plugin-state" / f"{PLUGIN_NAME}.sqlite3"
    with sqlite3.connect(state_path) as state:
        event = state.execute("SELECT action FROM recovery_events").fetchone()
        cap = state.execute("SELECT * FROM recovery_caps").fetchone()
    assert event == ("notify_only",)
    assert cap is None


@pytest.mark.parametrize(
    ("reason", "kind"),
    [
        ("Human approval is required before a rate limit change", "needs_input"),
        ("wait for the dependency", "dependency"),
    ],
)
def test_human_and_dependency_blocks_never_auto_create_recovery(
    isolated_home, reason, kind
):
    enable_plugin(
        isolated_home,
        enabled_boards=["default"],
        mode="safe_recovery",
        supervisor_profile="oink",
    )
    manager = PluginManager()
    manager.discover_and_load()
    callback = manager._hooks["kanban_task_blocked"][0]

    conn = kb.connect()
    try:
        source_id = kb.create_task(conn, title="Source", assignee="froink")
        kb.claim_task(conn, source_id)
        kb.block_task(conn, source_id, reason=reason, kind=kind)
    finally:
        conn.close()

    callback(task_id=source_id, board="default", run_id=2, reason=reason)

    conn = kb.connect()
    try:
        recoveries = [
            task for task in kb.list_tasks(conn, include_archived=True)
            if task.created_by == PLUGIN_NAME
        ]
    finally:
        conn.close()
    assert recoveries == []


def test_disabled_and_unknown_boards_are_ignored(isolated_home):
    enable_plugin(
        isolated_home,
        enabled_boards=[],
        mode="safe_recovery",
        supervisor_profile="oink",
    )
    manager = PluginManager()
    manager.discover_and_load()
    callback = manager._hooks["kanban_task_blocked"][0]

    conn = kb.connect()
    try:
        source_id = kb.create_task(
            conn,
            title="Disabled-board source",
            assignee="froink",
        )
        kb.claim_task(conn, source_id)
        kb.block_task(conn, source_id, reason="provider HTTP 503", kind="transient")
    finally:
        conn.close()

    callback(
        task_id=source_id,
        board="default",
        run_id=3,
        reason="provider HTTP 503",
    )
    callback(
        task_id=source_id,
        board="not-a-board",
        run_id=3,
        reason="provider HTTP 503",
    )

    conn = kb.connect()
    try:
        recoveries = [
            task for task in kb.list_tasks(conn, include_archived=True)
            if task.created_by == PLUGIN_NAME
        ]
    finally:
        conn.close()
    assert recoveries == []


def test_recovery_origin_is_never_recursively_supervised(isolated_home):
    enable_plugin(
        isolated_home,
        enabled_boards=["default"],
        mode="safe_recovery",
        supervisor_profile="oink",
    )
    manager = PluginManager()
    manager.discover_and_load()
    callback = manager._hooks["kanban_task_blocked"][0]

    conn = kb.connect()
    try:
        source_id = kb.create_task(
            conn,
            title="Recovery-origin source",
            assignee="oink",
            created_by=PLUGIN_NAME,
        )
        kb.claim_task(conn, source_id)
        kb.block_task(conn, source_id, reason="provider HTTP 503", kind="transient")
    finally:
        conn.close()

    callback(
        task_id=source_id,
        board="default",
        run_id=4,
        reason="provider HTTP 503",
    )

    conn = kb.connect()
    try:
        recoveries = [
            task for task in kb.list_tasks(conn, include_archived=True)
            if task.created_by == PLUGIN_NAME and task.id != source_id
        ]
    finally:
        conn.close()
    assert recoveries == []
