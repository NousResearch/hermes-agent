"""Tests for the kanban notifier zero-subscription early exit.

The notifier used to writable-open EVERY board DB on every tick even when a
board had zero subscriptions — paying schema init/migration on first open,
WAL/-shm sidecar creation, and checkpoint traffic for boards with nothing to
notify. Per-board work is now gated by a read-only subscription probe
(``kanban_db.count_notify_subs``), so boards with zero subscriptions are
never opened writable.

(The companion machine-global ``.notifier.lock`` singleton gate from PR
#63001 was deliberately NOT salvaged: a lock-winning default-profile gateway
cannot deliver a secondary profile's subscriptions in standalone-profile
deployments — profile routing fails closed in
``gateway/authz_mixin.py::_authorization_adapter`` — so the lock could
suppress delivery entirely. The read-only probe captures the per-tick cost
win without that risk.)
"""

import asyncio
import sqlite3

from unittest.mock import patch

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    return runner


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _create_completed_task(*, subscribe: bool) -> str:
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="owner gate", assignee="worker")
        if subscribe:
            kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary="done")
        return tid
    finally:
        conn.close()


def test_zero_sub_board_is_never_opened_writable(tmp_path, monkeypatch):
    """A board with zero subscriptions must be skipped BEFORE `_kb.connect`."""
    db_path = tmp_path / "zero-subs.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    _create_completed_task(subscribe=False)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    with patch.object(kbc, "connect", wraps=kbc.connect) as spy_connect:
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    spy_connect.assert_not_called()
    assert adapter.sent == []


def test_foreign_workflow_only_board_is_never_opened_writable(tmp_path, monkeypatch):
    """A workflow sub owned by another notifier profile is not eligible work."""
    db_path = tmp_path / "foreign-workflow-sub.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kbc.connect()
    try:
        acceptance = kb.create_task(conn, title="accept", tenant="tenant-a")
        actor = kb.KanbanActorContext(
            principal_id="svc:test", profile_name="default",
            board_identity=str(kb.kanban_db_path().resolve()), tenant="tenant-a",
            capabilities=frozenset({"workflow.manage", "workflow.admin"}),
            source_kind="test",
        )
        kb.create_workflow(
            conn, workflow_id="wf_foreign", name="release", tenant="tenant-a",
            designated_acceptance_task_id=acceptance, actor=actor, mutation_id="create",
        )
        conn.execute(
            "INSERT INTO kanban_workflow_subscriptions "
            "(workflow_id,role,platform,chat_id,notifier_profile,target_states,tenant,created_at) "
            "VALUES ('wf_foreign','origin','telegram','foreign-chat','other','[\"PASS\"]','tenant-a',0)",
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    with patch.object(kbc, "connect", wraps=kbc.connect) as spy_connect:
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    spy_connect.assert_not_called()
    assert adapter.sent == []


def test_workflow_probe_error_skips_board_without_writable_open(tmp_path, monkeypatch):
    """An unreadable workflow-subscription probe must fail closed for the tick."""
    db_path = tmp_path / "workflow-probe-error.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    with (
        patch.object(kbn, "count_workflow_subs", side_effect=sqlite3.DatabaseError("broken")),
        patch.object(kbc, "connect", wraps=kbc.connect) as spy_connect,
    ):
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    spy_connect.assert_not_called()
    assert adapter.sent == []


def test_workflow_only_board_opens_once_and_delivers(tmp_path, monkeypatch):
    """An eligible workflow sub must bypass only the task-subscription gate."""
    db_path = tmp_path / "workflow-only.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kbc.connect()
    try:
        acceptance = kb.create_task(conn, title="accept", tenant="tenant-a")
        actor = kb.KanbanActorContext(
            principal_id="svc:test", profile_name="default",
            board_identity=str(kb.kanban_db_path().resolve()), tenant="tenant-a",
            capabilities=frozenset({"workflow.manage", "workflow.admin"}),
            source_kind="test",
        )
        created = kb.create_workflow(
            conn, workflow_id="wf_only", name="release", tenant="tenant-a",
            designated_acceptance_task_id=acceptance, actor=actor, mutation_id="create",
        )
        conn.execute(
            "INSERT INTO kanban_workflow_subscriptions "
            "(workflow_id,role,platform,chat_id,notifier_profile,target_states,tenant,created_at,last_event_id) "
            "VALUES ('wf_only','origin','telegram','workflow-chat','default','[\"CANCELLED\"]','tenant-a',0,?)",
            (created["workflow"]["last_event_id"],),
        )
        kb.cancel_workflow(
            conn, workflow_id="wf_only", actor=actor, mutation_id="cancel",
            expected_version=created["workflow"]["version"], reason="superseded",
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    with patch.object(kbc, "connect", wraps=kbc.connect) as spy_connect:
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Collection opens once; successful delivery advances the cursor through
    # its own connection.
    assert spy_connect.call_count == 2
    assert [item["chat_id"] for item in adapter.sent] == ["workflow-chat"]


def test_task_only_board_opens_once_for_task_collection(tmp_path, monkeypatch):
    """An eligible task sub still collects while workflow probing stays read-only."""
    db_path = tmp_path / "task-only.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    _create_completed_task(subscribe=True)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._kanban_dispatcher_lock_handle = object()

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert [item["chat_id"] for item in adapter.sent] == ["chat-1"]
