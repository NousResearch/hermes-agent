"""Useful-task receipt certainty and provider-free worker startup proof."""
from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from gateway.live_todo import rendered_payload_hash
from gateway.work_presentation import TrustedWorkAudience, audience_dict
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_cli import kanban_db_surface as receipts


def _task_and_receipt(path: Path):
    with closing(kbc.connect(path)) as conn:
        task_id = kb.create_task(conn, title="Synthetic task")
        source = receipts.get_task_source(conn, task_id)
        receipt = receipts.ensure_delivery_receipt(
            conn, task_id=task_id, task_incarnation=source.task_incarnation,
            desired_revision=source.current_revision, platform="telegram",
            chat_id="-100", thread_id="7", notifier_profile="default",
        )
        conn.execute(
            "UPDATE kanban_delivery_receipts SET state='sent', "
            "destination_message_id='701', destination_profile='default', "
            "delivered_revision=?, renderer_hash=? WHERE id=?",
            (source.current_revision, rendered_payload_hash("Original", []), receipt.id),
        )
        return task_id, receipt.id


def test_unknown_retry_failure_cannot_ack_old_confirmed_payload(tmp_path):
    path = tmp_path / "board.db"
    task_id, receipt_id = _task_and_receipt(path)
    old_hash = rendered_payload_hash("Original", [])
    with closing(kbc.connect(path)) as conn:
        assert kb.edit_task(conn, task_id, title="Changed once")
        revision = receipts.get_task_source(conn, task_id).current_revision
        receipts.ensure_delivery_receipt(
            conn, task_id=task_id, desired_revision=revision, platform="telegram",
            chat_id="-100", thread_id="7", notifier_profile="default",
        )
        lease = receipts.claim_delivery_receipt(
            conn, receipt_id, owner_id="writer", expected_revision=revision, refresh=True)
        receipts.record_delivery_outcome(
            conn, receipt_id, owner_id=lease.owner_id, owner_epoch=lease.owner_epoch,
            attempt_id=lease.attempt_id, desired_revision=lease.desired_revision,
            state="unknown", error="unconfirmed dispatch")
        uncertain = receipts.get_delivery_receipt(conn, receipt_id)
        assert uncertain is not None and uncertain.renderer_hash is None

        retry = receipts.claim_delivery_receipt(
            conn, receipt_id, owner_id="writer-2", expected_revision=revision, refresh=True)
        receipts.record_delivery_outcome(
            conn, receipt_id, owner_id=retry.owner_id, owner_epoch=retry.owner_epoch,
            attempt_id=retry.attempt_id, desired_revision=retry.desired_revision,
            state="failed", retry_disposition="safe_retry",
            error="operation failed before request write")
        failed = receipts.get_delivery_receipt(conn, receipt_id)
        assert failed is not None and failed.state == "failed" and failed.renderer_hash is None
        with pytest.raises(receipts.DeliveryReceiptNotDue):
            receipts.confirm_equivalent_delivery(
                conn, receipt_id, desired_revision=revision,
                renderer_hash=old_hash, message_id="701")


def test_new_revision_recovers_only_exhausted_known_unchanged_edit(tmp_path):
    path = tmp_path / "board.db"
    task_id, receipt_id = _task_and_receipt(path)
    with closing(kbc.connect(path)) as conn:
        assert kb.edit_task(conn, task_id, title="Unchanged rejection")
        exhausted_revision = receipts.get_task_source(conn, task_id).current_revision
        receipts.ensure_delivery_receipt(
            conn, task_id=task_id, desired_revision=exhausted_revision,
            platform="telegram", chat_id="-100", thread_id="7",
            notifier_profile="default")
        conn.execute(
            "UPDATE kanban_delivery_receipts SET state='failed', failure_count=1, "
            "retry_disposition='exhausted', last_error='known message unchanged', "
            "attempted_revision=? WHERE id=?",
            (exhausted_revision, receipt_id),
        )
        with pytest.raises(receipts.DeliveryReceiptNotRetryable):
            receipts.claim_delivery_receipt(
                conn, receipt_id, owner_id="same-revision",
                expected_revision=exhausted_revision, refresh=True)

        assert kb.edit_task(conn, task_id, title="Later changed payload")
        later_revision = receipts.get_task_source(conn, task_id).current_revision
        receipts.ensure_delivery_receipt(
            conn, task_id=task_id, desired_revision=later_revision,
            platform="telegram", chat_id="-100", thread_id="7",
            notifier_profile="default")
        lease = receipts.claim_delivery_receipt(
            conn, receipt_id, owner_id="later-revision",
            expected_revision=later_revision, refresh=True)
        assert lease.desired_revision == later_revision
        assert lease.receipt.failure_count == 0


def test_old_receipt_reopen_adds_no_legacy_action_authority(tmp_path):
    path = tmp_path / "old.db"
    with closing(kbc.connect(path)) as current:
        task_id = kb.create_task(current, title="Legacy synthetic task")
        source = receipts.get_task_source(current, task_id)
        receipt = receipts.ensure_delivery_receipt(
            current, task_id=task_id, task_incarnation=source.task_incarnation,
            desired_revision=source.current_revision, platform="telegram",
            chat_id="-100", thread_id="7", notifier_profile="default")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("ALTER TABLE kanban_delivery_receipts DROP COLUMN control_hash")
        conn.execute(
            "UPDATE kanban_delivery_receipts SET state='sent', destination_message_id='701', "
            "delivered_revision=desired_revision, renderer_hash='legacy-token-hash' WHERE id=?",
            (receipt.id,),
        )
        receipts.migrate_delivery_receipts(conn)
        reopened = receipts.get_delivery_receipt(conn, receipt.id)
        assert reopened is not None
        assert reopened.renderer_hash == "legacy-token-hash"
        assert reopened.control_hash is None
    finally:
        conn.close()


def _worker_child() -> int:
    from hermes_cli.main import _prepare_agent_startup
    from hermes_cli.plugins import discover_plugins, get_plugin_manager

    args = SimpleNamespace(
        command="chat", tui=False, cli=True, yolo=False, safe_mode=False,
        ignore_user_config=False, query=None, oneshot=None, toolsets="kanban",
        accept_hooks=False,
    )
    _prepare_agent_startup(args)
    discover_plugins()
    manager = get_plugin_manager()
    service = getattr(manager, "_work_presentation_registration", None)
    if service is None or not service.active:
        return 11

    # Importing the native tools is the same synchronous registry boundary the
    # model-tool loader reaches after startup; no model or provider is invoked.
    from tools import kanban_tools  # noqa: F401
    from tools.registry import registry
    result = registry.dispatch(
        "kanban_block",
        {
            "reason": "Synthetic cancellation preference required",
            "kind": "needs_input",
            "presentation": {
                "title": "Synthetic task paused",
                "summary": "Work paused at the requested decision point.",
                "blocker": "Choose the synthetic cancellation preference.",
            },
        }, scope=manager.scope_key,
    )
    payload = json.loads(result)
    return 0 if payload.get("ok") is True and payload.get("status") == "blocked" else 12


def test_dispatcher_claimed_child_loads_plugin_and_publishes_blocker(tmp_path):
    pytest.importorskip("hermes_telegram_experience")
    home = tmp_path / "profile"
    home.mkdir()
    db = tmp_path / "board.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    route = dict(profile="default", platform="telegram", chat_id="-100", thread_id="7")
    config = {
        "plugins": {
            "enabled": ["hermes-telegram-experience"],
            "entries": {"hermes-telegram-experience": {"settings": {
                "enabled": True, "work_briefs": True,
                "scope": {"routes": [route], "task_resources": []},
            }}},
        },
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    audience = TrustedWorkAudience("default", "telegram", 123, "-100", "7", 42)
    publication = {
        "version": 1, "audience": audience_dict(audience),
        "presentation": {"title": "Synthetic task", "summary": "Queued work."},
        "steps": [], "run_id": None,
    }
    with closing(kbc.connect(db)) as conn:
        task_id = kb.create_task(
            conn, title="Synthetic task", assignee="default",
            workspace_kind="scratch", workspace_path=str(workspace), publication=publication)
        task = kb.claim_task(conn, task_id, claimer="synthetic-dispatcher")
        assert task is not None and task.current_run_id is not None and task.claim_lock

    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "HERMES_HOME": str(home),
        "HERMES_PROFILE": "default",
        "HERMES_KANBAN_DB": str(db),
        "HERMES_KANBAN_BOARD": "default",
        "HERMES_KANBAN_TASK": task_id,
        "HERMES_KANBAN_RUN_ID": str(task.current_run_id),
        "HERMES_KANBAN_CLAIM_LOCK": task.claim_lock,
        "HERMES_KANBAN_WORKSPACE": str(workspace),
        "TERMINAL_CWD": str(workspace),
        "HERMES_SESSION_SOURCE": "kanban",
    }
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-child"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr[-1000:]
    with closing(kbc.connect(db)) as conn:
        blocked = kb.get_task(conn, task_id)
        assert blocked is not None and blocked.status == "blocked"
        row = conn.execute(
            "SELECT run_id,payload FROM task_events WHERE task_id=? AND kind='blocked' "
            "ORDER BY id DESC LIMIT 1", (task_id,),
        ).fetchone()
        assert row is not None and row["run_id"] == task.current_run_id
        published = json.loads(row["payload"])["publication"]
        assert published["run_id"] == task.current_run_id
        assert published["presentation"]["blocker"] == "Choose the synthetic cancellation preference."


if __name__ == "__main__" and sys.argv[1:] == ["--worker-child"]:
    raise SystemExit(_worker_child())
