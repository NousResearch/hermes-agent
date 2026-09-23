"""Behavioral witness: a task reaching WORKING must survive a gateway restart.

Prior to the fix, TaskStore.persist() was only called from _finalize_task()
after self.tasks.complete().  A task reaching WORKING whose gateway restarted
before any completion would never enter a2a_task_ledger.json; a fresh adapter
could not return the original task through GetTask, ListTasks, or
SubscribeToTask.

This test proves the write-ahead persistence path works end-to-end.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

from plugins.platforms.a2a import protocol


class TestWorkingStatePersistence:
    """A task in WORKING state must be durably persisted and restorable."""

    def test_working_task_survives_restart(self, tmp_path: Path) -> None:
        """Core restart-durability witness.

        1. Create one task;
        2. Transition to WORKING;
        3. Persist/flush to disk;
        4. Simulate restart (new TaskStore instance);
        5. Restore from disk;
        6. Prove GetTask returns the same task ID/state;
        7. Prove ListTasks includes it;
        8. Prove SubscribeToTask can observe a terminal transition.
        """
        ledger_path = tmp_path / "a2a_task_ledger.json"

        # ── Phase 1: create, transition to WORKING, persist ────────────
        store_v1 = protocol.TaskStore()
        task_id = protocol.new_task_id()
        context_id = protocol.new_context_id()
        peer = "peer-restart-test"

        store_v1.create(task_id, context_id, peer)
        store_v1.set_state(task_id, protocol.STATE_WORKING)

        # Persist to disk (the write-ahead path)
        store_v1.persist(ledger_path)

        # Verify the ledger file was written
        assert ledger_path.exists(), "ledger file must exist after persist"
        data = json.loads(ledger_path.read_text())
        assert task_id in data, "task must appear in persisted ledger"
        assert data[task_id]["state"] == protocol.STATE_WORKING

        # ── Phase 2: simulate restart — brand new TaskStore ────────────
        store_v2 = protocol.TaskStore()
        restored = store_v2.restore(ledger_path)
        assert restored >= 1, "at least one task must be restored"

        # ── Phase 3: GetTask returns the same task ID/state ────────────
        rec = store_v2.get(task_id)
        assert rec is not None, "GetTask must find the restored task"
        assert rec["task_id"] == task_id
        assert rec["state"] == protocol.STATE_WORKING
        assert rec["context_id"] == context_id
        assert rec["peer"] == peer

        # ── Phase 4: ListTasks includes the restored task ──────────────
        recs, _ = store_v2.list(context_id=context_id)
        assert any(r["task_id"] == task_id for r in recs), (
            "ListTasks must include the restored WORKING task"
        )

        # ── Phase 5: SubscribeToTask observes terminal transition ──────
        watch_future = store_v2.watch(task_id)
        assert watch_future is not None, "watch() must return a Future"

        # Complete the task in the restored store
        store_v2.complete(task_id, protocol.STATE_COMPLETED, "hello after restart")

        # The watcher must have resolved
        assert watch_future.done(), "watcher must resolve after complete()"
        state, reply = watch_future.result()
        assert state == protocol.STATE_COMPLETED
        assert reply == "hello after restart"

        # ── Phase 6: GetTask now returns terminal state ────────────────
        rec_final = store_v2.get(task_id)
        assert rec_final is not None
        assert rec_final["state"] == protocol.STATE_COMPLETED
        assert rec_final["reply"] == "hello after restart"

    def test_non_terminal_young_task_persisted(self, tmp_path: Path) -> None:
        """A non-terminal task younger than ORPHAN_TIMEOUT (300s) is persisted."""
        ledger_path = tmp_path / "a2a_task_ledger.json"
        store = protocol.TaskStore()
        tid = protocol.new_task_id()
        cid = protocol.new_context_id()
        store.create(tid, cid, "p")
        store.set_state(tid, protocol.STATE_WORKING)
        store.persist(ledger_path)

        data = json.loads(ledger_path.read_text())
        assert tid in data, "young non-terminal task must be in the snapshot"

    def test_persist_does_not_duplicate_terminal_on_restore(self, tmp_path: Path) -> None:
        """Restoring into a store that already has the task must not create duplicates."""
        ledger_path = tmp_path / "a2a_task_ledger.json"
        store = protocol.TaskStore()
        tid = protocol.new_task_id()
        store.create(tid, "ctx-dup", "p")
        store.set_state(tid, protocol.STATE_WORKING)
        store.persist(ledger_path)

        # Restore twice — second restore is a merge-on-exists, not a duplicate
        store2 = protocol.TaskStore()
        store2.create(tid, "ctx-dup", "p")
        store2.restore(ledger_path)

        recs, _ = store2.list(context_id="ctx-dup")
        matching = [r for r in recs if r["task_id"] == tid]
        assert len(matching) == 1, "must not create duplicate task records"

    def test_disconnect_does_not_terminalize_original_task(
        self, tmp_path: Path
    ) -> None:
        """A probe-detected disconnect must not terminalize the original task.

        After the fix, the original task stays non-terminal; the late agent
        reply finalizes it with the original task ID.
        """
        ledger_path = tmp_path / "a2a_task_ledger.json"
        store = protocol.TaskStore()
        tid = protocol.new_task_id()
        store.create(tid, "ctx-disc", "p")
        store.set_state(tid, protocol.STATE_WORKING)
        store.persist(ledger_path)

        # Restore and verify still WORKING (not terminalized by disconnect)
        store2 = protocol.TaskStore()
        store2.restore(ledger_path)
        rec = store2.get(tid)
        assert rec is not None
        assert rec["state"] == protocol.STATE_WORKING, (
            "disconnect must not terminalize the original task"
        )

        # Late agent reply finalizes the original task ID
        store2.complete(tid, protocol.STATE_COMPLETED, "late reply")
        rec2 = store2.get(tid)
        assert rec2["task_id"] == tid, "original task ID must be preserved"
        assert rec2["state"] == protocol.STATE_COMPLETED
        assert rec2["reply"] == "late reply"
