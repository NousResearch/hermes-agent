"""K2c layers isolated. Salt R7's late-writer schedules are recovered by any one of three layers
(finish accounting, projection accounting, replay reopen); each cell here disables the others so
a single missing layer fails."""
import json, sqlite3
from cron import executions as ex, jobs
from tests.cron.test_r5_salt_legacy_and_races import store, prior, receipt, snap
from tests.cron.test_r6_salt_migration_authority import legacy_running


def _adopted_with_no_journal(store):
    """Fence committed while the legacy row's journal did not exist yet (the K2c window)."""
    eid, manifest = legacy_running(store)
    (ex._journal_root() / f"{eid}.json").unlink()
    assert ex.get_execution(eid)["delivery_manifest_pending"] == 0 and ex._legacy_intent_adopted()
    old = prior(); old.EXECUTIONS_FILE = ex.EXECUTIONS_FILE
    return eid, manifest, old


def test_finish_accounts_for_late_journal_without_replay(store, monkeypatch):
    eid, manifest, old = _adopted_with_no_journal(store)
    old.journal_manifest(eid, manifest)                      # pre-flag writer publishes late
    monkeypatch.setattr(ex, "_recover_unrecorded_manifests", lambda: {})  # no replay layer
    finished = ex.finish_execution(eid, success=True, delivery_outcome="queued")
    assert finished["delivery_outcome"] == "queued"
    assert finished["delivery_manifest_pending"] == 1        # intent adopted in finish's transaction


def test_projection_accounts_for_late_journal_without_replay(store, monkeypatch):
    eid, manifest, old = _adopted_with_no_journal(store)
    # Row is terminal-queued with only the placeholder: its external queue child is still
    # pending when the OLD module finishes it (healthy legacy contract), then the old writer
    # publishes the journal. The reconciler must not settle the placeholder as delivered.
    from cron import delivery_queue as dq
    with dq._transaction() as conn:
        conn.execute("UPDATE deliveries SET status='pending', finished_at=NULL WHERE execution_id=?", (eid,))
    with old._transaction() as conn:
        conn.execute("UPDATE executions SET process_id=? WHERE id=?", (old._PROCESS_ID, eid))
    assert old.finish_execution(eid, success=True, delivery_outcome="queued")["delivery_outcome"] == "queued"
    old.journal_manifest(eid, manifest)
    with dq._transaction() as conn:
        conn.execute("UPDATE deliveries SET status='delivered', finished_at=? WHERE execution_id=?",
                     (dq._hermes_now().isoformat(), eid))
    monkeypatch.setattr(ex, "_recover_unrecorded_manifests", lambda: {})
    ex.reconcile_delivery_projections()
    row = ex.get_execution(eid)
    assert row["delivery_outcome"] == "queued", "placeholder settled as delivered although its journal exists"
    assert row["delivery_projection_settled"] == 0


def test_replay_reopens_row_terminalized_by_old_code(store):
    """The OLD module (no accounting layers) finished the row as delivered from the placeholder,
    then published its journal. Only the replay's reopen can correct the ledger."""
    eid, manifest, old = _adopted_with_no_journal(store)
    with old._transaction() as conn:
        conn.execute("UPDATE executions SET process_id=? WHERE id=?", (old._PROCESS_ID, eid))
    finished = old.finish_execution(eid, success=True, delivery_outcome="queued")
    assert finished["delivery_outcome"] == "delivered"     # the pre-flag defect, reproduced
    old.journal_manifest(eid, manifest)
    ex.reconcile_delivery_projections()
    row = ex.get_execution(eid)
    assert row["delivery_outcome"] == "queued"
    assert '"bot"' in row["delivery_manifest"]
    assert jobs.get_job("child-job")["last_delivery_queued"]


def test_unknowable_journal_state_holds_after_adoption(store):
    """Adoption complete, journal directory later unreadable: a placeholder-only row cannot be
    proven journal-free, so it is held rather than terminalized."""
    eid, manifest, old = _adopted_with_no_journal(store)
    root = ex._journal_root(); mode = root.stat().st_mode & 0o777
    root.chmod(0)
    try:
        finished = ex.finish_execution(eid, success=True, delivery_outcome="queued")
    finally:
        root.chmod(mode)
    assert finished["delivery_outcome"] == "queued"
    assert finished["delivery_manifest_pending"] == 0      # nothing adopted: state was unknown, not present
