"""K2 mechanisms isolated: migration adoption and replay re-assert each carry cases the other cannot."""
import json, sqlite3
from pathlib import Path
import pytest
from cron import executions as ex, jobs, delivery_queue as dq, bot_chat_delivery as bot
from tests.cron.test_r5_salt_legacy_and_races import store, seed, receipt, set_receipt, snap


def _reject_bot_manifest(conn):
    conn.execute("CREATE TRIGGER k2_reject BEFORE UPDATE OF delivery_manifest ON executions "
                 "WHEN instr(NEW.delivery_manifest, '\"bot\"')>0 BEGIN SELECT RAISE(ABORT,'k2 manifest fault'); END")


def test_replay_reasserts_intent_on_already_migrated_row(store):
    """Schema already has the column (pending=0 default) and a journal entry appears later
    (e.g. copied from a backup, or a mark whose re-assert failed). Migration adoption cannot
    run again; only the replay's own intent assertion fences the placeholder."""
    eid = seed(store)                      # current schema: pending column exists, value 0
    manifest = receipt(store, eid)
    ex.journal_manifest(eid, manifest)     # journal written; simulate the flag being lost
    with ex._transaction() as conn:
        conn.execute("UPDATE executions SET delivery_manifest_pending=0 WHERE id=?", (eid,))
        _reject_bot_manifest(conn)
    ex.reconcile_delivery_projections()
    state = snap(eid)
    assert state['execution']['delivery_outcome'] == 'queued'
    assert state['execution']['delivery_manifest_pending'] == 1
    assert state['execution']['delivery_projection_settled'] == 0


def test_migration_adopts_journal_before_any_reader_even_without_replay(store, monkeypatch):
    """Old schema + outstanding journal. A reader that touches the row BEFORE any reconcile
    (finish_execution of a late worker, or prune) must already see the intent: the migration
    step, not the replay, is what fences it."""
    from tests.cron.test_r5_salt_legacy_and_races import prior
    old = prior(); eid = seed(store, old); manifest = receipt(store, eid)
    old.journal_manifest(eid, manifest)
    with sqlite3.connect(old.EXECUTIONS_FILE) as conn:
        assert 'delivery_manifest_pending' not in [r[1] for r in conn.execute('PRAGMA table_info(executions)')]
    # Make replay a no-op so only migration adoption can fence the row.
    monkeypatch.setattr(ex, '_recover_unrecorded_manifests', lambda: {})
    with ex._transaction() as conn:   # triggers schema init (migration) on the current module
        pass
    row = ex.get_execution(eid)
    assert row['delivery_manifest_pending'] == 1
    monkeypatch.setattr(ex, 'MAX_TERMINAL_EXECUTIONS', 0)
    with ex._transaction() as conn:
        ex._prune_unlocked(conn)
    assert ex.get_execution(eid) is not None
    ex.reconcile_delivery_projections()
    assert ex.get_execution(eid)['delivery_outcome'] == 'queued'


def test_finish_holds_placeholder_row_regardless_of_caller_outcome_while_adoption_incomplete(store, monkeypatch):
    """K2a finish path isolated: the caller says 'failed' (its own classification) for a legacy
    placeholder-only row while adoption is incomplete. finish must write queued, not trust the
    caller — the projection fence alone cannot do this because a None projection keeps the
    caller's value."""
    from tests.cron.test_r6_salt_migration_authority import legacy_running
    eid, manifest = legacy_running(store)
    root = store / 'cron' / 'manifest_journal'
    mode = root.stat().st_mode & 0o777
    root.chmod(0)
    try:
        ex.get_execution(eid)  # initialization: adoption fails, fence row absent
        assert not ex._legacy_intent_adopted()
        finished = ex.finish_execution(eid, success=False, error='worker failure', delivery_outcome='failed')
    finally:
        root.chmod(mode)
    assert finished['delivery_outcome'] == 'queued'
    ex.reconcile_delivery_projections()
    after = ex.get_execution(eid)
    assert after['delivery_outcome'] == 'queued' and '"bot"' in after['delivery_manifest']
