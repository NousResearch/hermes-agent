"""Salt R4 closure: the pre-send intent flag is the authoritative fence on every reader.

Each cell here kills one specific mutant that the ported J1-J4 probes do not:
pre-send mark when the journal is also unwritable; prune predicate; reconcile CAS re-check.
"""
import json, threading
from cron import executions, scheduler, jobs, bot_chat_delivery
from tests.cron.test_mixed_delivery_retention import setup
from tests.cron.test_r4_salt_journal_edges import reject_manifest, repaired, fast_gateway
from tests.cron.test_r4_salt_recovery_probes import restart


def test_presend_intent_alone_keeps_queued_when_ledger_and_journal_both_fault(setup, monkeypatch):
    """Ledger rejects the manifest AND the journal is entirely unavailable (its re-assert of the
    flag included): only the PRE-SEND flag can keep the run queued. Native sent once; no
    transport failure reported."""
    home, cfg, run, sent = setup
    reject_manifest()
    monkeypatch.setattr(executions, 'journal_manifest', lambda eid, manifest: False)
    root = home / 'cron' / 'manifest_journal'; root.mkdir(); root.chmod(0o500)
    try:
        job, row = run(False)
    finally:
        root.chmod(0o700)
    assert len(sent) == 1
    assert row['delivery_outcome'] == 'queued'
    assert row['delivery_manifest_pending'] == 1
    assert jobs.get_job(job['id']).get('last_delivery_queued')
    assert not jobs.get_job(job['id']).get('last_delivery_error')
    # Retention pressure with nothing else protecting the row.
    monkeypatch.setattr(executions, 'MAX_TERMINAL_EXECUTIONS', 0)
    with executions._transaction() as conn:
        executions._prune_unlocked(conn)
    assert executions.get_execution(row['id']) is not None


def test_prune_predicate_honours_pending_even_when_outcome_is_not_queued(setup, monkeypatch):
    """Direct-DB corruption of the outcome must not make a pending row prunable."""
    home, cfg, run, sent = setup
    reject_manifest()
    job, row = run(False)
    with executions._transaction() as conn:
        conn.execute("UPDATE executions SET delivery_outcome='delivered' WHERE id=?", (row['id'],))
    monkeypatch.setattr(executions, 'MAX_TERMINAL_EXECUTIONS', 0)
    with executions._transaction() as conn:
        executions._prune_unlocked(conn)
    assert executions.get_execution(row['id']) is not None


def test_reconcile_cas_rechecks_pending_at_write_time(setup, monkeypatch):
    """A reconciler that loaded the row BEFORE the pending flag flipped must not settle it."""
    home, cfg, run, sent = setup
    fast_gateway(monkeypatch)
    job, row = run(True)  # healthy: external placeholder + real bot receipt, manifest recorded
    assert row['delivery_outcome'] == 'queued'
    # Stale reader: its projection was computed from a snapshot with pending=0 and returns
    # 'delivered' unconditionally (no projection-time recheck); intent flips underneath it
    # before the write. Only the CAS predicate can refuse the settle.
    def stale(record):
        with executions._transaction() as conn:
            conn.execute("UPDATE executions SET delivery_manifest_pending=1 WHERE id=?", (record['id'],))
        return ('delivered', {'last_delivery_queued': None, 'last_delivery_unverified': None, 'last_delivery_error': None})
    monkeypatch.setattr(executions, '_delivery_projection', stale)
    executions.reconcile_delivery_projections()
    after = executions.get_execution(row['id'])
    assert after['delivery_outcome'] == 'queued'
    assert after['delivery_projection_settled'] == 0
    assert jobs.get_job(job['id']).get('last_delivery_queued')
