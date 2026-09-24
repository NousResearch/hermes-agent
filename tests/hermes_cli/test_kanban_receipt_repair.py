"""Exact transaction-boundary and once-per-attempt accounting regressions."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sys
from threading import Barrier

import pytest
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_cli import kanban_db_surface as s


def seed(path):
    with closing(kbc.connect(path)) as c:
        tid = kb.create_task(c, title='Synthetic receipt boundary')
        rev = s.get_task_source(c, tid).current_revision
        row = s.ensure_delivery_receipt(c, task_id=tid, platform='telegram', chat_id='-100',
                                        notifier_profile='default', desired_revision=rev)
    return tid, row


def evidence(lease, state='sent'):
    args = dict(owner_id=lease.owner_id, owner_epoch=lease.owner_epoch,
                attempt_id=lease.attempt_id, desired_revision=lease.desired_revision, state=state)
    if state == 'sent':
        args.update(message_id='701', destination_profile='default', delivered_revision=lease.desired_revision)
    elif state == 'failed':
        args.update(retry_disposition='safe_retry')
    return args


@pytest.mark.parametrize('late', [False, True])
def test_claim_commit_boundary_freezes_exact_receipt(tmp_path, late):
    path = tmp_path/'board.db'
    tid, row = seed(path)
    fired = []
    def compete():
        with closing(kbc.connect(path)) as other:
            kb.add_comment(other, tid, 'synthetic', 'new demand in commit-return gap')
            rev = s.get_task_source(other, tid).current_revision
            s.ensure_delivery_receipt(other, task_id=tid, platform='telegram', chat_id='-100',
                                      notifier_profile='default', desired_revision=rev)
            with pytest.raises(s.DeliveryReceiptLeaseLost):
                s.claim_delivery_receipt(other, row.id, owner_id='competitor', now=101)
            return rev
    with closing(kbc.connect(path)) as c, ThreadPoolExecutor(max_workers=1) as pool:
        # Independent boundary control: context-manager exit AFTER the actual
        # COMMIT, before claim resumes; no source-line number or result replacement.
        def trace(frame, event, arg):
            if (not fired and event == 'return' and arg is None
                    and frame.f_code is getattr(s._write_scope, '__wrapped__').__code__):
                assert not c.in_transaction
                fired.append(pool.submit(compete).result(timeout=8))
            return trace
        previous = sys.gettrace()
        sys.settrace(trace)
        try:
            lease = s.claim_delivery_receipt(c, row.id, owner_id='first', now=100, lease_seconds=5)
        finally:
            sys.settrace(previous)
        assert len(fired) == 1 and fired[0] > row.desired_revision
        assert lease.desired_revision == lease.receipt.attempted_revision == row.desired_revision
        assert lease.receipt.desired_revision == row.desired_revision
        if late:
            with pytest.raises(s.DeliveryReceiptUnknown):
                s.claim_delivery_receipt(c, row.id, owner_id='expired', now=106)
        result = s.reconcile_delivery_outcome(c, row.id, **evidence(lease), now=106 if late else 101)
        assert result.destination_message_id == '701'
        assert result.delivered_revision == row.desired_revision and result.desired_revision == fired[0]
        successor = s.claim_delivery_receipt(c, row.id, owner_id='next', now=107)
        assert successor.receipt.destination_message_id == '701' and successor.desired_revision == fired[0]


@pytest.mark.parametrize('quarantine', [False, True])
@pytest.mark.parametrize('late_state', ['unknown', 'failed', 'sent'])
def test_unresolved_attempt_charge_once_and_stale_evidence_fenced(tmp_path, quarantine, late_state):
    path = tmp_path/'board.db'
    tid, row = seed(path)
    with closing(kbc.connect(path)) as c:
        first = s.claim_delivery_receipt(c, row.id, owner_id='create', now=100)
        s.reconcile_delivery_outcome(c, row.id, **evidence(first), now=101)
        kb.add_comment(c, tid, 'synthetic', 'edit')
        rev = s.get_task_source(c, tid).current_revision
        s.ensure_delivery_receipt(c, task_id=tid, platform='telegram', chat_id='-100',
                                  notifier_profile='default', desired_revision=rev)
        lease = s.claim_delivery_receipt(c, row.id, owner_id='edit', now=102, lease_seconds=1)
        if quarantine:
            s.quarantine_delivery(c, lease)
            s.quarantine_delivery(c, lease)
        else:
            with pytest.raises(s.DeliveryReceiptUnknown):
                s.claim_delivery_receipt(c, row.id, owner_id='expire', now=104)
        charged = s.get_delivery_receipt(c, row.id)
        assert charged is not None and charged.failure_count == 1
        settled = s.reconcile_delivery_outcome(c, row.id, **evidence(lease, late_state), now=105)
        assert settled.failure_count == (0 if late_state == 'sent' else 1)
        if late_state == 'failed':
            with pytest.raises(s.DeliveryReceiptLeaseLost):
                s.reconcile_delivery_outcome(c, row.id, **evidence(lease, late_state), now=105)
        else:
            assert s.reconcile_delivery_outcome(c, row.id, **evidence(lease, late_state), now=105) == settled
        kb.add_comment(c, tid, 'synthetic', 'successor demand')
        rev = s.get_task_source(c, tid).current_revision
        s.ensure_delivery_receipt(c, task_id=tid, platform='telegram', chat_id='-100',
                                  notifier_profile='default', desired_revision=rev)
    # Reopen, then acquire a successor. Neither stale success nor failure may
    # reset/charge its budget, and the old writer never regains admission.
    with closing(kbc.connect(path)) as c:
        successor = s.claim_delivery_receipt(c, row.id, owner_id='next', now=106)
        before = s.get_delivery_receipt(c, row.id)
        for state in ('sent', 'failed', 'unknown'):
            with pytest.raises(s.DeliveryReceiptLeaseLost):
                s.reconcile_delivery_outcome(c, row.id, **evidence(lease, state), now=107)
            assert s.get_delivery_receipt(c, row.id) == before
        assert successor.receipt.destination_message_id == '701'


@pytest.mark.parametrize('outcome', ['unknown', 'sent'])
def test_expiry_and_late_settlement_transaction_race(tmp_path, outcome):
    path = tmp_path/'board.db'
    _, row = seed(path)
    with closing(kbc.connect(path)) as c:
        lease = s.claim_delivery_receipt(c, row.id, owner_id='first', now=100, lease_seconds=1)
    barrier = Barrier(2)
    def expire():
        with closing(kbc.connect(path)) as c:
            barrier.wait(timeout=8)
            with pytest.raises(s.DeliveryReceiptError):
                s.claim_delivery_receipt(c, row.id, owner_id='expire', now=102)
    def settle():
        with closing(kbc.connect(path)) as c:
            barrier.wait(timeout=8)
            return s.reconcile_delivery_outcome(c, row.id, **evidence(lease, outcome), now=102)
    with ThreadPoolExecutor(max_workers=2) as pool:
        expiry = pool.submit(expire); settlement = pool.submit(settle)
        expiry.result(timeout=8); settlement.result(timeout=8)
    with closing(kbc.connect(path)) as c:
        result = s.get_delivery_receipt(c, row.id)
        assert result is not None
        assert result.failure_count == (1 if outcome == 'unknown' else 0)
        assert result.state == outcome
        duplicate = s.reconcile_delivery_outcome(c, row.id, **evidence(lease, outcome), now=103)
        assert duplicate.failure_count == result.failure_count
        assert duplicate.state == result.state and duplicate.attempt_id == result.attempt_id


def test_exhausted_crash_budget_survives_ensure_new_demand_and_reopen(tmp_path):
    path = tmp_path/'board.db'
    tid, row = seed(path)
    with closing(kbc.connect(path)) as c:
        first = s.claim_delivery_receipt(c, row.id, owner_id='create', now=100)
        s.reconcile_delivery_outcome(c, row.id, **evidence(first), now=101)
        kb.add_comment(c, tid, 'synthetic', 'edit demand')
        rev = s.get_task_source(c, tid).current_revision
        s.ensure_delivery_receipt(c, task_id=tid, platform='telegram', chat_id='-100',
                                  notifier_profile='default', desired_revision=rev)
    lease = None
    for i in range(3):
        with closing(kbc.connect(path)) as c:
            lease = s.claim_delivery_receipt(c, row.id, owner_id=f'edit-{i}', now=110+i*10, lease_seconds=1)
            with pytest.raises(s.DeliveryReceiptUnknown):
                s.claim_delivery_receipt(c, row.id, owner_id='expire', now=112+i*10)
            current = s.get_delivery_receipt(c, row.id)
            assert current is not None and current.failure_count == i+1
    assert lease is not None
    for i in range(3):
        with closing(kbc.connect(path)) as c:
            kb.add_comment(c, tid, 'synthetic', 'new demand cannot renew crash budget')
            rev = s.get_task_source(c, tid).current_revision
            ensured = s.ensure_delivery_receipt(c, task_id=tid, platform='telegram', chat_id='-100',
                                                notifier_profile='default', desired_revision=rev)
            assert ensured.failure_count == 3 and ensured.destination_message_id == '701'
            with pytest.raises(s.DeliveryReceiptNotRetryable):
                s.claim_delivery_receipt(c, row.id, owner_id='new-runtime', now=200+i)
            result = s.reconcile_delivery_outcome(c, row.id, **evidence(lease, 'unknown'), now=200+i)
            assert result.failure_count == 3
    # Exhaustion is not evidence destruction: the last exact proof remains usable.
    with closing(kbc.connect(path)) as c:
        settled = s.reconcile_delivery_outcome(c, row.id, **evidence(lease), now=210)
        assert settled.failure_count == 0 and settled.delivered_revision == lease.desired_revision
        assert settled.delivered_revision is not None
        assert settled.desired_revision > settled.delivered_revision
        assert s.claim_delivery_receipt(c, row.id, owner_id='reconciled-next', now=211).receipt.destination_message_id == '701'
