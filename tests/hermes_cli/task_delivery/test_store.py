"""Persistent delivery facts survive delegation, review and resume boundaries."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_cli.task_delivery.store import DeliveryError, WorkLedger
from hermes_state import SessionDB


@pytest.fixture
def ledger(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('owner', source='gui')
        yield WorkLedger(db), db


def register(ledger, task='fix'):
    return ledger.register('owner', task, title='Repair the service', acceptance='Real request succeeds')


def claim(ledger, owner='worker', request='claim', task='fix'):
    return ledger.claim('owner', task, owner=owner, role='implement', request_id=request,
                        base='base-commit', files=['service.py'])


def deliver(ledger, tmp_path, lease, text='patch', task='fix'):
    artifact = tmp_path / (text + '.patch')
    artifact.write_text(text)
    return ledger.deliver('owner', task, owner=lease['claim']['owner'],
                          generation=lease['claim']['generation'], path=artifact, summary='Implemented')


def report(digest, request='review', passed=True):
    return {'request_id': request, 'reviewer': 'independent', 'candidate_digest': digest, 'passed': passed,
            'findings': [] if passed else [{'issue_id': 'wrong-result', 'category': 'regression',
                'blocking': True, 'evidence_ref': 'probe-1', 'message': 'Real request still fails'}]}


def test_claim_is_atomic_and_delivery_does_not_mean_acceptance(ledger, tmp_path):
    work, db = ledger
    register(work)

    def compete(owner):
        try:
            return claim(WorkLedger(db), owner=owner, request=owner)
        except DeliveryError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = [x for x in pool.map(compete, ['one', 'two']) if x]
    assert len(claimed) == 1
    delivered = deliver(work, tmp_path, claimed[0])
    assert delivered['implementation'] == 'delivered'
    assert delivered['acceptance'] is None
    with pytest.raises(DeliveryError, match='already delivered'):
        claim(work, owner='duplicate', request='duplicate')
    with pytest.raises(DeliveryError, match='review'):
        work.advance('owner', 'fix', stage='acceptance', candidate_digest=delivered['candidate']['digest'],
                     evidence='Customer flow passed')


def test_evidence_is_bound_to_candidate_and_preserved_when_reopened(ledger, tmp_path):
    work, db = ledger
    register(work)
    delivered = deliver(work, tmp_path, claim(work))
    digest = delivered['candidate']['digest']
    work.record_review('owner', 'fix', report(digest))
    for stage in ['integration', 'verification', 'acceptance']:
        work.advance('owner', 'fix', stage=stage, candidate_digest=digest, evidence=stage + ' evidence')
    resumed = WorkLedger(db).get('owner', 'fix')
    assert resumed['acceptance']['candidate_digest'] == digest
    work.reopen('owner', 'fix', reason='New regression confirmed')
    assert work.get('owner', 'fix')['acceptance'] is None
    assert any(event['kind'] == 'acceptance' for event in work.get('owner', 'fix')['events'])
    next_delivery = deliver(work, tmp_path, claim(work, request='repair'), text='new-patch')
    with pytest.raises(DeliveryError, match='candidate'):
        work.advance('owner', 'fix', stage='verification', candidate_digest=digest, evidence='Old test')
    assert next_delivery['candidate']['digest'] != digest


def test_old_worker_cannot_publish_after_explicit_reconciliation(ledger, tmp_path):
    work, _ = ledger
    register(work)
    old = claim(work)
    work.release('owner', 'fix', owner='worker', generation=old['claim']['generation'], reason='Worker stopped')
    current = claim(work, owner='replacement', request='replacement')
    with pytest.raises(DeliveryError, match='claim'):
        deliver(work, tmp_path, old)
    assert current['claim']['generation'] > old['claim']['generation']


def test_failed_reviews_bound_repairs_across_candidates_and_restart(ledger, tmp_path):
    work, db = ledger
    register(work)
    for index in range(3):
        state = deliver(work, tmp_path, claim(work, request=f'claim-{index}'), text=f'patch-{index}')
        state = work.record_review('owner', 'fix', report(state['candidate']['digest'], f'review-{index}', False))
        work = WorkLedger(db)
    assert state['review']['status'] == 'needs_decision'
    with pytest.raises(DeliveryError, match='decision'):
        claim(work, request='fourth')
    with pytest.raises(DeliveryError, match='review'):
        work.advance('owner', 'fix', stage='acceptance', candidate_digest=state['candidate']['digest'], evidence='done')


def test_todo_sync_keeps_history_and_refuses_premature_completion(ledger):
    work, _ = ledger
    register(work)
    work.sync_todos('owner', [{'id': 'fix', 'content': 'Repair service', 'status': 'in_progress'}])
    with pytest.raises(DeliveryError, match='acceptance'):
        work.check_todos('owner', [{'id': 'fix', 'content': 'Repair service', 'status': 'completed'}])
    work.sync_todos('owner', [])
    assert work.get('owner', 'fix')['listed'] is False
    assert work.get('owner', 'fix')['acceptance_criteria'] == 'Real request succeeds'
    assert 'fix' in work.recovery_context('owner')


def test_profile_and_session_scope_and_file_ownership(ledger):
    work, db = ledger
    register(work)
    claim(work)
    register(work, 'other')
    with pytest.raises(DeliveryError, match='file'):
        claim(work, task='other', owner='other', request='other')
    db.create_session('different', source='gui')
    assert work.list('different') == []
    with pytest.raises(DeliveryError, match='not found'):
        work.get('different', 'fix')


def test_claim_requests_and_delivery_replays_cannot_start_another_worker(ledger, tmp_path):
    work, _ = ledger
    register(work)
    lease = claim(work)
    state = deliver(work, tmp_path, lease)
    assert deliver(work, tmp_path, lease)['revision'] == state['revision']
    work.reopen('owner', 'fix', reason='Changed requirement')
    with pytest.raises(DeliveryError, match='already settled'):
        claim(work)
    current = claim(work, request='new-attempt')
    renewed = work.renew('owner', 'fix', owner='worker', generation=current['generation'])
    assert renewed['claim']['generation'] == current['generation']


def test_new_unreviewed_candidate_cannot_reuse_previous_changes_requested(ledger, tmp_path):
    work, _ = ledger
    register(work)
    first = deliver(work, tmp_path, claim(work))
    work.record_review('owner', 'fix', report(first['candidate']['digest'], passed=False))
    second = deliver(work, tmp_path, claim(work, request='repair'), text='repair')
    with pytest.raises(DeliveryError, match='already delivered'):
        claim(work, request='duplicate-repair')
    assert second['candidate']['digest'] != first['candidate']['digest']


def test_execution_receipt_cannot_replace_artifact_or_mark_acceptance(ledger, tmp_path):
    work, _ = ledger
    register(work)
    lease = claim(work)
    item = deliver(work, tmp_path, lease)
    receipt = work.finish_execution('owner', 'fix', owner='worker', generation=lease['generation'],
                                    status='completed', result='All done')
    assert receipt['candidate'] == item['candidate']
    assert receipt['implementation'] == 'delivered' and receipt['acceptance'] is None
    assert work.finish_execution('owner', 'fix', owner='worker', generation=lease['generation'],
                                  status='completed', result='All done')['revision'] == receipt['revision']


def test_an_accepted_candidate_requires_explicit_reopen_before_new_review(ledger, tmp_path):
    work, _ = ledger
    register(work)
    item = deliver(work, tmp_path, claim(work))
    digest = item['candidate']['digest']
    passed = report(digest)
    work.record_review('owner', 'fix', passed)
    for stage in ['integration', 'verification', 'acceptance']:
        work.advance('owner', 'fix', stage=stage, candidate_digest=digest, evidence='Recorded evidence')
    with pytest.raises(DeliveryError, match='reopened'):
        work.record_review('owner', 'fix', report(digest, 'new-failure', False))
    assert work.get('owner', 'fix')['review']['status'] == 'passed'
    work.reopen('owner', 'fix', reason='New evidence invalidates acceptance')
    state = work.record_review('owner', 'fix', report(digest, 'new-failure', False))
    assert state['acceptance'] is None and state['review']['status'] == 'changes_requested'


def test_unchanged_artifacts_cannot_buy_unlimited_automatic_repairs(ledger, tmp_path):
    work, _ = ledger
    register(work)
    first = deliver(work, tmp_path, claim(work))
    work.record_review('owner', 'fix', report(first['candidate']['digest'], passed=False))
    for index in range(2):
        deliver(work, tmp_path, claim(work, request=f'repair-{index}'))
    assert work.get('owner', 'fix')['needs_decision'] is True
    with pytest.raises(DeliveryError, match='decision'):
        claim(work, request='another-repair')


def test_scope_decision_preserves_exhausted_epoch_and_fences_old_execution(ledger, tmp_path):
    work, _ = ledger
    register(work)
    for index in range(3):
        lease = claim(work, request=f'attempt-{index}')
        item = deliver(work, tmp_path, lease, text=f'candidate-{index}')
        work.record_review('owner', 'fix', report(item['candidate']['digest'], f'review-{index}', False))
    with pytest.raises(DeliveryError, match='decision'):
        work.reopen('owner', 'fix', reason='Try again')
    state = work.decision('owner', 'fix', reason='Operator selected a smaller supported workflow',
                          acceptance='The selected workflow succeeds')
    assert state['review_epoch'] == 2 and state['repair_attempts'] == 0
    assert state['events'][-1]['previous']['review']['status'] == 'needs_decision'
    with pytest.raises(DeliveryError, match='superseded'):
        work.finish_execution('owner', 'fix', owner='worker', generation=lease['generation'],
                              status='completed', result='late result')


def test_permanent_delete_removes_work_history_without_touching_another_session(ledger):
    work, db = ledger
    register(work)
    db.create_session('other-session', source='gui')
    work.register('other-session', 'keep', title='Keep', acceptance='Separate session')
    assert db.delete_session('owner')
    assert work.list('owner') == []
    assert work.get('other-session', 'keep')['task_id'] == 'keep'
