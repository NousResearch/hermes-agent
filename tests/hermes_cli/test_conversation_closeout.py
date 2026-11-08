"""Closeout keeps recoverable history and never repeats uncertain preservation."""

import importlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_state import SessionDB


@pytest.fixture
def closeout_db(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('old', source='desktop')
    db.create_session('fresh', source='desktop')
    yield db
    db.close()


def _store(db):
    return importlib.import_module('hermes_cli.conversation_closeout').CloseoutStore(db)


def _old(**changes):
    return {'logical_session_id': 'old', 'stored_session_id': 'old',
            'runtime_id': 'runtime-old', 'profile': 'default', **changes}


def test_archive_then_new_reuses_preservation_at_its_final_cutoff(closeout_db):
    store = _store(closeout_db)
    first, created = store.prepare(_old(), 'archive-request', 'archive', 'old:1')
    assert created
    assert store.claim_review(first['closeout_id'])
    store.record_review(first['closeout_id'], {'memory': 'accepted', 'tasks': 'reconciled'},
                        'old:3', memory_source_id='source-confirmed', task_ids=['existing-task'])
    replay, created = store.prepare(_old(), 'new-request', 'new', 'old:3')
    assert not created
    assert replay['closeout_id'] == first['closeout_id']
    assert replay['memory_source_id'] == 'source-confirmed'
    assert not store.claim_review(replay['closeout_id'])
    assert replay['next_create_request_id'] == first['next_create_request_id']


def test_online_pending_does_not_prevent_recoverable_archival(closeout_db):
    store = _store(closeout_db)
    record, _ = store.prepare(_old(), 'archive-request', 'archive', 'old:1')
    store.record_review(record['closeout_id'], {'memory': 'pending', 'tasks': 'pending'}, 'old:2')
    result = store.archive(record['closeout_id'], settled=True)
    assert result['status'] == 'preserved' and result['archived']
    assert result['destinations'] == {'memory': 'pending', 'tasks': 'pending'}
    assert closeout_db.get_session('old')['archived']
    assert not closeout_db.get_session('fresh')['archived']


def test_unsettled_work_can_be_soft_archived_without_claiming_settlement(closeout_db):
    store = _store(closeout_db)
    record, _ = store.prepare(_old(), 'archive-request', 'archive', 'old:1')
    store.record_review(record['closeout_id'], {'memory': 'skipped', 'tasks': 'reconciled'}, 'old:2')
    result = store.archive(record['closeout_id'], settled=False)
    assert not result['settled'] and result['archived']
    assert closeout_db.get_session('old') is not None
    assert not closeout_db.get_session('fresh')['archived']


def test_request_id_cannot_be_retargeted(closeout_db):
    store = _store(closeout_db)
    store.prepare(_old(), 'same-request', 'archive', 'old:1')
    with pytest.raises(ValueError, match='request_target_mismatch'):
        store.prepare(_old(logical_session_id='fresh', stored_session_id='fresh'),
                      'same-request', 'new', 'fresh:1')


def test_new_real_content_creates_a_new_preservation_batch(closeout_db):
    store = _store(closeout_db)
    first, _ = store.prepare(_old(), 'archive-request', 'archive', 'old:1')
    store.record_review(first['closeout_id'], {'memory': 'skipped', 'tasks': 'reconciled'}, 'old:2')
    next_record, created = store.prepare(_old(), 'changed-request', 'archive', 'old:4')
    assert created and next_record['closeout_id'] != first['closeout_id']


def test_restart_marks_uncertain_destinations_without_repeating_review(closeout_db):
    store = _store(closeout_db)
    record, _ = store.prepare(_old(), 'archive-request', 'archive', 'old:1')
    assert store.claim_review(record['closeout_id'])
    restored = _store(closeout_db)
    result = restored.recover_uncertain(record['closeout_id'])
    assert result['status'] == 'preserved'
    assert result['destinations'] == {'memory': 'unknown', 'tasks': 'unknown'}
    assert not restored.claim_review(record['closeout_id'])


def test_simultaneous_requests_select_one_closeout_operation(closeout_db):
    path = closeout_db.db_path

    def prepare(request):
        with SessionDB(db_path=path) as db:
            return _store(db).prepare(_old(), request, 'archive', 'old:1')

    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = list(pool.map(prepare, ['left', 'right']))
    assert left[0]['closeout_id'] == right[0]['closeout_id']
    assert sum([left[1], right[1]]) == 1


def test_corrupt_durable_receipt_refuses_new_preservation(closeout_db):
    store = _store(closeout_db)
    record, _ = store.prepare(_old(), 'archive-request', 'archive', 'old:1')
    key = store.operation_key(record['closeout_id'])
    closeout_db.set_meta(key, json.dumps({'unexpected': True}))
    with pytest.raises(ValueError, match='closeout_receipt_unreadable'):
        store.prepare(_old(), 'next-request', 'new', 'old:1')


def test_unconfirmed_archive_is_never_reported_as_success(closeout_db, monkeypatch):
    import hermes_cli.conversation_closeout_jobs as jobs
    store = _store(closeout_db)
    record, _ = store.prepare(_old(), 'archive-request', 'archive', 'old:1')
    store.record_review(record['closeout_id'], {'memory': 'pending', 'tasks': 'pending'}, 'old:2')
    monkeypatch.setattr(jobs, 'archive_conversation_in_transaction', lambda db, conn, sid: [])
    with pytest.raises(ValueError, match='closeout_archive_unconfirmed'):
        store.archive(record['closeout_id'], settled=True)
    assert not store.read(record['closeout_id'])['archived']


def test_late_confirmed_capture_resolves_uncertainty_without_downgrade(closeout_db):
    store = _store(closeout_db)
    record, _ = store.prepare(_old(), 'archive-request', 'archive', 'old:1')
    store.claim_review(record['closeout_id'])
    store.recover_uncertain(record['closeout_id'])
    assert store.claim_capture(record['closeout_id'], 'old')
    store.record_capture(record['closeout_id'], 'old', memory='accepted', source_id='confirmed-id')
    store.record_capture(record['closeout_id'], 'old', memory='pending')
    result = store.read(record['closeout_id'])
    assert result['destinations']['memory'] == 'accepted'
    assert result['memory_source_id'] == 'confirmed-id'


def test_capture_from_another_conversation_cannot_change_receipt(closeout_db):
    store = _store(closeout_db)
    record, _ = store.prepare(_old(), 'archive-request', 'archive', 'old:1')
    with pytest.raises(ValueError, match='capture_target_mismatch'):
        store.record_capture(record['closeout_id'], 'fresh', memory='accepted', source_id='wrong')
    assert store.read(record['closeout_id'])['destinations']['memory'] == 'pending'


def test_capture_claim_is_durable_and_allows_only_one_writer(closeout_db):
    store = _store(closeout_db)
    record, _ = store.prepare(_old(), 'archive-request', 'archive', 'old:1')
    assert store.claim_review(record['closeout_id'])
    assert store.claim_capture(record['closeout_id'], 'old')
    assert not _store(closeout_db).claim_capture(record['closeout_id'], 'old')


@pytest.mark.parametrize('memory,source_id', [('accepted', 'confirmed-id'), ('skipped', None)])
def test_review_completion_cannot_downgrade_a_confirmed_capture(closeout_db, memory, source_id):
    store = _store(closeout_db)
    record, _ = store.prepare(_old(), 'archive-request', 'archive', 'old:1')
    store.claim_review(record['closeout_id'])
    store.claim_capture(record['closeout_id'], 'old')
    store.record_capture(record['closeout_id'], 'old', memory=memory, source_id=source_id)
    store.record_review(record['closeout_id'], {'memory': 'pending', 'tasks': 'pending'}, 'old:2')
    assert store.read(record['closeout_id'])['destinations']['memory'] == memory


def test_superseded_closeout_generation_cannot_archive_the_current_batch(closeout_db):
    store = _store(closeout_db)
    first, _ = store.prepare(_old(), 'first-request', 'archive', 'old:0')
    store.record_review(first['closeout_id'], {'memory': 'pending', 'tasks': 'pending'}, 'old:0')
    second, created = store.prepare(_old(), 'new-work-request', 'archive', 'old:1')
    assert created and second['closeout_id'] != first['closeout_id']
    stale = store.archive(first['closeout_id'], settled=True)
    assert not stale['archived'] and not closeout_db.get_session('old')['archived']


def test_archive_flag_rolls_back_if_its_receipt_cannot_be_written(closeout_db, monkeypatch):
    import hermes_cli.conversation_closeout as closeout
    store = _store(closeout_db)
    record, _ = store.prepare(_old(), 'request', 'archive', 'old:0')
    store.record_review(record['closeout_id'], {'memory': 'pending', 'tasks': 'pending'}, 'old:0')
    save = closeout._save
    def unavailable(conn, key, value):
        raise OSError('synthetic receipt write failure')
    monkeypatch.setattr(closeout, '_save', unavailable)
    with pytest.raises(OSError, match='synthetic receipt write failure'):
        store.archive(record['closeout_id'], settled=True)
    monkeypatch.setattr(closeout, '_save', save)
    assert not closeout_db.get_session('old')['archived']
    assert not store.read(record['closeout_id'])['archived']
