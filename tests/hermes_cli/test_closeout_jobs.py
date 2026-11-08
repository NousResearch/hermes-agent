"""A bounded preservation job archives recoverably without resubmitting unknown work."""
import asyncio
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


@pytest.fixture
def env(tmp_path):
    from hermes_cli.conversation_closeout import CloseoutStore
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('old', source='desktop')
        db.create_session('fresh', source='desktop')
        yield SimpleNamespace(db=db, store=CloseoutStore(db))


@pytest.mark.asyncio
async def test_job_uses_one_primary_review_and_same_provider_receipt(env):
    from hermes_cli.conversation_closeout_jobs import begin_closeout, old_identity
    calls, events, jobs = [], [], []
    async def submit(prompt, operation):
        calls.append(prompt)
        env.store.claim_capture(operation, 'old')
        env.store.record_capture(operation, 'old', memory='accepted', source_id='source', tasks='reconciled')
        return {'status': 'streaming'}
    def launch(coro):
        jobs.append(asyncio.create_task(coro))
    old = old_identity(env.db, 'old', runtime_id='r', profile='default')
    record = begin_closeout(env.db, old, 'request', 'new', submit=submit, settled=lambda: True,
                            publish=events.append, launch=launch, wait_seconds=0)
    assert record['status'] == 'reviewing'
    await asyncio.gather(*jobs)
    final = env.store.read(record['closeout_id'])
    assert final['archived'] and final['destinations'] == {'memory': 'accepted', 'tasks': 'reconciled'}
    assert not env.db.get_session('fresh')['archived']
    replay = begin_closeout(env.db, old, 'request', 'new', submit=submit, settled=lambda: True,
                            publish=events.append, launch=launch, wait_seconds=0)
    assert replay['closeout_id'] == record['closeout_id'] and len(calls) == 1
    assert calls[0].startswith('archive\n\n[Hermes closeout: ' + record['closeout_id'] + ']')


@pytest.mark.asyncio
async def test_expiry_soft_archives_unknown_execution_without_settling_it(env):
    from hermes_cli.conversation_closeout_jobs import begin_closeout, old_identity
    jobs = []
    record = begin_closeout(env.db, old_identity(env.db, 'old', profile='default'), 'request', 'new',
        submit=lambda *args: {'status': 'streaming'}, settled=lambda: False,
        launch=lambda coro: jobs.append(asyncio.create_task(coro)), wait_seconds=0)
    await asyncio.gather(*jobs)
    result = env.store.read(record['closeout_id'])
    assert result['archived'] and not result['settled']
    assert result['destinations'] == {'memory': 'unknown', 'tasks': 'unknown'}


def test_restart_never_resubmits_uncertain_primary_review(env):
    from hermes_cli.conversation_closeout_jobs import begin_closeout, old_identity
    old = old_identity(env.db, 'old', profile='default')
    record, _ = env.store.prepare(old, 'request', 'new', 'old:0')
    env.store.claim_review(record['closeout_id'])
    calls = []
    result = begin_closeout(env.db, old, 'request', 'new', submit=lambda *args: calls.append(1),
        settled=lambda: False, launch=lambda coro: None, wait_seconds=0)
    assert result['archived'] and result['destinations']['memory'] == 'unknown' and not calls


def test_fence_expires_and_unarchive_is_one_line_release(env):
    from hermes_cli.conversation_closeout_fence import require_admission, review_admission
    from hermes_state_runtime import RuntimeStoreError
    record, _ = env.store.prepare({'logical_session_id': 'old', 'stored_session_id': 'old',
        'profile': 'default'}, 'request', 'new', 'old:0')
    with env.db._read_ctx() as conn:
        with pytest.raises(RuntimeStoreError, match='conversation_closing'):
            require_admission(env.db, conn, 'old')
        require_admission(env.db, conn, 'old', now=record['_fence_until'] + 1)
        with review_admission(env.db, record['closeout_id']):
            require_admission(env.db, conn, 'old')
    env.db.set_session_archived('old', False)
    with env.db._read_ctx() as conn:
        require_admission(env.db, conn, 'old')


def test_public_review_text_cannot_bypass_fence_but_owner_submission_can(env):
    from hermes_cli.conversation_closeout_fence import review_admission
    from hermes_state_runtime import RuntimeStoreError, admit_session_input, begin_runtime_epoch
    record, _ = env.store.prepare({'logical_session_id': 'old', 'stored_session_id': 'old',
        'profile': 'default'}, 'request', 'new', 'old:0')
    epoch = begin_runtime_epoch(env.db, instance_id='fixture')
    operation = record['closeout_id']
    args = dict(epoch=epoch, principal_id='human', session_id='old', request_id='closeout-review-' + operation,
                payload={'text': 'archive\n\n[Hermes closeout: ' + operation + ']'})
    with pytest.raises(RuntimeStoreError, match='conversation_closing'):
        admit_session_input(env.db, **args)
    with review_admission(env.db, operation):
        accepted = admit_session_input(env.db, **args)
        with pytest.raises(RuntimeStoreError, match='conversation_closing'):
            admit_session_input(env.db, **{**args, 'request_id': 'different-input'})
    assert admit_session_input(env.db, **args)['admission_id'] == accepted['admission_id']


def test_archive_covers_validated_local_reset_segments_without_touching_execution(env):
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from gateway.session_policy import build_policy
    from hermes_cli.conversation_closeout_jobs import old_identity
    from hermes_state_keys import local_identity
    from hermes_state_local import commit_local_session
    from hermes_state_local_lineage import reset_local_target
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input, get_session_admission
    db = env.db
    epoch = begin_runtime_epoch(db, instance_id='fixture')
    sid = local_identity('profile', 'human', 'r')
    source = SessionSource(platform=Platform.LOCAL, chat_id=sid, user_id='human', chat_type='dm')
    now = _now()
    entry = SessionEntry('local:' + sid, sid, now, now, origin=source, platform=Platform.LOCAL)
    policy = build_policy({'source': 'cli', 'cwd': str(Path(db.db_path).parent), 'model': 'fixture', 'toolsets': []},
                          {'platform_toolsets': {'cli': []}}, private_secrets={})
    commit_local_session(db, epoch=epoch, receipt={'profile_id': 'profile', 'principal_id': 'human',
        'request_id': 'r', 'session_id': sid, 'route': entry.session_key, 'entry': entry.to_dict(), 'policy': asdict(policy)})
    reset = SessionEntry(entry.session_key, 'tip', now, now, origin=source, platform=Platform.LOCAL, is_fresh_reset=True)
    reset_local_target(db, epoch=epoch, parent_session_id=sid, entry=reset.to_dict())
    queued = admit_session_input(db, epoch=epoch, principal_id='human', session_id=sid,
        request_id='work', payload={'text': 'synthetic ongoing work'})
    started = claim_session_input(db, epoch=epoch, session_id=sid)
    before = db.get_session(sid)['runtime_generation']
    assert old_identity(db, 'tip', profile='default')['logical_session_id'] == sid
    old = old_identity(db, 'tip', profile='default')
    record, _ = env.store.prepare(old, 'archive-started-reset', 'archive', 'tip:0')
    env.store.record_review(record['closeout_id'], {'memory': 'pending', 'tasks': 'pending'}, 'tip:0')
    assert env.store.archive(record['closeout_id'], settled=False)['archived']
    assert db.get_session(sid)['archived'] and db.get_session('tip')['archived']
    assert db.get_session(sid)['runtime_generation'] == before
    assert get_session_admission(db, admission_id=queued['admission_id'])['status'] == started['status'] == 'started'


def test_late_review_final_rows_remain_the_same_preservation_batch(env):
    from hermes_cli.conversation_closeout_jobs import history_cutoff, old_identity
    old = old_identity(env.db, 'old', profile='default')
    record, _ = env.store.prepare(old, 'archive-request', 'archive', history_cutoff(env.db, 'old'))
    operation = record['closeout_id']
    env.store.claim_review(operation)
    env.store.recover_uncertain(operation)
    env.store.archive(operation, settled=False)
    env.db.append_message('old', 'assistant', 'Synthetic late review result')
    assert env.store.claim_capture(operation, 'old')
    env.store.record_capture(operation, 'old', memory='accepted', source_id='late-source')
    replay, created = env.store.prepare(old, 'new-request', 'new', history_cutoff(env.db, 'old'))
    assert not created and replay['closeout_id'] == operation


@pytest.mark.asyncio
async def test_local_archive_failure_keeps_receipt_and_retries_only_local_metadata(env, monkeypatch):
    import hermes_cli.conversation_closeout_jobs as jobs_module
    real_archive = jobs_module.archive_conversation_in_transaction
    attempts, reviews, jobs = [], [], []
    def archive(db, conn, sid):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError('synthetic local storage failure')
        return real_archive(db, conn, sid)
    monkeypatch.setattr(jobs_module, 'archive_conversation_in_transaction', archive)
    def submit(*args):
        reviews.append(1)
        return {'status': 'queued'}
    old = jobs_module.old_identity(env.db, 'old', profile='default')
    params = dict(submit=submit, settled=lambda: True,
                  launch=lambda coro: jobs.append(asyncio.create_task(coro)), wait_seconds=0)
    record = jobs_module.begin_closeout(env.db, old, 'request', 'new', **params)
    await asyncio.gather(*jobs)
    failed = env.store.read(record['closeout_id'])
    assert failed['status'] == 'failed' and not failed['archived']
    assert not env.db.get_session('old')['archived']
    recovered = jobs_module.begin_closeout(env.db, old, 'request', 'new', **params)
    assert recovered['archived'] and reviews == [1]


def test_restore_committed_before_archive_transaction_cancels_stale_worker(env, monkeypatch):
    from hermes_cli.conversation_closeout_jobs import _refresh, old_identity
    old = old_identity(env.db, 'old', profile='default')
    record, _ = env.store.prepare(old, 'request', 'archive', 'old:0')
    operation = record['closeout_id']
    env.store.claim_review(operation)
    record = env.store.record_review(operation, {'memory': 'pending', 'tasks': 'pending'}, 'old:0')
    entered, release = Event(), Event()
    write = env.store._write
    def paused_write(change):
        entered.set()
        assert release.wait(5)
        return write(change)
    monkeypatch.setattr(env.store, '_write', paused_write)
    with ThreadPoolExecutor(max_workers=1) as pool:
        stale = pool.submit(_refresh, env.store, record, lambda: True)
        assert entered.wait(5)
        with SessionDB(db_path=env.db.db_path) as other:
            other.set_session_archived('old', False)
        release.set()
        result = stale.result(timeout=5)
    monkeypatch.setattr(env.store, '_write', write)
    current = env.store.read(operation)
    assert not env.db.get_session('old')['archived']
    assert current['_fence_released'] and not current['archived'] and not result['archived']
    # A late same-provider acceptance is independent from the cancelled archive.
    env.store.record_capture(operation, 'old', memory='accepted', source_id='late-source')
    assert env.store.read(operation)['memory_source_id'] == 'late-source'
    assert not env.store.read(operation)['archived']


def test_restore_after_archive_commit_wins_row_and_receipt(env):
    from hermes_cli.conversation_closeout_jobs import _refresh, old_identity
    old = old_identity(env.db, 'old', profile='default')
    record, _ = env.store.prepare(old, 'request', 'archive', 'old:0')
    operation = record['closeout_id']
    record = env.store.record_review(operation, {'memory': 'pending', 'tasks': 'pending'}, 'old:0')
    archived = _refresh(env.store, record, lambda: True)
    assert archived['archived'] and env.db.get_session('old')['archived']
    with SessionDB(db_path=env.db.db_path) as other:
        other.set_session_archived('old', False)
    # Even an old in-memory receipt must not undo the later explicit restore.
    result = _refresh(env.store, archived, lambda: True)
    assert not result['archived'] and result['_fence_released']
    assert not env.db.get_session('old')['archived']
