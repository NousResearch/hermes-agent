"""Independent probes for failed-settlement cleanup and ownership races."""
import asyncio
from pathlib import Path
import threading

import pytest

from gateway.session_authority import admission_row
from gateway.session_contract import Submission
from persistence.core.errors import AuthorityRefusal
from tests.gateway.fixtures.storage_handles import close_storage_handles
from tests.gateway.test_session_authority_cancel_settlement import _authority, ACTOR, REF


@pytest.mark.asyncio
async def test_commit_then_error_still_releases_terminal_native_media(tmp_path, monkeypatch):
    from gateway import session_results
    from gateway.platforms.base import get_image_cache_dir
    db, authority = _authority(tmp_path, monkeypatch)
    store = authority.persistence
    image = get_image_cache_dir() / 'timeout.png'
    image.write_bytes(b'\x89PNG\r\n\x1a\nowned')
    async def execute(owner, ref, row):
        owner.pending_results[row['admission_id']] = {'result': {'final_response': 'done'}, 'usage': {}}
        return 'done'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    original = session_results.finish_result
    def commit_then_error(*args, **kwargs):
        monkeypatch.setattr(session_results, 'finish_result', original)
        original(*args, **kwargs)
        raise OSError('acknowledgement after commit lost')
    monkeypatch.setattr(session_results, 'finish_result', commit_then_error)
    with db:
        receipt = await authority.submit(ACTOR, Submission('image', REF, {
            'text': 'work', 'attachments': [{'path': str(image), 'mime': 'image/png'}]}, 'queue'))
        row = store.admissions.get(admission_id=receipt.admission_id)
        retained = Path(row.payload['attachments_v1']['media'][0]['path'])
        assert retained.exists()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert store.admissions.get(admission_id=receipt.admission_id).status == 'terminal'
        from gateway.session_runtime_workers import join_authority_work
        await join_authority_work(authority, timeout=5)
        assert not retained.exists(), 'commit-then-error skipped terminal media cleanup'


@pytest.mark.asyncio
async def test_discard_during_recovery_stamp_does_not_replay_head_or_lose_follower(tmp_path, monkeypatch):
    from gateway import session_results, session_settlement_recovery
    db, authority = _authority(tmp_path, monkeypatch)
    store = authority.persistence
    schedule = authority._schedule
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    calls = []
    async def execute(owner, ref, row):
        calls.append(row['request_id'])
        owner.pending_results[row['admission_id']] = {'result': {'final_response': row['request_id']}, 'usage': {}}
        return row['request_id']
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    original = session_results.finish_result
    # Storage stays down across every bounded settlement attempt, so the claim reaches the
    # recovery stamp; it comes back once the stamp is entered (the follower settles normally).
    monkeypatch.setattr(session_settlement_recovery, '_SETTLE_RETRY_DELAYS_S', (0.01, 0.01), raising=False)
    def unavailable(*args, **kwargs):
        raise OSError('lost settlement')
    monkeypatch.setattr(session_results, 'finish_result', unavailable)
    mark = session_settlement_recovery._mark_unknown
    entered, release = threading.Event(), threading.Event()
    def held_mark(*args):
        monkeypatch.setattr(session_results, 'finish_result', original)
        result = mark(*args)
        entered.set()
        assert release.wait(10)
        return result
    monkeypatch.setattr(session_settlement_recovery, '_mark_unknown', held_mark)
    with db:
        head = await authority.submit(ACTOR, Submission('head', REF, {'text': 'head'}, 'queue'))
        follower = await authority.submit(ACTOR, Submission('follower', REF, {'text': 'follower'}, 'queue'))
        schedule(REF)
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            row = store.admissions.get(admission_id=head.admission_id)
            await authority.resolve_unknown(ACTOR, REF, head.admission_id, row.generation)
        finally:
            release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert calls == ['head', 'follower']
        assert store.admissions.get(admission_id=follower.admission_id).outcome == 'completed'
        assert authority.sessions['s'].event_stream.execution == {}


@pytest.mark.asyncio
async def test_stale_owner_cannot_stamp_a_new_epoch_claim_unknown(tmp_path, monkeypatch):
    from gateway.session_settlement_recovery import recover_failed_settlement
    db, authority = _authority(tmp_path, monkeypatch)
    store = authority.persistence
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        receipt = await authority.submit(ACTOR, Submission('work', REF, {'text': 'work'}, 'queue'))
        new_epoch = store.runtime.begin_epoch(instance_id='new-owner')
        row = admission_row(store.admissions.claim(epoch=new_epoch, session_id='s'))
        with pytest.raises(AuthorityRefusal):
            await recover_failed_settlement(authority, row)
        current = store.admissions.get(admission_id=receipt.admission_id)
        assert current.status == 'started' and current.owner_epoch == new_epoch


@pytest.mark.asyncio
@pytest.mark.parametrize('failure_stage', ['cleanup', 'after_completion'])
async def test_post_commit_cleanup_error_still_publishes_the_committed_completion(tmp_path, monkeypatch, failure_stage):
    from gateway import session_ingress_media
    db, authority = _authority(tmp_path, monkeypatch)
    store = authority.persistence
    frames = []
    authority.sessions['s'].event_stream.observers.add(frames.append)
    async def execute(owner, ref, row):
        owner.pending_results[row['admission_id']] = {'result': {'final_response': 'done'}, 'usage': {}}
        return 'done'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    if failure_stage == 'cleanup':
        original = session_ingress_media.release_admission_media
        def unavailable_once(*args):
            monkeypatch.setattr(session_ingress_media, 'release_admission_media', original)
            raise OSError('post-commit cleanup temporarily unavailable')
        monkeypatch.setattr(session_ingress_media, 'release_admission_media', unavailable_once)
    else:
        original = authority._publish_pending
        def unavailable_after_completion(ref):
            if any(frame['params']['type'] == 'message.complete' for frame in frames):
                monkeypatch.setattr(authority, '_publish_pending', original)
                raise OSError('idle publication temporarily unavailable')
            return original(ref)
        monkeypatch.setattr(authority, '_publish_pending', unavailable_after_completion)
    with db:
        receipt = await authority.submit(ACTOR, Submission('cleanup', REF, {'text': 'work'}, 'queue'))
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert store.admissions.get(admission_id=receipt.admission_id).outcome == 'completed'
        complete = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete']
        assert len(complete) == 1 and complete[0]['text'] == 'done'
