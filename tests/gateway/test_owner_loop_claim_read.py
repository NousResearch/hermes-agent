"""The FIFO claim's pending-row reads wait off the owner loop.

Every turn's drain reads the session's pending admissions before its claim (and re-reads the head
after an awaited preclaim check). On slow storage that read used to stop the whole owner loop, every
other session's turn, Stop and socket included. Real SessionAuthority + state.db; reads are slowed
only for the session under test and only while the drain claims, and the stall is measured as
loop-tick lag. The claim still takes the head the read validated.
"""
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

SLOW_S = 2.0


@pytest.mark.asyncio
async def test_claim_pending_read_keeps_the_owner_loop_ticking(tmp_path, monkeypatch):
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionRef
    from persistence.sqlite.authority.admissions import SQLiteAdmissionRepository
    from persistence.sqlite.conversation.sessions import SQLiteSessionRepository
    db = storage_handle(tmp_path)
    seed_session(db, 's', 'cron')
    authority = SessionAuthority(SimpleNamespace(_draining=False, adapters={}), profile_id='p', instance_id='t',
                                 persistence=db, epoch=db.runtime.begin_epoch(instance_id='t'))
    live = authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='u'), 'route')
    row, _ = db.admissions.admit(epoch=authority.epoch, principal_id='u', session_id='s',
                                 request_id='r1', payload={'text': 'x'})
    queued, _ = db.admissions.admit(epoch=authority.epoch, principal_id='u', session_id='s',
                                    request_id='r2', payload={'text': 'follower'})
    assert queued.admission_id != row.admission_id
    real, real_session_get, real_session_list_row = (
        SQLiteAdmissionRepository.list, SQLiteSessionRepository.get, SQLiteSessionRepository.list_row)
    slow = threading.Event()
    slow_calls, slow_session_get_calls, slow_session_list_row_calls = [], [], []

    def slow_list(*args, **kwargs):
        if slow.is_set():
            slow_calls.append((threading.current_thread().name, threading.get_ident()))
            time.sleep(SLOW_S)  # health: allow HX013 -- the slow storage read IS the stall under test
        return real(*args, **kwargs)

    def slow_session_get(*args, **kwargs):
        if slow.is_set():
            slow_session_get_calls.append((threading.current_thread().name, threading.get_ident()))
            time.sleep(SLOW_S)  # health: allow HX013 -- the session lookup shares the claim publication path
        return real_session_get(*args, **kwargs)

    def slow_session_list_row(*args, **kwargs):
        if slow.is_set():
            slow_session_list_row_calls.append((threading.current_thread().name, threading.get_ident()))
            time.sleep(SLOW_S)  # health: allow HX013 -- catch the queued-controls source-cache fallback
        return real_session_list_row(*args, **kwargs)

    monkeypatch.setattr(SQLiteAdmissionRepository, 'list', slow_list)
    monkeypatch.setattr(SQLiteSessionRepository, 'get', slow_session_get)
    monkeypatch.setattr(SQLiteSessionRepository, 'list_row', slow_session_list_row)
    gaps = []
    owner_thread = threading.get_ident()

    async def tick():
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.02)  # health: allow HX013 -- the ticker's period is the owner-loop stall measurement
            now = time.monotonic()
            gaps.append(now - last)
            last = now
    ticker = asyncio.create_task(tick())
    await asyncio.sleep(0.05)  # health: allow HX013 -- the ticker runs before the claim starts
    slow.set()
    try:
        claimed, first = await asyncio.wait_for(authority._claim_next(SessionRef('p', 's'), live), 30)
    finally:
        slow.clear()
        ticker.cancel()
        db.close()
    assert first['admission_id'] == row.admission_id
    assert claimed and claimed['admission_id'] == row.admission_id and claimed['status'] == 'started'
    assert slow_calls and all(ident != owner_thread for _, ident, *_ in slow_calls), (
        f'slow admission reads must run off the owner loop; owner_thread={owner_thread}; slow_calls={slow_calls}'
    )
    assert slow_session_get_calls and all(ident != owner_thread for _, ident, *_ in slow_session_get_calls), (
        f'slow session-row reads must run off the owner loop; owner_thread={owner_thread}; '
        f'slow_session_get_calls={slow_session_get_calls}'
    )
    assert not slow_session_list_row_calls, (
        f'claim snapshot must supply the source to queued controls instead of using list_row: '
        f'{slow_session_list_row_calls}'
    )
    assert max(gaps) < 0.2, f'owner loop stalled {max(gaps):.2f}s behind the claim read'


@pytest.mark.asyncio
async def test_queued_cancel_waits_for_claim_pending_snapshot(tmp_path, monkeypatch):
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionActor, SessionRef
    from persistence.sqlite.authority.admissions import SQLiteAdmissionRepository

    db = storage_handle(tmp_path)
    seed_session(db, 's', 'cron')
    authority = SessionAuthority(SimpleNamespace(_draining=False, adapters={}), profile_id='p', instance_id='t',
                                 persistence=db, epoch=db.runtime.begin_epoch(instance_id='t'))
    live = authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='u'), 'route')
    first, _ = db.admissions.admit(epoch=authority.epoch, principal_id='u', session_id='s',
                                   request_id='r1', payload={'text': 'first'})
    second, _ = db.admissions.admit(epoch=authority.epoch, principal_id='u', session_id='s',
                                    request_id='r2', payload={'text': 'second'})
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    actor = SessionActor('u', 'p', frozenset({'session:submit', 'session:control'}), 'test')
    ref = SessionRef('p', 's')
    loop = asyncio.get_running_loop()
    snapshot_ready = asyncio.Event()
    release_snapshot = threading.Event()
    count_lock = threading.Lock()
    list_calls = 0
    owner_thread = threading.get_ident()
    offloop_list_reads = []
    frames, claim_info_index = [], []
    cancelled_info = asyncio.Event()
    cancel_info_index = []

    def observe(frame):
        frames.append(frame)
        params = frame['params']
        if params['type'] != 'session.info':
            return
        pending_ids = {item['admission_id'] for item in params.get('payload', {}).get('pending', [])}
        if first.admission_id in pending_ids and second.admission_id in pending_ids:
            claim_info_index.append(len(frames) - 1)
        if (claim_info_index and second.admission_id not in pending_ids and not cancelled_info.is_set()):
            cancel_info_index.append(len(frames) - 1)
            cancelled_info.set()

    live.event_stream.observers.add(observe)
    real_list = SQLiteAdmissionRepository.list
    stale_queued_ids = []

    def paused_snapshot_list(*args, **kwargs):
        nonlocal list_calls
        rows = real_list(*args, **kwargs)
        with count_lock:
            list_calls += 1
            call = list_calls
            if threading.get_ident() != owner_thread:
                offloop_list_reads.append(call)
        if call == 2:
            # The snapshot includes the queued follower. The serialized cancel must wait until the
            # claim publishes this still-current projection; the adjacent regression covers a
            # cancellation commit before publication.
            stale_queued_ids.extend(row.admission_id for row in rows if row.status == 'queued')
            loop.call_soon_threadsafe(snapshot_ready.set)
            if not release_snapshot.wait(10):
                raise AssertionError('test did not release the in-flight pending snapshot')
        return rows

    monkeypatch.setattr(SQLiteAdmissionRepository, 'list', paused_snapshot_list)
    claim_task = asyncio.create_task(authority._claim_next(ref, live))
    cancel_task = None
    try:
        await asyncio.wait_for(snapshot_ready.wait(), 10)
        assert second.admission_id in stale_queued_ids, 'claim snapshot must contain the queued follower'
        cancel_task = asyncio.create_task(authority.cancel_queued(actor, ref, second.admission_id))
        release_snapshot.set()
        claimed, pending_before = await asyncio.wait_for(claim_task, 20)
        cancel_result = await asyncio.wait_for(cancel_task, 20)
        assert (cancel_result.status, cancel_result.outcome) == ('terminal', 'cancelled')
        await asyncio.wait_for(cancelled_info.wait(), 10)
    finally:
        release_snapshot.set()
        tasks = [claim_task] + ([cancel_task] if cancel_task is not None else [])
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 20)
        finally:
            db.close()

    assert pending_before['admission_id'] == first.admission_id
    assert claimed and claimed['admission_id'] == first.admission_id and claimed['status'] == 'started'
    assert offloop_list_reads == [1, 2], (
        f'one FIFO read and one publication snapshot should run off-loop without a cancellation retry: '
        f'{offloop_list_reads}'
    )
    assert claim_info_index and cancel_info_index, 'claim and cancellation pending projections must both publish'
    assert claim_info_index[0] < cancel_info_index[0], 'claim projection must publish before cancellation commits'
    post_cancel_infos = [frame['params'] for frame in frames[cancel_info_index[0]:]
                         if frame['params']['type'] == 'session.info'
                         and 'pending' in frame['params'].get('payload', {})]
    assert post_cancel_infos, 'the cancellation projection must be observable'
    assert all(item['admission_id'] != second.admission_id
               for params in post_cancel_infos for item in params['payload']['pending']), (
        'a post-cancel publication must not reintroduce the cancelled queued row'
    )


@pytest.mark.asyncio
async def test_claim_snapshot_retries_for_changes_not_duplicate_projections(tmp_path, monkeypatch):
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionRef
    from persistence.sqlite.authority.admissions import SQLiteAdmissionRepository

    db = storage_handle(tmp_path)
    seed_session(db, 's', 'cron')
    authority = SessionAuthority(SimpleNamespace(_draining=False, adapters={}), profile_id='p', instance_id='t',
                                 persistence=db, epoch=db.runtime.begin_epoch(instance_id='t'))
    live = authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='u'), 'route')
    row, _ = db.admissions.admit(epoch=authority.epoch, principal_id='u', session_id='s',
                                 request_id='r1', payload={'text': 'first'})
    ref = SessionRef('p', 's')
    authority._publish_pending(ref)  # establish the pre-claim projection
    loop = asyncio.get_running_loop()
    snapshot_ready = asyncio.Queue()
    releases = {2: threading.Event(), 3: threading.Event()}
    count_lock = threading.Lock()
    owner_thread = threading.get_ident()
    offloop_reads = []
    real_list = SQLiteAdmissionRepository.list

    def paused_list(*args, **kwargs):
        rows = real_list(*args, **kwargs)
        with count_lock:
            if threading.get_ident() != owner_thread:
                offloop_reads.append(len(offloop_reads) + 1)
                call = len(offloop_reads)
            else:
                call = None
        if call in releases:
            loop.call_soon_threadsafe(snapshot_ready.put_nowait, call)
            if not releases[call].wait(10):
                raise AssertionError(f'test did not release pending snapshot {call}')
        return rows

    monkeypatch.setattr(SQLiteAdmissionRepository, 'list', paused_list)
    claim_task = asyncio.create_task(authority._claim_next(ref, live))
    try:
        assert await asyncio.wait_for(snapshot_ready.get(), 10) == 2
        # The first pending publication reflects the committed claim; the second is identical. Local-only
        # info events also arrive, but neither should cause more than the changed-pending retry.
        authority._publish_pending(ref)
        authority._publish_pending(ref)
        live.event_stream.publish(ref.session_id, {'model': 'selected'}, event_type='session.info')
        live.event_stream.publish(ref.session_id, {'model': 'selected'}, event_type='session.info')
        releases[2].set()

        assert await asyncio.wait_for(snapshot_ready.get(), 10) == 3
        authority._publish_pending(ref)
        live.event_stream.publish(ref.session_id, {'model': 'selected'}, event_type='session.info')
        releases[3].set()
        claimed, first = await asyncio.wait_for(claim_task, 20)
    finally:
        for release in releases.values():
            release.set()
        if not claim_task.done():
            claim_task.cancel()
        db.close()

    assert first['admission_id'] == row.admission_id
    assert claimed and claimed['admission_id'] == row.admission_id and claimed['status'] == 'started'
    assert offloop_reads == [1, 2, 3], f'only the changed projection should cause a retry: {offloop_reads}'


@pytest.mark.asyncio
async def test_empty_claim_does_not_publish_pending_snapshot(tmp_path):
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionRef

    db = storage_handle(tmp_path)
    seed_session(db, 's', 'cron')
    authority = SessionAuthority(SimpleNamespace(_draining=False, adapters={}), profile_id='p', instance_id='t',
                                 persistence=db, epoch=db.runtime.begin_epoch(instance_id='t'))
    live = authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='u'), 'route')
    ref = SessionRef('p', 's')
    watermark = live.event_stream.watermark()
    try:
        claimed, first = await asyncio.wait_for(authority._claim_next(ref, live), 10)
        replay = live.event_stream.since(*watermark)
    finally:
        db.close()

    assert claimed is None and first is None
    assert replay['events'] == [], 'a no-op claim published a redundant session.info frame'


@pytest.mark.asyncio
async def test_queued_cancel_commit_cannot_race_claim_snapshot_publication(tmp_path, monkeypatch):
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionActor, SessionRef
    from persistence.sqlite.authority.admissions import SQLiteAdmissionRepository

    class LockProbe(asyncio.Lock):
        """Instrument the shared per-session mutation gate without changing its FIFO lock behavior."""
        def __init__(self):
            super().__init__()
            self.requests, self.acquisitions = asyncio.Queue(), asyncio.Queue()
            self._next = 0
            self._holder = None
            self._state_lock = threading.Lock()

        async def __aenter__(self):
            with self._state_lock:
                self._next += 1
                token = self._next
            self.requests.put_nowait(token)
            await super().acquire()
            with self._state_lock:
                self._holder = token
            self.acquisitions.put_nowait(token)

        async def __aexit__(self, exc_type, exc_value, traceback):
            with self._state_lock:
                self._holder = None
            super().release()

        def holder(self):
            with self._state_lock:
                return self._holder

    db = storage_handle(tmp_path)
    seed_session(db, 's', 'cron')
    authority = SessionAuthority(SimpleNamespace(_draining=False, adapters={}), profile_id='p', instance_id='t',
                                 persistence=db, epoch=db.runtime.begin_epoch(instance_id='t'))
    live = authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='u'), 'route')
    first, _ = db.admissions.admit(epoch=authority.epoch, principal_id='u', session_id='s',
                                   request_id='r1', payload={'text': 'first'})
    cancelled, _ = db.admissions.admit(epoch=authority.epoch, principal_id='u', session_id='s',
                                       request_id='r2', payload={'text': 'cancel me'})
    ref = SessionRef('p', 's')
    actor = SessionActor('u', 'p', frozenset({'session:submit', 'session:control'}), 'test')
    gate = LockProbe()
    monkeypatch.setattr(live, 'pending_snapshot_lock', gate)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    loop = asyncio.get_running_loop()
    snapshot_ready, cancel_entered = asyncio.Event(), asyncio.Event()
    release_snapshot, release_cancel_write = threading.Event(), threading.Event()
    cancel_committed = asyncio.Event()
    frames, commit_frame_count, commit_lock_holder = [], [], []
    frames_lock = threading.Lock()
    real_list, real_cancel = SQLiteAdmissionRepository.list, SQLiteAdmissionRepository.cancel
    list_calls = 0
    list_calls_lock = threading.Lock()

    def observe(frame):
        with frames_lock:
            frames.append(frame['params'])

    live.event_stream.observers.add(observe)

    def paused_list(repository, *args, **kwargs):
        nonlocal list_calls
        rows = real_list(repository, *args, **kwargs)
        with list_calls_lock:
            list_calls += 1
            call = list_calls
        if call == 2:  # the post-claim publication snapshot, after the initial FIFO read
            assert any(row.admission_id == cancelled.admission_id and row.status == 'queued' for row in rows)
            loop.call_soon_threadsafe(snapshot_ready.set)
            if not release_snapshot.wait(10):
                raise AssertionError('test did not release the claim pending snapshot')
        return rows

    def commit_then_pause(repository, *args, **kwargs):
        record = real_cancel(repository, *args, **kwargs)
        with frames_lock:
            commit_frame_count.append(len(frames))
            commit_lock_holder.append(gate.holder())
        loop.call_soon_threadsafe(cancel_committed.set)
        if not release_cancel_write.wait(10):
            raise AssertionError('test did not release the committed cancellation')
        return record

    monkeypatch.setattr(SQLiteAdmissionRepository, 'list', paused_list)
    monkeypatch.setattr(SQLiteAdmissionRepository, 'cancel', commit_then_pause)
    real_cancel_queued = authority.cancel_queued

    async def observed_cancel_queued(actor, ref, admission_id):
        cancel_entered.set()
        return await real_cancel_queued(actor, ref, admission_id)

    monkeypatch.setattr(authority, 'cancel_queued', observed_cancel_queued)
    claim_task = asyncio.create_task(authority._claim_next(ref, live))
    cancel_task = None
    cancel_mutation_task = None
    request_task = commit_wait_task = None
    try:
        assert await asyncio.wait_for(gate.requests.get(), 10) == 1
        assert await asyncio.wait_for(gate.acquisitions.get(), 10) == 1
        await asyncio.wait_for(snapshot_ready.wait(), 10)
        mutation_tasks_before = set(getattr(authority, '_mutation_tasks', ()))
        cancel_task = asyncio.create_task(authority.cancel_queued(actor, ref, cancelled.admission_id))
        await asyncio.wait_for(cancel_entered.wait(), 10)
        request_task = asyncio.create_task(gate.requests.get())
        commit_wait_task = asyncio.create_task(cancel_committed.wait())
        done, _ = await asyncio.wait((request_task, commit_wait_task), timeout=10,
                                     return_when=asyncio.FIRST_COMPLETED)
        assert done, 'cancel neither joined the claim mutation lock nor reached its database commit'
        serialized = request_task in done
        if serialized:
            assert request_task.result() == 2
            assert not cancel_committed.is_set(), 'cancel committed while the claim still owned the pending-snapshot lock'
        else:
            assert cancel_committed.is_set()
            assert commit_lock_holder == [1], 'expected the old path to commit while the claim held the lock'
        cancel_mutation_tasks = set(getattr(authority, '_mutation_tasks', ())) - mutation_tasks_before
        assert len(cancel_mutation_tasks) == 1, f'expected one tracked cancellation operation, got {cancel_mutation_tasks}'
        cancel_mutation_task = next(iter(cancel_mutation_tasks))
        cancel_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancel_task

        release_snapshot.set()
        claimed_row, pending_before = await asyncio.wait_for(claim_task, 15)
        assert pending_before['admission_id'] == first.admission_id
        assert claimed_row and claimed_row['admission_id'] == first.admission_id
        if serialized:
            assert await asyncio.wait_for(gate.acquisitions.get(), 10) == 2
        if not cancel_committed.is_set():
            await asyncio.wait_for(commit_wait_task, 15)
            assert cancel_committed.is_set()
        release_cancel_write.set()
        result = await asyncio.wait_for(cancel_mutation_task, 15)
        assert result['status'] == 'terminal' and result['outcome'] == 'cancelled'

        with frames_lock:
            after_commit = list(frames[commit_frame_count[0]:])
        stale = [params for params in after_commit
                 if params['type'] == 'session.info'
                 and any(row['admission_id'] == cancelled.admission_id
                         for row in params.get('payload', {}).get('pending', []))]
        final_projection = [params for params in after_commit if params['type'] == 'session.info']
        assert not stale, f'claim publication reintroduced a durably cancelled queued row: {stale}'
        assert serialized, 'queued cancellation did not join the claim pending-snapshot lock'
        assert commit_lock_holder == [2], f'cancel commit ran under the wrong pending-snapshot-lock owner: {commit_lock_holder}'
        before_cancel_commit = frames[:commit_frame_count[0]]
        starts = [i for i, params in enumerate(before_cancel_commit)
                  if params['type'] == 'message.start' and params.get('admission_id') == first.admission_id]
        claim_infos = [i for i, params in enumerate(before_cancel_commit)
                       if params['type'] == 'session.info'
                       and any(row['admission_id'] == cancelled.admission_id
                               for row in params.get('payload', {}).get('pending', []))]
        assert starts and claim_infos and starts[0] < claim_infos[0], (
            'follower cancellation must not publish a pending snapshot before the claim start frame'
        )
        assert final_projection and all(
            row['admission_id'] != cancelled.admission_id
            for params in final_projection for row in params.get('payload', {}).get('pending', [])), (
                'the settled cancellation projection still lists the cancelled admission'
            )
    finally:
        release_snapshot.set()
        release_cancel_write.set()
        for task in (request_task, commit_wait_task):
            if task is not None and not task.done():
                task.cancel()
        pending = [task for task in (claim_task, cancel_task, cancel_mutation_task)
                   if task is not None and not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        db.close()
