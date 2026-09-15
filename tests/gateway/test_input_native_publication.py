"""#111362: real overlapping publication and ordered equal-image custody."""
import asyncio
import hashlib
import os
from pathlib import Path
from threading import Event

import pytest

from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.hosted_room_driver import TaskIdentity
from gateway.hosted_room_input_preparation import prepare_hosted_input
from gateway.hosted_room_input_reclamation import (
    collect_legacy_input_aliases, initialize_working_copies, verified_identity,
)
from gateway.session_contract import Submission
from gateway import session_ingress_media as native
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from tests.gateway.input_reclamation_fixtures import owned, close, rpc_files, expire


def params(bound):
    return dict(task=TaskIdentity('room', 'task', 'thread', 'turn'), execution_generation=1,
        prompt='read', attachments=[item for item, _ in bound], on_terminal=lambda value: None)


def native_row(db):
    rows = db._conn.execute("SELECT * FROM input_custody_copies WHERE namespace='native'").fetchall()
    assert len(rows) == 1
    return dict(rows[0])


def reopen(db, owner, home):
    db.close()
    db = SessionDB(db_path=home / 'state.db')
    owner.db = db
    initialize_working_copies(db, epoch=owner.epoch)
    return db


async def settle(owner, rpc, monkeypatch):
    async def inert(*args):
        return 'inert result'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', inert)
    await owner._drain(rpc.ref)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['refusal', 'after_publish', 'before_commit'])
@pytest.mark.parametrize('live_b', [False, True])
async def test_competing_winner_survives_failed_handoff_then_reclaims(tmp_path, monkeypatch, failure, live_b):
    db, owner = owned(tmp_path, monkeypatch)
    planned, release = Event(), Event()
    task = None
    try:
        rpc, bound = rpc_files(tmp_path, owner, count=2, image=True)
        digest = hashlib.sha256(bound[-1][1]).hexdigest()
        image = native._media_root() / digest / (digest + '.png')
        assert not image.exists()
        executed = db._execute_write
        published = native._publish_native_media
        observations = {}

        def observe_write(operation, *args, **kwargs):
            def transaction(conn):
                result = operation(conn)
                if (failure == 'before_commit' and observations.get('reused')
                        and native._preparation_capture.get() is not None):
                    observations['rolled_back'] = True
                    raise RuntimeError('fixture identity commit rollback')
                return result
            result = executed(transaction, *args, **kwargs)
            # Observe the committed native intent, outside the non-recursive writer lock.
            if native._preparation_capture.get() is not None and not planned.is_set():
                with db._read_ctx() as conn:
                    row = conn.execute("SELECT * FROM input_custody_copies WHERE namespace='native'").fetchone()
                if row is not None:
                    observations['intent'] = dict(row)
                    observations['staging'] = [p.stat().st_ino for p in native._media_root().glob('.capture-*')]
                    planned.set()
                    assert release.wait(15), 'fixture publisher barrier timed out'
            return result

        def observe_publish(staged, references):
            published(staged, references)
            if native._preparation_capture.get() is not None:
                observations['reused'] = verified_identity(image, digest, len(bound[-1][1]))
                if failure == 'after_publish':
                    raise RuntimeError('fixture after actual reuse')

        with monkeypatch.context() as patch:
            patch.setattr(db, '_execute_write', observe_write)
            patch.setattr(native, '_publish_native_media', observe_publish)
            authorization = []
            def refuse(*args):
                authorization.append(args)
                return False
            rpc.authorizer = refuse
            task = asyncio.create_task(rpc._submit(params(bound)))
            assert await asyncio.to_thread(planned.wait, 15), 'fixture never reached committed intent'
            assert not image.exists()
            # Normal unscoped capture, on the owner loop, with exactly A's native target.
            from gateway.platforms.base import get_image_cache_dir
            source = get_image_cache_dir() / image.name
            receipt = await owner.submit(rpc.principal, Submission('normal-b', rpc.ref,
                {'text': 'keep', 'attachments': [{'path': str(source), 'mime': 'image/png'}]}, 'queue'))
            winner = verified_identity(image, digest, len(bound[-1][1]))
            assert int(winner[1]) not in observations['staging']
            release.set()
            expected = RuntimeStoreError if failure == 'refusal' else RuntimeError
            with pytest.raises(expected, match='permission_denied|fixture'):
                await asyncio.wait_for(asyncio.shield(task), 20)
            assert observations['reused'] == winner
            assert bool(authorization) is (failure == 'refusal')
            assert bool(observations.get('rolled_back')) is (failure == 'before_commit')
        after_handoff = native_row(db)
        assert not list(native._media_root().glob('.capture-*'))
        assert image.stat().st_nlink == 1
        assert db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 1
        if not live_b:
            await settle(owner, rpc, monkeypatch)
            assert get_session_admission(db, admission_id=receipt.admission_id)['status'] == 'terminal'
            assert image.exists(), 'A lease must outlive ordinary B terminal release'
        db._execute_write(lambda conn: conn.execute('UPDATE input_custody_preparations SET expires_at=0'))
        db = reopen(db, owner, tmp_path)
        result = collect_legacy_input_aliases(db, epoch=owner.epoch)
        if live_b:
            assert image.exists() and result['removed'] == 0, 'live B must retain provisional or ready bytes'
            await settle(owner, rpc, monkeypatch)
            collect_legacy_input_aliases(db, epoch=owner.epoch)
        assert not image.exists(), 'expired failed handoff must reclaim the competing winner without another capture'
        assert native_row(db)['state'] == 'removed'
        if failure == 'refusal':
            assert (after_handoff['device'], after_handoff['inode']) == winner
        else:
            assert after_handoff['state'] == 'preparing'
            assert (after_handoff['device'], after_handoff['inode']) == (None, None)
        if not live_b:
            assert (native_row(db)['device'], native_row(db)['inode']) == winner
    finally:
        release.set()
        if task is not None and not task.done():
            await asyncio.wait_for(asyncio.shield(task), 20)
        close(db, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize('refused', [False, True])
async def test_cold_equal_images_preserve_order_with_one_physical_owner(tmp_path, monkeypatch, refused):
    from tests.gateway.test_api_media_retention import PNG
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner)  # One document; real distinct image uploads below.
        store = HostedRoomAttachmentStore(db.db_path)
        images = [store.put(room_id='room', upload_id=f'equal-{i}', kind='image', name=f'{name}.png',
            mime='image/png', data=PNG) for i, name in enumerate(('second', 'first'))]
        manifest = [{k: saved[k] for k in ('attachment_id', 'kind', 'name', 'size', 'mime')} for saved in images]
        assert len({item['attachment_id'] for item in manifest}) == 2
        store.commit_message(room_id='room', event_id='images', manifest=manifest, recipient_member_ids=['member'])
        bound += [({**item, 'event_id': 'images'}, PNG) for item in manifest]
        digest = hashlib.sha256(PNG).hexdigest()
        image = native._media_root() / digest / (digest + '.png')
        assert not image.exists()
        from gateway import hosted_room_input_preparation as preparation
        resolve = preparation.resolve_inputs
        resolved = []
        def observe_inputs(*args, **kwargs):
            inputs = resolve(*args, **kwargs)
            resolved.append([item['attachment_id'] for item, _ in inputs])
            return inputs
        monkeypatch.setattr(preparation, 'resolve_inputs', observe_inputs)
        normalized = []
        from gateway.session_submission_payload import normalize_submission_payload
        def observe_payload(*args, **kwargs):
            payload = normalize_submission_payload(*args, **kwargs)
            normalized.append(payload)
            return payload
        monkeypatch.setattr('gateway.session_submission_payload.normalize_submission_payload', observe_payload)
        rpc.authorizer = lambda *args: not refused
        if refused:
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                await rpc._submit(params(bound))
            payload = normalized[-1]
        else:
            receipt = await rpc._submit(params(bound))
            payload = get_session_admission(db, admission_id=receipt['admission_id'])['payload']
        assert resolved == [[item['attachment_id'] for item, _ in bound]]
        expected = {'path': str(image), 'sha256': digest, 'size': len(PNG)}
        assert payload['attachments_v1'] == {'media': [expected, expected], 'media_types': ['image/png', 'image/png']}
        copy = native_row(db)
        assert copy['state'] == 'ready'
        assert (copy['device'], copy['inode']) == verified_identity(image, digest, len(PNG))
        assert db._conn.execute('SELECT count(*) FROM input_custody_native_items').fetchone()[0] == 1
        if refused:
            db._execute_write(lambda conn: conn.execute('UPDATE input_custody_preparations SET expires_at=0'))
            db = reopen(db, owner, tmp_path)
            collect_legacy_input_aliases(db, epoch=owner.epoch)
        else:
            await settle(owner, rpc, monkeypatch)
        assert not image.exists()
    finally:
        close(db, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation', ['replace', 'symlink', 'hardlink', 'missing'])
async def test_ready_native_identity_never_becomes_provisional_deletion_authority(tmp_path, monkeypatch, mutation):
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner, count=2, image=True)
        args = dict(prompt='read', attachments=[item for item, _ in bound])
        prepared = prepare_hosted_input(rpc, request_id='first', **args)
        before = native_row(db)
        image = native._media_root() / before['digest'] / before['name']
        other = tmp_path / 'original.png'
        if mutation == 'hardlink':
            os.link(image, other)
        else:
            image.rename(other)
            if mutation == 'replace':
                image.write_bytes(bound[-1][1])
            elif mutation == 'symlink':
                image.symlink_to(other)
        if mutation != 'hardlink':
            with pytest.raises(RuntimeStoreError, match='input_preparation_busy|storage_unavailable'):
                prepare_hosted_input(rpc, request_id='second', **args)
        expire(db, prepared.handle)
        db._execute_write(lambda conn: conn.execute('UPDATE input_custody_preparations SET expires_at=0'))
        db = reopen(db, owner, tmp_path)
        if mutation == 'missing':
            # Normal native settlement can remove a bound path; a new generation may publish anew.
            next_prepared = prepare_hosted_input(rpc, request_id='third', **args)
            after = native_row(db)
            assert after['generation'] == before['generation'] + 1
            assert (after['device'], after['inode']) != (before['device'], before['inode'])
            expire(db, next_prepared.handle)
        try:
            collect_legacy_input_aliases(db, epoch=owner.epoch)
        except RuntimeStoreError as exc:
            assert mutation == 'symlink' and exc.reason == 'storage_unavailable'
        assert image.exists() is (mutation != 'missing')
        assert other.read_bytes() == bound[-1][1]
        if mutation != 'missing':
            after = native_row(db)
            assert (after['generation'], after['device'], after['inode']) == (
                before['generation'], before['device'], before['inode'])
    finally:
        close(db, tmp_path)
