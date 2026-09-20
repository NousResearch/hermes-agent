import asyncio
import os
import sqlite3
import threading
from dataclasses import replace

import pytest

from gateway import hosted_rooms
from gateway import session_group_files_revoke as cleanup
from gateway.platforms import api_server_room_attachments as attachments_api
from tests.gateway.test_canonical_peer_files_target import files_target
from tests.gateway.test_canonical_peer_target_setup import target
from tests.gateway.test_canonical_peer_native_consent import _drift
from tests.gateway.platforms.test_room_attachment_staging_provenance import case_for, manifest_request, put_request
from tests.gateway.test_native_revoked_staging_cleanup import stage, revoke, batch, paths, denied, file_rows, assert_reclaimed

__all__ = ['target', 'files_target']

pytestmark = pytest.mark.linux_only


def assert_locked(target, *, shared=True):
    for name in ('shared-state.db', 'state.db') if shared else ('state.db',):
        with sqlite3.connect(target.home / name, timeout=0) as conn:
            with pytest.raises(sqlite3.OperationalError, match='locked'):
                conn.execute('BEGIN IMMEDIATE')


@pytest.mark.asyncio
@pytest.mark.parametrize('drift', ['epoch', 'stored_epoch', 'registry', 'actor', 'transport', 'native_binding', 'close', 'instance'])
async def test_frozen_native_owner_cannot_refresh_for_cleanup(files_target, monkeypatch, drift):
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    entered, resume = threading.Event(), threading.Event()
    original = cleanup._collect

    def held(*args):
        entered.set()
        assert resume.wait(10)
        return original(*args)

    monkeypatch.setattr(cleanup, '_collect', held)
    pending = asyncio.create_task(revoke(files_target, case))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        denied(files_target, case)
        if drift == 'instance':
            files_target.authority.instance_id = 'replacement-instance'
        else:
            await _drift(files_target, drift)
    finally:
        resume.set()
    result = await asyncio.wait_for(pending, 10)
    assert result['result']['revoked']
    denied(files_target, case)
    assert batch(case) is not None
    assert [p.read_bytes() for p in paths(case)] == case.raw


@pytest.mark.asyncio
@pytest.mark.parametrize('missing', ['shared-state.db', 'state.db'])
async def test_positive_both_store_evidence_is_required_after_deny(files_target, monkeypatch, missing):
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    original = cleanup._collect

    def changed(*args):
        with sqlite3.connect(files_target.home / missing) as conn:
            conn.execute('DELETE FROM hosted_room_revoked_grant_tokens')
            conn.execute('DELETE FROM hosted_room_peer_reservations')
        return original(*args)

    monkeypatch.setattr(cleanup, '_collect', changed)
    assert (await revoke(files_target, case))['result']['revoked']
    assert batch(case) is not None and [p.read_bytes() for p in paths(case)] == case.raw
    for name in ('shared-state.db', 'state.db'):
        assert hosted_rooms.room_grant_is_revoked(files_target.home / name, claims=case.claims) == (name != missing)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['sql-stored', 'sql-complete', 'sql-commit', 'rename', 'purge', 'rollback-winner', 'symlink', 'hardlink', 'root', 'uncommitted'])
async def test_reclamation_failure_never_reopens_denial_or_erases_unknown_data(files_target, monkeypatch, failure):
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    before_batch, before_files = batch(case), file_rows(case)
    old_paths = paths(case)
    winner = files_target.home / 'unrelated-winner'
    winner.write_bytes(b'unknown winner')
    original_restore = cleanup._Quarantine.restore
    restored = []

    def restore(q):
        assert_locked(files_target)
        restored.append(True)
        if failure == 'rollback-winner':
            old_paths[0].write_bytes(winner.read_bytes())
        return original_restore(q)

    monkeypatch.setattr(cleanup._Quarantine, 'restore', restore)
    if failure in {'sql-stored', 'sql-complete', 'rollback-winner'}:
        table, column = ('roomlink_attachment_files', 'stored') if failure == 'sql-stored' else ('roomlink_attachment_batches', 'complete')
        with sqlite3.connect(case.spool.db_path) as conn:
            conn.execute(f"CREATE TRIGGER fail_cleanup BEFORE UPDATE OF {column} ON {table} "
                         "BEGIN SELECT RAISE(ABORT, 'fixture cleanup failure'); END")
    elif failure == 'sql-commit':
        connect = attachments_api.RoomAttachmentSpool._connect

        class Connection:
            def __init__(self, conn):
                self.conn = conn
            def __getattr__(self, key):
                return getattr(self.conn, key)
            def commit(self):
                if self.conn.in_transaction and not self.conn.execute('SELECT 1 FROM roomlink_attachment_files WHERE stored=1').fetchone():
                    assert_locked(files_target)
                    raise sqlite3.OperationalError('synthetic commit failure')
                return self.conn.commit()

        monkeypatch.setattr(attachments_api.RoomAttachmentSpool, '_connect', lambda spool: Connection(connect(spool)))
    elif failure == 'rename':
        original_move = cleanup._Quarantine.move
        calls = []
        def move(q, *args):
            calls.append(True)
            if len(calls) == 2:
                raise OSError('synthetic rename failure')
            return original_move(q, *args)
        monkeypatch.setattr(cleanup._Quarantine, 'move', move)
    elif failure == 'purge':
        def purge(q, authorize):
            assert_locked(files_target, shared=False)
            raise OSError('synthetic unlink failure')
        monkeypatch.setattr(cleanup._Quarantine, 'purge', purge)
    elif failure in {'symlink', 'hardlink'}:
        old_paths[0].unlink()
        if failure == 'symlink':
            old_paths[0].symlink_to(winner)
        else:
            winner.write_bytes(case.raw[0])
            os.link(winner, old_paths[0])
    elif failure == 'uncommitted':
        with sqlite3.connect(case.spool.db_path) as conn:
            conn.execute('UPDATE roomlink_attachment_files SET stored=0 WHERE attachment_id=?',
                         (case.manifest[1]['attachment_id'],))
        before_files = file_rows(case)
    elif failure == 'root':
        original = cleanup._collect
        def collect(*args):
            case.spool.root.rename(files_target.home / 'retained-old-spool')
            case.spool.root.mkdir()
            old_paths[0].write_bytes(winner.read_bytes())
            return original(*args)
        monkeypatch.setattr(cleanup, '_collect', collect)
    assert (await revoke(files_target, case))['result']['revoked']
    denied(files_target, case)
    assert winner.read_bytes() == (case.raw[0] if failure == 'hardlink' else b'unknown winner')
    if failure == 'purge':
        assert_reclaimed(case, before_batch, before_files)
        assert sorted(p.read_bytes() for q in case.spool.root.glob('.revoke-*') for p in q.iterdir()) == sorted(case.raw)
    else:
        assert batch(case) == before_batch and file_rows(case) == before_files
        if failure in {'rollback-winner', 'symlink', 'root'}:
            assert old_paths[0].read_bytes() == winner.read_bytes()
        else:
            assert [p.read_bytes() for p in old_paths] == case.raw
        if failure in {'sql-stored', 'sql-complete', 'sql-commit', 'rename', 'rollback-winner', 'uncommitted'}:
            assert restored
        if failure == 'rollback-winner':
            assert [p.read_bytes() for q in case.spool.root.glob('.revoke-*') for p in q.iterdir()] == [case.raw[0]]


@pytest.mark.asyncio
@pytest.mark.parametrize('generation', [1, 2])
@pytest.mark.parametrize('expired', [False, True], ids=['retained-attempt', 'expired-attempt'])
async def test_postcommit_purge_cannot_delete_reused_old_key_or_higher_generation(files_target, monkeypatch, generation, expired):
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    other = attachments_api.RoomAttachmentSpool(case.spool.db_path)
    assert other._lock is not case.spool._lock
    entered, resume = threading.Event(), threading.Event()
    original = cleanup._Quarantine.purge

    def held(q, authorize):
        assert_locked(files_target, shared=False)
        entered.set()
        assert resume.wait(10)
        return original(q, authorize)

    monkeypatch.setattr(cleanup._Quarantine, 'purge', held)
    pending = asyncio.create_task(revoke(files_target, case))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        assert batch(case)['complete'] == 0
        assert all(row['stored'] == 0 for row in file_rows(case))
        other.clock = lambda: case.claims['issued_at'] + (attachments_api.SPOOL_TTL_SECONDS + 1 if expired else 1)
        import hashlib
        from gateway.hosted_room_peer import attachment_manifest_digest
        winner_bytes = case.raw[0] if generation == 1 and not expired else b'different replacement bytes'
        manifest = [case.manifest[0] | {'size': len(winner_bytes), 'sha256': hashlib.sha256(winner_bytes).hexdigest()}, case.manifest[1]]
        successor = replace(case.value, execution_generation=generation, attachment_manifest_digest=attachment_manifest_digest(manifest))
        # This lower-level cross-instance publisher exercises the physical
        # shared-commit boundary. HTTP writers also retain the profile fence
        # and cannot finish until cleanup releases it; no grant is bypassed
        # by the production collector. Real replacement grants are tested above.
        other.prepare(successor, manifest)
        other.put(claims=case.claims, task_id=case.value.task_id, execution_generation=generation,
                  attachment_id=case.manifest[0]['attachment_id'], data=winner_bytes)
        higher = replace(case.value, task_id='higher-task', execution_generation=2)
        other.prepare(higher, case.manifest)
        other.put(claims=case.claims, task_id=higher.task_id, execution_generation=2,
                  attachment_id=case.manifest[1]['attachment_id'], data=case.raw[1])
    finally:
        resume.set()
    assert (await asyncio.wait_for(pending, 10))['result']['revoked']
    assert other._file_path(attachments_api._batch_key(successor), case.manifest[0]['attachment_id']).read_bytes() == winner_bytes
    assert other._file_path(attachments_api._batch_key(higher), case.manifest[1]['attachment_id']).read_bytes() == case.raw[1]
    assert not list(case.spool.root.glob('.revoke-*'))


@pytest.mark.asyncio
@pytest.mark.parametrize('bound', ['MAX_SCAN_BATCHES', 'MAX_CLEAN_BATCHES', 'MAX_CLEAN_FILES', 'MAX_CLEAN_BYTES'])
async def test_collection_bounds_preserve_remainder_and_attempt_fences(files_target, monkeypatch, bound):
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    sibling = type(case)(**(vars(case) | {'value': replace(case.value, task_id='sibling-task')}))
    await stage(sibling)
    with sqlite3.connect(case.spool.db_path) as conn:
        fences = conn.execute('SELECT * FROM roomlink_attachment_attempt_fences ORDER BY task_id').fetchall()
    monkeypatch.setattr(cleanup, bound, 1)
    assert (await revoke(files_target, case))['result']['revoked']
    assert all(batch(c) is not None for c in (case, sibling))
    remaining = sum(any(row['stored'] for row in file_rows(c)) for c in (case, sibling))
    assert remaining == (1 if bound in {'MAX_SCAN_BATCHES', 'MAX_CLEAN_BATCHES'} else 2)
    with sqlite3.connect(case.spool.db_path) as conn:
        assert conn.execute('SELECT * FROM roomlink_attachment_attempt_fences ORDER BY task_id').fetchall() == fences


@pytest.mark.asyncio
async def test_cold_collector_never_prunes_a_replaced_root_before_ownership_check(files_target, monkeypatch):
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    attachments_api._spool.cache_clear()
    original = attachments_api.RoomAttachmentSpool.__init__
    winners = []

    def initialize(spool, *args, **kwargs):
        denied(files_target, case)
        case.spool.root.rename(files_target.home / 'retained-before-construction')
        case.spool.root.mkdir()
        winner = case.spool.root / 'unknown-new-owner-bytes'
        winner.write_bytes(b'new owner winner')
        winners.append(winner)
        original(spool, *args, **kwargs)

    monkeypatch.setattr(attachments_api.RoomAttachmentSpool, '__init__', initialize)
    assert (await revoke(files_target, case))['result']['revoked']
    assert len(winners) == 1 and winners[0].read_bytes() == b'new owner winner'
    assert batch(case) is not None


@pytest.mark.asyncio
async def test_scope_intersects_actual_persisted_cutoffs(files_target, monkeypatch):
    from tests.gateway.platforms.test_room_attachment_staging_provenance import renewed_token
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    token = renewed_token(files_target, case)
    sibling = type(case)(**(vars(case) | {'value': replace(case.value, task_id='newer-token-task'),
                                       'issued': case.issued | {'grant': token}}))
    await stage(sibling)
    original = cleanup._collect

    def collect(*args):
        for name, cutoff in [('shared-state.db', 2), ('state.db', 0.5)]:
            with sqlite3.connect(files_target.home / name) as conn:
                conn.execute('UPDATE hosted_room_revoked_grants SET revoked_before=?',
                             (case.claims['issued_at'] + cutoff,))
        return original(*args)

    monkeypatch.setattr(cleanup, '_collect', collect)
    assert (await revoke(files_target, case, exact=False))['result']['revoked']
    assert batch(case)['complete'] == 0 and not any(p.exists() for p in paths(case))
    assert all(row['stored'] == 0 for row in file_rows(case))
    assert batch(sibling) is not None and [p.read_bytes() for p in paths(sibling)] == case.raw


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['expiry', 'transport'])
async def test_authority_or_deny_expiry_after_first_move_restores_under_both_locks(files_target, monkeypatch, change):
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    original = cleanup._Quarantine.move
    moves = []

    def move(q, *args):
        assert_locked(files_target)
        original(q, *args)
        moves.append(True)
        if change == 'expiry':
            monkeypatch.setattr(hosted_rooms.time, 'time', lambda: case.claims['status_expires_at'] + 1)
        else:
            files_target.authority.events[files_target.connection.actor.transport_id] = object()

    monkeypatch.setattr(cleanup._Quarantine, 'move', move)
    assert (await revoke(files_target, case))['result']['revoked']
    assert len(moves) == 1
    assert batch(case) is not None and [p.read_bytes() for p in paths(case)] == case.raw
    assert not list(case.spool.root.glob('.revoke-*'))
    for path in (files_target.home / 'shared-state.db', files_target.home / 'state.db'):
        with sqlite3.connect(path) as conn:
            assert conn.execute('SELECT token_sha256 FROM hosted_room_revoked_grant_tokens').fetchone()[0] == case.claims['_token_sha256']


@pytest.mark.asyncio
async def test_historical_receiver_epoch_is_not_a_refreshed_operation(files_target, monkeypatch):
    from hermes_state_runtime import begin_runtime_epoch
    from gateway.session_controls import AuthorityConnection
    from unittest.mock import AsyncMock
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    files_target.authority.epoch = begin_runtime_epoch(files_target.db, instance_id='new-owner')
    files_target.authority.instance_id = 'new-owner'
    files_target.connection = AuthorityConnection(files_target.authority, AsyncMock(),
        files_target.identity | {'instance_id': 'new-owner'}, operator=True)
    assert (await revoke(files_target, case))['result']['revoked']
    assert batch(case)['complete'] == 0 and not any(p.exists() for p in paths(case))
    assert all(row['stored'] == 0 for row in file_rows(case))


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['shared-state.db', 'state.db'])
async def test_replaced_database_identity_never_authorizes_deletion(files_target, monkeypatch, name):
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    original = cleanup._collect

    def collect(*args):
        replacement = files_target.home / ('replacement-' + name)
        with sqlite3.connect(files_target.home / name) as source, sqlite3.connect(replacement) as dest:
            source.backup(dest)
        os.replace(replacement, files_target.home / name)
        return original(*args)

    monkeypatch.setattr(cleanup, '_collect', collect)
    assert (await revoke(files_target, case))['result']['revoked']
    assert batch(case) is not None and [p.read_bytes() for p in paths(case)] == case.raw


@pytest.mark.asyncio
async def test_changed_quarantined_bytes_are_retained_as_uncertainty(files_target, monkeypatch):
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    original = cleanup._Quarantine.purge
    retained = []

    def purge(q, authorize):
        name = q.files[0][1]
        descriptor = os.open(name, os.O_WRONLY | os.O_TRUNC, dir_fd=q.fd)
        try:
            os.write(descriptor, b'unexpected changed bytes')
        finally:
            os.close(descriptor)
        retained.append(case.spool.root / q.name / name)
        return original(q, authorize)

    monkeypatch.setattr(cleanup._Quarantine, 'purge', purge)
    assert (await revoke(files_target, case))['result']['revoked']
    denied(files_target, case)
    assert batch(case)['complete'] == 0
    assert all(row['stored'] == 0 for row in file_rows(case))
    assert len(retained) == 1 and retained[0].read_bytes() == b'unexpected changed bytes'


@pytest.mark.asyncio
async def test_unknown_fifo_probe_is_nonblocking_and_preserves_denial(files_target, monkeypatch):
    import stat
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    unknown = paths(case)[0]
    unknown.unlink()
    os.mkfifo(unknown, mode=0o600)
    original = os.open
    observed = []

    def open_file(path, flags, *args, **kwargs):
        if path == unknown.name and kwargs.get('dir_fd') is not None:
            observed.append(flags)
            flags |= os.O_NONBLOCK
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, 'open', open_file)
    monkeypatch.setattr(os, 'supports_dir_fd', os.supports_dir_fd | {open_file})
    assert (await revoke(files_target, case))['result']['revoked']
    denied(files_target, case)
    assert observed and all(flags & os.O_NONBLOCK for flags in observed)
    assert stat.S_ISFIFO(unknown.stat().st_mode)
    assert batch(case) is not None and paths(case)[1].read_bytes() == case.raw[1]
