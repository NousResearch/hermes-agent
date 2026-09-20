"""Pending peer retirement must survive the real outbox reclamation siblings."""
import json
import os
import stat
import time

import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_discard import custody, wire_discard
from gateway.hosted_room_artifacts import RoomArtifactOutbox
from gateway.session_results import admission_result, _RESULT_PREFIX
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


def saved(c):
    return admission_result(c.target.db, c.row['admission_id'])['peer_output_discard']


def replace_record(c, record):
    value = admission_result(c.target.db, c.row['admission_id'])
    value['peer_output_discard'] = record
    c.target.db._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?',
        (json.dumps(value), _RESULT_PREFIX + c.row['admission_id'])))


@pytest.mark.asyncio
@pytest.mark.parametrize('missing', [0, 1, 2])
async def test_pending_cleanup_survives_aged_constructor_and_partial_row_loss(files_target, monkeypatch, missing):
    async with peer_case(files_target, monkeypatch, output_count=2) as c:
        scope = custody(c)[0]
        outbox = c.target.adapter._peer_output_outbox
        rows = c.target.db._conn.execute('SELECT artifact_id, blob_name FROM hosted_room_output_artifacts WHERE scope_key=?', (scope.key,)).fetchall()
        paths = [outbox.blob_root / row['blob_name'] for row in rows]
        original = os.unlink
        def fail_blob(path, *args, **kwargs):
            if str(path).split('/')[-1].startswith('blob_'):
                raise OSError('inert persistent unlink failure')
            return original(path, *args, **kwargs)
        with monkeypatch.context() as fault:
            fault.setattr(os, 'unlink', fail_blob)
            with pytest.raises(PeerRunsHTTPError) as error:
                await wire_discard(c)
            assert error.value.status_code == 503 and error.value.retryable
            old = time.time() - 200000
            c.target.db._execute_write(lambda conn: conn.execute('UPDATE hosted_room_output_artifacts SET acknowledged_at=?, created_at=?, receipt_expires_at=? WHERE scope_key=?', (old, old, old, scope.key)))
            for path in paths:
                os.utime(path, (old, old))
            # This is the reviewer's actual same-live-owner constructor schedule,
            # not a direct replay with all initial rows conveniently untouched.
            RoomArtifactOutbox(outbox.db_path, root=outbox.root)
            assert all(path.exists() for path in paths)
            assert c.target.db._conn.execute('SELECT count(*) FROM hosted_room_output_artifacts WHERE scope_key=?', (scope.key,)).fetchone()[0] == len(rows)
            assert saved(c)['state'] == 'pending'
            assert {blob['blob_name'] for blob in saved(c)['blobs']} == {row['blob_name'] for row in rows}
        # Even if an older/prior consumer lost some or all artifact rows, the
        # retained exact per-blob intent must still remove every private byte.
        for row in rows[:missing]:
            c.target.db._execute_write(lambda conn, row=row: conn.execute('DELETE FROM hosted_room_output_artifacts WHERE artifact_id=?', (row['artifact_id'],)))
        assert c.target.db._conn.execute('SELECT count(*) FROM hosted_room_output_artifacts WHERE scope_key=?', (scope.key,)).fetchone()[0] == len(rows) - missing
        if not missing:
            # A healthy retry consumer may remove rows after unlink. It must
            # not erase the Run's independent pending per-blob authority.
            RoomArtifactOutbox(outbox.db_path, root=outbox.root)
            assert not any(path.exists() for path in paths)
            assert saved(c)['state'] == 'pending' and len(saved(c)['blobs']) == len(paths)
        assert await wire_discard(c) == {'discarded': True, 'removed': len(paths)}
        assert not any(path.exists() for path in paths)
        assert saved(c)['state'] == 'completed'
        assert saved(c)['blobs'] == []
        # Ordinary pruning is allowed after positive completion. Replay is a
        # typed retained result, not a requirement to reread deleted files/rows.
        RoomArtifactOutbox(outbox.db_path, root=outbox.root)
        assert await wire_discard(c) == {'discarded': True, 'removed': len(paths)}


@pytest.mark.asyncio
@pytest.mark.parametrize('fault_kind', ['old', 'missing_evidence', 'partial_evidence', 'invalid_receipt', 'fsync', 'completion_sql'])
async def test_incomplete_cleanup_evidence_or_commit_never_confirms(files_target, monkeypatch, fault_kind):
    async with peer_case(files_target, monkeypatch, output_count=2) as c:
        outbox = c.target.adapter._peer_output_outbox
        paths = [outbox.blob_root / r[0] for r in c.target.db._conn.execute('SELECT blob_name FROM hosted_room_output_artifacts')]
        original = os.unlink
        def fail_blob(path, *args, **kwargs):
            if str(path).split('/')[-1].startswith('blob_'):
                raise OSError('inert unlink failure')
            return original(path, *args, **kwargs)
        with monkeypatch.context() as fault:
            fault.setattr(os, 'unlink', fail_blob)
            with pytest.raises(PeerRunsHTTPError):
                await wire_discard(c)
        record = saved(c)
        if fault_kind in ('old', 'missing_evidence', 'partial_evidence', 'invalid_receipt'):
            broken = dict(record)
            if fault_kind == 'old':
                broken = {k: record[k] for k in ('commitment', 'receipt')}
            elif fault_kind == 'invalid_receipt':
                broken['receipt'] = dict(discarded=True, removed=True)
            elif fault_kind == 'missing_evidence':
                broken.pop('blobs', None)
            else:
                broken['blobs'] = broken.get('blobs', [])[:1]
            replace_record(c, broken)
            with pytest.raises(PeerRunsHTTPError) as error:
                await wire_discard(c)
            assert error.value.status_code == 503 and error.value.retryable
            assert all(path.exists() for path in paths)
            replace_record(c, record)
        else:
            with monkeypatch.context() as fault:
                if fault_kind == 'fsync':
                    real_fsync = os.fsync
                    def fail_directory(fd):
                        if stat.S_ISDIR(os.fstat(fd).st_mode):
                            raise OSError('inert directory sync failure')
                        return real_fsync(fd)
                    fault.setattr(os, 'fsync', fail_directory)
                else:
                    c.target.db._execute_write(lambda conn: conn.execute("CREATE TRIGGER reject_completed BEFORE UPDATE ON state_meta WHEN json_extract(NEW.value, '$.peer_output_discard.state')='completed' BEGIN SELECT RAISE(ABORT, 'inert completion SQL failure'); END"))
                with pytest.raises(PeerRunsHTTPError) as error:
                    await wire_discard(c)
                assert error.value.status_code == 503 and error.value.retryable
            assert saved(c)['state'] == 'pending'
            assert len(saved(c)['blobs']) == len(paths)
            if fault_kind == 'completion_sql':
                c.target.db._execute_write(lambda conn: conn.execute('DROP TRIGGER reject_completed'))
        assert await wire_discard(c) == {'discarded': True, 'removed': len(paths)}
        assert not any(path.exists() for path in paths)
        assert saved(c)['state'] == 'completed'
