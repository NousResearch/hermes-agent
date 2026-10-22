"""Public attachments are bounded per admission and their retained bytes are released."""
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import Principal, SessionRef, Submission
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch, list_session_admissions

PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 8


def _authority(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    epoch = begin_runtime_epoch(db, instance_id='current')
    authority = SessionAuthority(SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False)),
                                 profile_id='owned', instance_id='current', db=db, epoch=epoch)
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'route')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    return db, authority


def _staged(tmp_path, name, size):
    from gateway.platforms.base import get_image_cache_dir
    path = get_image_cache_dir() / name
    path.write_bytes(PNG + b'\x00' * max(0, size - len(PNG)))
    return {'path': str(path), 'mime': 'image/png'}


def _submit(authority, request_id, attachments):
    actor = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
    return authority.submit(actor, Submission(request_id=request_id, ref=SessionRef('owned', 's'),
                                              payload={'text': 'hi', 'attachments': attachments}, intent='queue'))


@pytest.mark.asyncio
async def test_attachment_cap_is_per_admission_not_per_file(tmp_path, monkeypatch):
    from gateway.platforms import base
    monkeypatch.setattr(base, 'get_inbound_media_max_bytes', lambda: 3 * 1024)
    db, authority = _authority(tmp_path, monkeypatch)
    with db:
        files = [_staged(tmp_path, f'part{i}.png', 2 * 1024) for i in range(2)]
        # Each file is under the cap; together they are not.
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            await _submit(authority, 'too-big', files)
        assert list_session_admissions(db, session_id='s', pending_only=False) == []
        receipt = await _submit(authority, 'fits', files[:1])
        assert receipt.status == 'queued'


@pytest.mark.asyncio
async def test_retained_bytes_are_released_after_cancel_and_settlement(tmp_path, monkeypatch):
    from gateway.session_ingress_media import _media_root, admission_media_references
    db, authority = _authority(tmp_path, monkeypatch)
    actor = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
    ref = SessionRef('owned', 's')
    with db:
        shared = _staged(tmp_path, 'shared.png', 64)
        first = await _submit(authority, 'first', [shared])
        second = await _submit(authority, 'second', [shared])
        row = next(r for r in list_session_admissions(db, session_id='s') if r['admission_id'] == first.admission_id)
        retained = admission_media_references(row['payload'])[0]['path']
        from pathlib import Path
        assert Path(retained).parent.parent == _media_root() and Path(retained).exists()
        # Same digest, two live rows: cancelling one must keep the bytes the other still needs.
        await authority.cancel_queued(actor, ref, first.admission_id)
        assert Path(retained).exists()
        # Settling the last live referent (through the real drain loop) releases them.
        from gateway import session_finite
        async def execute(authority, ref, row):
            assert Path(retained).exists(), 'bytes must survive until the turn has settled'
            return 'done'
        monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
        await authority._drain(ref)
        settled = next(r for r in list_session_admissions(db, session_id='s', pending_only=False)
                       if r['admission_id'] == second.admission_id)
        assert settled['status'] == 'terminal' and settled['outcome'] == 'completed'
        assert not Path(retained).exists()
        assert not Path(retained).parent.exists()
        assert list(_media_root().iterdir()) == []
