"""Fixed upgrade cohort, exact retirement evidence, and pre-ingress refusal."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.hosted_room_input_custody import initialize_input_custody, _READY
from gateway.session_ingress_media import capture_native_media, release_admission_media, _media_root
from hermes_state_runtime import admit_session_input, claim_session_input, get_session_admission, settle_session_input
from hermes_state_terminal import ADMISSION_PREFIX
from tests.gateway.test_native_media_budget import _authority


def candidate(home, owner, name, data):
    path = home / name
    path.write_bytes(data)
    refs = capture_native_media([path])
    owner.db.create_session(name, source='telegram')
    row = admit_session_input(owner.db, epoch=owner.epoch, principal_id='native', session_id=name,
        request_id=name, payload={'text': name, 'native_text_v1': {'media': refs}})
    started = claim_session_input(owner.db, epoch=owner.epoch, session_id=name)
    settle_session_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'],
        generation=started['generation'], outcome='completed')
    return row, Path(refs[0]['path'])


@pytest.mark.parametrize('named', [False, True])
def test_fixed_cohort_holds_only_old_paths_until_exact_ids_retire(tmp_path, monkeypatch, named):
    home = tmp_path / 'profiles' / 'member' if named else tmp_path
    home.mkdir(parents=True, exist_ok=True)
    db, owner = _authority(home, monkeypatch)
    with db:
        old, old_path = candidate(home, owner, 'old.txt', b'old document')
        db.create_session('legacy', source='cli')
        legacy = admit_session_input(db, epoch=owner.epoch, principal_id='human', session_id='legacy',
            request_id='hosted:old', payload={'text': 'opaque old document reference ' + str(old_path)})
        sibling = admit_session_input(db, epoch=owner.epoch, principal_id='human', session_id='legacy',
            request_id='hosted:old-sibling', payload={'text': 'another opaque reference'})
        initialize_input_custody(db)
        future, future_path = candidate(home, owner, 'future.txt', b'old document')
        admit_session_input(db, epoch=owner.epoch, principal_id='human', session_id='s',
            request_id='hosted:future', payload={'text': 'new hosted text'})
        initialize_input_custody(db)
        with db._lock:
            db._conn.execute('VACUUM')
        initialize_input_custody(db)
        assert release_admission_media(db, old['admission_id']) == 0
        assert release_admission_media(db, future['admission_id']) == 1 and not future_path.exists()
        assert get_session_admission(db, admission_id=legacy['admission_id'])['payload'] == legacy['payload']
        with db._read_ctx() as conn:
            assert {r[0] for r in conn.execute('SELECT admission_id FROM gateway_legacy_input_admissions')} == {
                legacy['admission_id'], sibling['admission_id']}
        started = claim_session_input(db, epoch=owner.epoch, session_id='legacy')
        settle_session_input(db, epoch=owner.epoch, admission_id=legacy['admission_id'],
            generation=started['generation'], outcome='completed')
        assert release_admission_media(db, old['admission_id']) == 0  # Terminal is not retired.
        with db._read_ctx() as conn:
            retired = dict(conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (legacy['admission_id'],)).fetchone())
        retired.update(payload_json='{}', lineage_json='[]')
        key = ADMISSION_PREFIX + legacy['admission_id']
        # Preconstructed retirement evidence, not a source-close operation.
        def seed_retirement(conn):
            conn.execute('DELETE FROM session_admissions WHERE admission_id=?', (legacy['admission_id'],))
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, json.dumps({**retired, 'payload_digest': 'wrong'})))
        db._execute_write(seed_retirement)
        assert release_admission_media(db, old['admission_id']) == 0
        db._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?', ('[1]', key)))
        assert release_admission_media(db, old['admission_id']) == 0
        db._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(retired), key)))
        assert release_admission_media(db, old['admission_id']) == 0
        started = claim_session_input(db, epoch=owner.epoch, session_id='legacy')
        settle_session_input(db, epoch=owner.epoch, admission_id=sibling['admission_id'],
            generation=started['generation'], outcome='completed')
        with db._read_ctx() as conn:
            sibling_retired = dict(conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (sibling['admission_id'],)).fetchone())
        sibling_retired.update(payload_json='{}', lineage_json='[]')
        def seed_sibling(conn):
            conn.execute('DELETE FROM session_admissions WHERE admission_id=?', (sibling['admission_id'],))
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                (ADMISSION_PREFIX + sibling['admission_id'], json.dumps(sibling_retired)))
        db._execute_write(seed_sibling)
        assert release_admission_media(db, old['admission_id']) == 1 and not old_path.exists()
        initialize_input_custody(db)
        with db._read_ctx() as conn:
            assert conn.execute('SELECT count(*) FROM gateway_legacy_input_admissions').fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('uncertain', ['digest-directory', 'file'])
async def test_uncertain_inventory_refuses_before_epoch_or_readiness(tmp_path, monkeypatch, uncertain):
    from gateway import session_authority
    from hermes_state_runtime import RuntimeStoreError
    db, owner = _authority(tmp_path, monkeypatch)
    with db:
        row, path = candidate(tmp_path, owner, 'old.txt', b'old document')
        admit_session_input(db, epoch=owner.epoch, principal_id='human', session_id='s',
            request_id='hosted:old', payload={'text': 'opaque'})
        outside = tmp_path / 'outside'
        outside.mkdir()
        if uncertain == 'digest-directory':
            link = _media_root() / ('0' * 64)
            link.symlink_to(outside, target_is_directory=True)
        else:
            (path.parent / 'unreadable-link').symlink_to(outside)
        def forbidden_epoch(*args, **kwargs):
            raise AssertionError('inventory must finish before a new owner epoch')
        monkeypatch.setattr(session_authority, 'begin_runtime_epoch', forbidden_epoch)
        with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
            await session_authority.initialize_session_authority(SimpleNamespace(), profile_id='owned', instance_id='new', db=db)
        with db._read_ctx() as conn:
            assert conn.execute('SELECT 1 FROM state_meta WHERE key=?', (_READY,)).fetchone() is None
        assert release_admission_media(db, row['admission_id']) == 0 and path.exists()
