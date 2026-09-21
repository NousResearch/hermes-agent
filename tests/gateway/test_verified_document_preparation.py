"""Real G bytes and final API admission identity; no API adapter or executor."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from gateway.hosted_room_input_preparation import prepare_verified_documents
from gateway.hosted_room_input_reclamation import collect_working_copies
from hermes_state_runtime import RuntimeStoreError, admit_session_input
from tests.gateway.input_reclamation_fixtures import owned, close, retire_metadata


def documents(tmp_path):
    result = []
    for name, data in [('one.txt', b'first verified bytes'), ('two.txt', b'second verified bytes')]:
        (tmp_path / name).write_bytes(data)
        result.append({'name': name, 'data': data, 'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)})
    return result


def test_final_api_payload_is_bound_once_and_replay_needs_no_preparation(tmp_path, monkeypatch):
    db, authority = owned(tmp_path, monkeypatch)
    try:
        inputs = documents(tmp_path)
        built = []
        def final_payload(refs):
            built.append(refs)
            return {'text': 'original API prompt', 'api_turn_v1': {'history': [],
                'settings': {'room_input_media': {'media': list(refs)}}, 'run_owner_scope': 'a' * 64}}
        prepared = prepare_verified_documents(authority, principal_id='api', session_id='s',
            request_id='api-request', documents=inputs, build_payload=final_payload)
        assert len(built) == 1
        assert prepared.payload['text'] == 'original API prompt'
        assert 'token' not in json.dumps(prepared.payload)
        for reference, item in zip(built[0], inputs):
            assert Path(reference['path']).read_bytes() == item['data']
            assert not os.path.samefile(reference['path'], tmp_path / item['name'])
            assert Path(reference['path']).stat().st_nlink == 1
            (tmp_path / item['name']).unlink()
        args = dict(epoch=authority.epoch, principal_id='api', session_id='s', request_id='api-request',
                    payload=prepared.payload)
        # Grant denial and wrong final settings must leave the ready private copies unaccepted.
        def deny(conn):
            assert conn.in_transaction
            assert conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
            raise PermissionError('revoked')
        with pytest.raises(PermissionError, match='revoked'):
            admit_session_input(db, **args, input_custody=prepared.handle, _authorize_write=deny)
        changed = json.loads(json.dumps(prepared.payload))
        changed['api_turn_v1']['history'] = [{'role': 'user', 'content': 'changed'}]
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            admit_session_input(db, **(args | {'payload': changed}), input_custody=prepared.handle,
                _authorize_write=lambda conn: None)
        assert db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
        assert db._conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == 0
        row = admit_session_input(db, **args, input_custody=prepared.handle, _authorize_write=lambda conn: None)
        raw = dict(db._conn.execute('SELECT * FROM session_admissions').fetchone())
        refs = [dict(r) for r in db._conn.execute('SELECT * FROM input_custody_refs ORDER BY ordinal')]
        assert len(refs) == len(inputs)
        for ref in refs:
            for key in ('principal_id', 'target_session_id', 'request_id', 'payload_digest', 'intent'):
                assert ref[key] == raw[key]
            copy = db._conn.execute('SELECT * FROM input_custody_copies WHERE copy_id=?', (ref['copy_id'],)).fetchone()
            assert copy['generation'] == ref['generation']
        def no_copy(*args, **kwargs):
            pytest.fail('replay recaptured private bytes')
        monkeypatch.setattr('gateway.hosted_room_input_preparation._copy', no_copy)
        for retired in (False, True):
            if retired:
                retire_metadata(db, row['admission_id'])
                from hermes_state_mutation_retirement import RETIRED_PREFIX
                db._execute_write(lambda conn: conn.execute(
                    'INSERT INTO state_meta(key,value) VALUES(?,?)', (RETIRED_PREFIX + 's', '{}')))
            replay = admit_session_input(db, **args, _authorize_write=deny)
            assert replay['admission_id'] == row['admission_id']
            assert [dict(r) for r in db._conn.execute('SELECT * FROM input_custody_refs ORDER BY ordinal')] == refs
        assert len(built) == 1
    finally:
        close(db, tmp_path)


@pytest.mark.parametrize('mismatch', ['principal_id', 'session_id', 'request_id', 'generation', 'bytes'])
def test_verified_documents_refuse_wrong_identity_or_bytes(tmp_path, monkeypatch, mismatch):
    db, authority = owned(tmp_path, monkeypatch)
    try:
        inputs = documents(tmp_path)
        prepare = dict(principal_id='api', session_id='s', request_id='api-request', documents=inputs,
            build_payload=lambda refs: {'text': 'api', 'api_turn_v1': {'history': [], 'settings': {'files': list(refs)}}})
        if mismatch == 'bytes':
            inputs[1]['data'] += b'changed'
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                prepare_verified_documents(authority, **prepare)
            assert db._conn.execute('SELECT count(*) FROM input_custody_copies').fetchone()[0] == 0
            return
        prepared = prepare_verified_documents(authority, **prepare)
        args = dict(epoch=authority.epoch, principal_id='api', session_id='s', request_id='api-request',
            payload=prepared.payload, input_custody=prepared.handle)
        if mismatch == 'generation':
            db._execute_write(lambda conn: conn.execute('UPDATE input_custody_copies SET generation=generation+1'))
        else:
            args[mismatch] = 'other'
            if mismatch == 'session_id':
                db.create_session('other', source='api')
        with pytest.raises(RuntimeStoreError, match='admission_conflict|input_preparation_expired'):
            admit_session_input(db, **args, _authorize_write=lambda conn: None)
        assert db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
        assert db._conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == 0
        # Expired abandoned preparations, unlike accepted refs, may be collected.
        db._execute_write(lambda conn: conn.execute('UPDATE input_custody_preparations SET expires_at=0'))
        if mismatch != 'generation':
            assert collect_working_copies(db, epoch=authority.epoch)['removed'] == 2
    finally:
        close(db, tmp_path)
