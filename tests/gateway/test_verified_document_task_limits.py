"""G task-count alignment: real private custody, final API identity and preflight."""
import hashlib
import json
from pathlib import Path

import pytest

from gateway.hosted_room_attachments import validate_manifest, validate_task_manifest
from gateway.hosted_room_input_preparation import prepare_verified_documents
from gateway.session_admission import admission_fingerprint
from hermes_state_runtime import RuntimeStoreError, admit_session_input
from tests.gateway.input_reclamation_fixtures import close, owned


def document(index, data=b'verified document'):
    return dict(name=f'document-{index}.txt', data=data, size=len(data),
                sha256=hashlib.sha256(data).hexdigest())


def final_payload(refs):
    return {'text': 'task prompt', 'api_turn_v1': {'history': [],
        'settings': {'room_input_media': {'media': list(refs)}}, 'run_owner_scope': 'a' * 64}}


@pytest.mark.parametrize('count', [8, 9, 16])
def test_task_documents_bind_all_ordinals_and_final_api_digest(tmp_path, monkeypatch, count):
    db, authority = owned(tmp_path, monkeypatch)
    try:
        inputs = [document(i, f'input {i}'.encode()) for i in range(count)]
        built = []
        def build(refs):
            built.append(refs)
            return final_payload(refs)
        prepared = prepare_verified_documents(authority, principal_id='api', session_id='s',
            request_id='count-task', documents=inputs, build_payload=build)
        assert len(built) == 1
        references = built[0]
        assert len(references) == count
        assert [Path(ref['path']).read_bytes() for ref in references] == [item['data'] for item in inputs]
        assert all(Path(ref['path']).stat().st_nlink == 1 for ref in references)
        assert prepared.handle.token not in json.dumps(prepared.payload)
        digest = admission_fingerprint(canonical_target='s', payload={'input': prepared.payload, 'intent': 'queue'})
        args = dict(epoch=authority.epoch, principal_id='api', session_id='s', request_id='count-task',
                    payload=prepared.payload, input_custody=prepared.handle)
        def deny(conn):
            assert conn.in_transaction
            raise PermissionError('revoked')
        with pytest.raises(PermissionError, match='revoked'):
            admit_session_input(db, **args, _authorize_write=deny)
        changed = json.loads(json.dumps(prepared.payload))
        changed['api_turn_v1']['history'] = [{'role': 'user', 'content': 'changed'}]
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            admit_session_input(db, **(args | {'payload': changed}), _authorize_write=lambda conn: None)
        assert db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
        assert db._conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == 0
        admitted = admit_session_input(db, **args, _authorize_write=lambda conn: None)
        raw = dict(db._conn.execute('SELECT * FROM session_admissions').fetchone())
        refs = [dict(row) for row in db._conn.execute('SELECT * FROM input_custody_refs ORDER BY ordinal')]
        assert raw['payload_digest'] == digest
        assert [ref['ordinal'] for ref in refs] == list(range(count))
        for ref, item in zip(refs, inputs):
            assert all(ref[k] == raw[k] for k in
                       ('principal_id', 'target_session_id', 'request_id', 'payload_digest', 'intent'))
            copy = db._conn.execute('SELECT * FROM input_custody_copies WHERE copy_id=?', (ref['copy_id'],)).fetchone()
            assert (copy['name'], copy['size'], copy['digest'], copy['generation']) == (
                item['name'], item['size'], item['sha256'], ref['generation'])
        replay = admit_session_input(db, **args, _authorize_write=deny)
        assert replay['admission_id'] == admitted['admission_id']
        assert [dict(row) for row in db._conn.execute('SELECT * FROM input_custody_refs ORDER BY ordinal')] == refs
        assert [Path(ref['path']).read_bytes() for ref in references] == [item['data'] for item in inputs]
    finally:
        close(db, tmp_path)


@pytest.mark.parametrize('case', [
    '17-documents', 'zero-size', 'boolean-size', 'wrong-size', 'wrong-hash', 'per-file-bytes',
    'task-bytes', 'platform-bytes', 'message-count', 'message-bytes', 'mixed-task-bytes',
])
def test_task_preflight_preserves_message_and_byte_limits(tmp_path, monkeypatch, case):
    db, authority = owned(tmp_path, monkeypatch)
    try:
        def forbidden(*args, **kwargs):
            pytest.fail('rejected batch began custody preparation or copied bytes')
        monkeypatch.setattr('gateway.hosted_room_input_preparation.begin_preparation', forbidden)
        monkeypatch.setattr('gateway.hosted_room_input_preparation._copy', forbidden)
        inputs = [document(0)]
        error = 'permission_denied'
        if case == '17-documents':
            inputs = [document(i) for i in range(17)]
            error = 'invalid_params'
        elif case in {'zero-size', 'boolean-size', 'wrong-size', 'per-file-bytes'}:
            inputs[0]['size'] = {'zero-size': 0, 'boolean-size': True, 'wrong-size': 1,
                                 'per-file-bytes': 15_000_001}[case]
        elif case == 'wrong-hash':
            inputs[0]['sha256'] = '0' * 64
        elif case == 'task-bytes':
            # One real buffer, four distinct names: valid hashes/sizes, no huge copies.
            data = b'x' * 12_500_001
            inputs = [document(i, data) for i in range(4)]
            assert all(item['data'] is data for item in inputs)
            (tmp_path / 'config.yaml').write_text('gateway:\n  max_inbound_media_bytes: 0\n')
            error = 'invalid_params'
        elif case == 'platform-bytes':
            inputs = [document(i, b'x' * 16) for i in range(2)]
            (tmp_path / 'config.yaml').write_text('gateway:\n  max_inbound_media_bytes: 31\n')
            error = 'invalid_params'
        else:
            # Real source-message and whole mixed-task validators retain their own budgets.
            sizes = {'message-count': [1] * 9, 'message-bytes': [12_500_000, 12_500_001],
                     'mixed-task-bytes': [12_500_000] * 4 + [1]}[case]
            manifest = [dict(attachment_id=f'att_{i:032x}', name=f'{i}.txt', kind='file',
                             mime='text/plain', size=size) for i, size in enumerate(sizes)]
            if case == 'mixed-task-bytes':
                manifest[-1].update(name='image.png', kind='image', mime='image/png')
                assert validate_task_manifest(manifest[:-1]) == manifest[:-1]
                validator = validate_task_manifest
            else:
                assert validate_manifest(manifest[:-1]) == manifest[:-1]
                validator = validate_manifest
            with pytest.raises(ValueError):
                validator(manifest)
        if case not in {'message-count', 'message-bytes', 'mixed-task-bytes'}:
            with pytest.raises(RuntimeStoreError, match=error):
                prepare_verified_documents(authority, principal_id='api', session_id='s',
                    request_id='refused', documents=inputs, build_payload=forbidden)
        for table in ('input_custody_preparations', 'input_custody_copies', 'input_custody_items',
                      'input_custody_refs', 'session_admissions'):
            assert db._conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0] == 0
    finally:
        close(db, tmp_path)
