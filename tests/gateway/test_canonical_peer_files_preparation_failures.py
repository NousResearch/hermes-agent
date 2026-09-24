"""Files preparation failures must not become durable, replayable Run receipts.

Real native grants, source Files, HTTP staging and admission; only transport and
post-admission execution are inert. Filesystem failures stay in the owned Home.
"""
import asyncio
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import (
    files_target, files_state, http_admitted_files, source_files,  # noqa: F401
)
from tests.gateway.test_canonical_peer_text_admission import run_request


# All run-keyed state initialized by api_server_runs._initialize_run_state,
# including surfaces not normally populated before admission.
_RUN_STATE = (
    '_run_idempotency_ids', '_stopping_run_ids', '_run_owners', '_run_streams',
    '_run_streams_created', '_active_run_agents', '_active_run_tasks',
    '_run_statuses', '_run_approval_sessions',
)


def run_state(adapter):
    return {name: set(getattr(adapter, name)) for name in _RUN_STATE}


def rows(conn, table):
    return {tuple(row) for row in conn.execute(f'SELECT * FROM {table}')}


async def stage_candidate(target, case, *, mixed):
    from gateway.hosted_room_peer import attachment_manifest_digest
    from tui_gateway.hosted_room_peer_attachments import bound_attachment_payloads
    # Independent source event/copies: never damage the accepted control's bytes.
    source = target.home / 'candidate-source'
    source.mkdir()
    store, _, attachments, data = source_files(source, mixed=mixed)
    from gateway.session_hosted_attachments import append_user_event
    from gateway import hosted_rooms
    from types import SimpleNamespace
    uploads = [store.put(room_id='room-one', upload_id=f'candidate-{index}',
        name=('candidate-' if item['kind'] != 'image' else '') + item['name'],
        kind=item['kind'], mime=item['mime'],
        data=raw + b' candidate private input' if item['kind'] != 'image' else raw)
        for index, (item, raw) in enumerate(zip(attachments, data))]
    manifest = [{key: item[key] for key in ('attachment_id', 'kind', 'name', 'size', 'mime')}
                for item in uploads]
    service = SimpleNamespace(db_path=store.db_path,
        _room=lambda room: hosted_rooms.room_state(store.db_path, room_id=room))
    event = append_user_event(service, room_id='room-one', event_id='candidate-event',
        payload={'text': case.value.prompt, 'thread_id': 'thread-one', 'attachments': manifest},
        gateway_id='home-gateway', epoch=1)
    pending = bound_attachment_payloads(store, 'room-one', 'member-one',
        [dict(item, event_id=event['event_id']) for item in manifest])
    manifest = [{k: v for k, v in item.items() if k != 'data'} for item in pending]
    candidate = replace(case.value, task_id='unaccepted-files', source_event_seq=event['seq'],
        attachment_manifest_digest=attachment_manifest_digest(manifest))
    raw = [item['data'] for item in pending]
    staged = await asyncio.to_thread(case.client.stage_attachments,
        dispatch=candidate.as_mapping(), attachments=[dict(item, data=data)
            for item, data in zip(manifest, raw)], grant=case.issued['grant'])
    assert staged['complete'] is True and len(manifest) == 2
    return candidate


def install_failure(monkeypatch, target, case, candidate, failure, observed):
    from gateway import session_peer_input as bridge
    from gateway import hosted_room_input_preparation as preparation
    from gateway.platforms import api_server_room_attachments as attachments

    blocked = target.home / 'private-io-blocker'
    blocked.write_bytes(b'test-owned non-directory')
    if failure == 'dispose-after-materialize':
        original = case.spool.materialize
        def dispose(dispatch):
            items = original(dispatch)  # Both first reads really succeeded.
            assert dispatch == candidate and len(items) == 2
            assert all(Path(item['path']).is_file() for item in items)
            claims = target.adapter._room_grant_claims(case.req, permission='status')
            assert claims == case.claims and 'attachment.stage' in claims['permissions']
            removed = case.spool.discard_attempt(claims=claims, task_id=dispatch.task_id,
                execution_generation=dispatch.execution_generation,
                authorize_write=attachments._write_guard(target.adapter, case.req,
                    claims, 'status', dispatch))
            assert removed == 1 and all(not Path(item['path']).exists() for item in items)
            observed.append(failure)
            return items
        monkeypatch.setattr(case.spool, 'materialize', dispose)
    elif failure == 'spool-init':
        def initialize():
            observed.append(failure)
            # Canonical access no longer creates directories. Its existing-only
            # reader must still normalize a real inaccessible spool path.
            return attachments.RoomAttachmentSpool(case.spool.db_path,
                root=blocked / 'spool', _existing_only=True)
        monkeypatch.setattr(attachments, '_request_spool', initialize)
    elif failure in {'image-temp-directory', 'image-temp-write'}:
        original = bridge.tempfile.TemporaryDirectory
        @contextmanager
        def temporary(*args, **kwargs):
            observed.append(failure)
            if failure == 'image-temp-directory':
                kwargs['dir'] = blocked
            with original(*args, **kwargs) as directory:
                if failure == 'image-temp-write':
                    (Path(directory) / 'pixel.png').mkdir()
                yield directory
        monkeypatch.setattr(bridge.tempfile, 'TemporaryDirectory', temporary)
    elif failure in {'native-capture', 'document-copy'}:
        original = bridge.tempfile.mkstemp
        prefix = '.capture-' if failure == 'native-capture' else '.document-'
        def mkstemp(*args, **kwargs):
            if kwargs.get('prefix') == prefix:
                observed.append(failure)
                kwargs['dir'] = blocked
            return original(*args, **kwargs)
        monkeypatch.setattr(bridge.tempfile, 'mkstemp', mkstemp)
    elif failure == 'document-directory':
        original = preparation._directory
        def directory(path):
            observed.append(failure)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(b'test-owned non-directory')
            return original(path)
        monkeypatch.setattr(preparation, '_directory', directory)
    elif failure == 'document-sync':
        original = preparation._sync_directory
        def sync(path):
            observed.append(failure)
            # Real directory open fails; do not modify another copy or lower API.
            return original(blocked / path.name)
        monkeypatch.setattr(preparation, '_sync_directory', sync)
    elif failure == 'native-structured-control':
        original = bridge.capture_native_media
        def capture(paths):
            observed.append(failure)
            return original([blocked.parent])  # Real _open_regular refuses a directory.
        monkeypatch.setattr(bridge, 'capture_native_media', capture)
    elif failure == 'document-busy-control':
        target.db._conn.execute("""CREATE TRIGGER candidate_busy AFTER INSERT ON input_custody_copies
            WHEN NEW.name LIKE 'candidate-%' BEGIN
            UPDATE input_custody_copies SET state='sealed' WHERE copy_id=NEW.copy_id; END""")
    else:
        assert failure == 'document-finish-sql-control'
        target.db._conn.execute("""CREATE TRIGGER candidate_finish BEFORE UPDATE OF payload_digest
            ON input_custody_preparations BEGIN
            SELECT RAISE(ABORT, 'private final preparation error'); END""")


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [
    'dispose-after-materialize', 'spool-init', 'image-temp-directory', 'image-temp-write',
    'native-capture', 'document-directory', 'document-copy', 'document-sync',
    'native-structured-control', 'document-busy-control', 'document-finish-sql-control',
])
async def test_failed_preparation_forgets_only_unaccepted_run(files_target, monkeypatch, failure):
    target = files_target
    case = await http_admitted_files(target, monkeypatch)
    assert set(case.claims['permissions']) == {'approve', 'dispatch', 'status', 'stop', 'attachment.stage'}
    accepted = files_state(target, case.spool)
    candidate = await stage_candidate(target, case, mixed=failure.startswith(('image-', 'native-')))
    before = files_state(target, case.spool)
    admission_rows = rows(target.db._conn, 'session_admissions')
    receipt_rows = rows(target.adapter._run_idempotency_store._conn, 'run_idempotency')
    live = run_state(target.adapter)
    assert len(admission_rows) == len(receipt_rows) == 1  # The independent control exists.
    reserved = []
    reserve = target.adapter._run_idempotency_store.reserve
    def record_reservation(*args, **kwargs):
        result = reserve(*args, **kwargs)
        if result[0] == 'created':
            reserved.append(args[3])
        return result
    monkeypatch.setattr(target.adapter._run_idempotency_store, 'reserve', record_reservation)
    observed = []
    install_failure(monkeypatch, target, case, candidate, failure, observed)
    outcomes = []
    for _ in range(2):
        # Record both attempts even on unfixed code so RED proves the queued 202
        # replay, not only the exception. Never treat that exception as refusal.
        try:
            response = await target.adapter._handle_runs(run_request(case.issued['grant'], candidate))
        except (OSError, ValueError) as exc:
            outcomes.append({'escaped': type(exc).__name__})
        else:
            outcomes.append({'status': response.status, 'body': json.loads(response.text)})
        await asyncio.sleep(0)
    after = files_state(target, case.spool)
    assert rows(target.db._conn, 'session_admissions') == admission_rows
    assert after['custody'] == accepted['custody']
    assert after['accepted_bytes'] == accepted['accepted_bytes']
    assert all(row in after['copies'] for row in accepted['copies'])
    assert after['reservations'] == before['reservations']
    assert len(case.launched) == 1
    assert rows(target.db._conn, 'worker_executions') == set()
    # Even while preparation remains unavailable, accepted replay bypasses it.
    replay = await asyncio.to_thread(case.client.dispatch,
        dispatch=case.value.as_mapping(), grant=case.issued['grant'])
    assert replay == case.accepted | {'replayed': True}
    for table, control_rows in accepted['staging'].items():
        assert all(row in after['staging'][table] for row in control_rows)
    assert all(after['bytes'][key] == value for key, value in accepted['bytes'].items())
    if failure == 'dispose-after-materialize':
        assert observed == [failure]
        assert after['bytes'] == accepted['bytes']
        assert after['staging']['roomlink_attachment_batches'] == accepted['staging']['roomlink_attachment_batches']
    elif not failure.endswith('-control'):
        assert observed
        assert after['staging'] == before['staging'] and after['bytes'] == before['bytes']
    code = {'native-structured-control': 'invalid_params',
            'document-busy-control': 'input_preparation_busy'}.get(failure, 'storage_unavailable')
    expected_status = 503 if failure == 'document-finish-sql-control' else 409
    assert all(outcome.get('status') == expected_status for outcome in outcomes), outcomes
    assert all(outcome['body']['error']['code'] == code for outcome in outcomes), outcomes
    assert all(outcome['body']['error']['message'] == code for outcome in outcomes), outcomes
    assert all('not_admitted' not in outcome['body'] for outcome in outcomes)
    assert rows(target.adapter._run_idempotency_store._conn, 'run_idempotency') == receipt_rows
    assert run_state(target.adapter) == live
    assert len(set(reserved)) == 2
    assert all(run_id not in getattr(target.adapter, name) for run_id in reserved for name in _RUN_STATE)
    serialized = json.dumps(outcomes)
    assert str(target.home) not in serialized and case.issued['grant'] not in serialized
    assert all(hashlib.sha256(data).hexdigest() not in serialized for data in case.raw)
