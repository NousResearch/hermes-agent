import asyncio
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from gateway import hosted_rooms
from gateway import session_group_files_revoke as cleanup
from gateway.hosted_room_peer import attachment_manifest_digest, decode_room_grant
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from gateway.platforms import api_server_room_attachments as attachments_api
from tests.gateway.platforms.test_room_attachment_staging_provenance import (
    case_for, manifest_request, put_request, renewed_token,
)
from tests.gateway.test_canonical_peer_files_target import files_target, files_state
from tests.gateway.test_canonical_peer_target_setup import target, invite, invitation

__all__ = ['target', 'files_target']

pytestmark = pytest.mark.linux_only


async def stage(case, *, complete=True):
    await manifest_request(case)
    await put_request(case)
    if complete:
        await put_request(case, index=1)


async def revoke(target, case, *, exact=True):
    return await target.connection.dispatch(dict(id=2,
        method='groups.peer.revoke_exact' if exact else 'groups.peer.revoke',
        params={'grant': case.issued['grant']}))


def batch(case):
    with sqlite3.connect(case.spool.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute('SELECT * FROM roomlink_attachment_batches WHERE task_id=?',
                           (case.value.task_id,)).fetchone()
        return dict(row) if row else None


def file_rows(case):
    with sqlite3.connect(case.spool.db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(
            'SELECT * FROM roomlink_attachment_files WHERE batch_key=? ORDER BY attachment_id',
            (attachments_api._batch_key(case.value),))]


def assert_reclaimed(case, before_batch, before_files):
    # Only availability changes: identities, provenance and timestamps survive.
    assert batch(case) == before_batch | {'complete': 0}
    assert file_rows(case) == [row | {'stored': 0} for row in before_files]
    assert not any(path.exists() for path in paths(case))
    with sqlite3.connect(case.spool.db_path) as conn:
        assert conn.execute('SELECT COALESCE(SUM(size), 0) FROM roomlink_attachment_files '
                            'WHERE batch_key=? AND stored=1', (before_batch['batch_key'],)).fetchone()[0] == 0


def paths(case):
    key = attachments_api._batch_key(case.value)
    return [case.spool._file_path(key, item['attachment_id']) for item in case.manifest]


def denied(target, case):
    assert all(hosted_rooms.room_grant_is_revoked(target.home / name, claims=case.claims)
               for name in ('shared-state.db', 'state.db'))


@pytest.mark.asyncio
@pytest.mark.parametrize('exact', [False, True])
@pytest.mark.parametrize('complete', [False, True])
async def test_native_reclaims_only_denied_origin_preserving_new_token_and_fences(files_target, monkeypatch, exact, complete):
    case = await case_for(files_target, monkeypatch)
    await stage(case, complete=complete)
    before_batch, before_files = batch(case), file_rows(case)
    replacement = renewed_token(files_target, case)
    newer = replace(case.value, task_id='replacement-task')
    await manifest_request(case, grant=replacement, value=newer)
    other = type(case)(**(vars(case) | {'value': newer, 'issued': case.issued | {'grant': replacement}}))
    await put_request(other, grant=replacement)
    before = files_state(files_target, case.spool)
    files_target.runner.adapters.clear()
    files_target.runner._draining = True
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError, match='runtime_draining'):
        files_target.authority._require_admission_open()
    monkeypatch.setattr(hosted_rooms.time, 'time', lambda: case.claims['issued_at'] + 0.5)
    result = await revoke(files_target, case, exact=exact)
    assert result.get('result') == {'revoked': True, 'exact': exact}, result
    denied(files_target, case)
    assert_reclaimed(case, before_batch, before_files)
    assert batch(other) is not None and paths(other)[0].read_bytes() == other.raw[0]
    after = files_state(files_target, case.spool)
    assert after['staging']['roomlink_attachment_attempt_fences'] == before['staging']['roomlink_attachment_attempt_fences']
    assert case.spool.prepare(case.value, case.manifest)['complete'] is False
    higher = replace(case.value, execution_generation=2)
    case.spool.prepare(higher, case.manifest)
    case.spool.put(claims=case.claims, task_id=higher.task_id, execution_generation=2,
                   attachment_id=case.manifest[0]['attachment_id'], data=case.raw[0])
    assert case.spool._file_path(attachments_api._batch_key(higher), case.manifest[0]['attachment_id']).read_bytes() == case.raw[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('exact', [False, True])
@pytest.mark.parametrize('first_touch', ['prepare', 'put'])
async def test_reclaimed_attempt_resumes_but_explicit_disposal_still_retires(files_target, monkeypatch, exact, first_touch):
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    before_batch, before_files = batch(case), file_rows(case)
    with sqlite3.connect(case.spool.db_path) as conn:
        fences = conn.execute('SELECT * FROM roomlink_attachment_attempt_fences').fetchall()
    counts = []
    collect = cleanup._collect
    def collected(*args):
        result = collect(*args)
        counts.append(result)
        return result
    monkeypatch.setattr(cleanup, '_collect', collected)
    cutoff = case.claims['issued_at'] + 0.5
    monkeypatch.setattr(hosted_rooms.time, 'time', lambda: cutoff)
    assert (await revoke(files_target, case, exact=exact))['result']['revoked']
    assert_reclaimed(case, before_batch, before_files)
    denied(files_target, case)
    # Same persisted cutoff on repeat; empty metadata is not newly reclaimed work.
    assert (await revoke(files_target, case, exact=exact))['result']['revoked']
    assert counts == [1, 0]
    assert_reclaimed(case, before_batch, before_files)
    if not exact:
        for name in ('state.db', 'shared-state.db'):
            with sqlite3.connect(files_target.home / name) as conn:
                assert conn.execute('SELECT revoked_before FROM hosted_room_revoked_grants').fetchone()[0] == cutoff
    monkeypatch.setattr(hosted_rooms.time, 'time', lambda: case.claims['issued_at'] + 1)
    replacement = (renewed_token(files_target, case) if exact else
                   (await invite(files_target, invitation() | {'request_id': 'post-scope-revoke'}))['grant'])
    claims = decode_room_grant(files_target.adapter._room_grant_secret(), replacement, permission='attachment.stage')
    assert claims['issued_at'] > cutoff
    assert all(hosted_rooms.peer_room_grant_is_current(files_target.home / name, claims=claims)
               for name in ('state.db', 'shared-state.db'))
    for operation in (manifest_request, put_request):
        with pytest.raises(PeerRunsHTTPError) as old:
            await operation(case)
        assert old.value.status_code in {401, 403}
    changed = [case.manifest[0] | {'name': 'different.txt'}, case.manifest[1]]
    with pytest.raises(PeerRunsHTTPError) as conflict:
        await manifest_request(case, grant=replacement, manifest=changed,
            value=replace(case.value, attachment_manifest_digest=attachment_manifest_digest(changed)))
    assert conflict.value.status_code == 409
    with pytest.raises(PeerRunsHTTPError) as conflict:
        await put_request(case, grant=replacement, data=b'changed content')
    assert conflict.value.status_code == 409
    with pytest.raises(PeerRunsHTTPError) as conflict:
        await manifest_request(case, grant=replacement, value=replace(case.value, member_id='other-member'))
    assert conflict.value.status_code in {400, 401, 403}
    assert_reclaimed(case, before_batch, before_files)
    if first_touch == 'prepare':
        assert (await manifest_request(case, grant=replacement)) == {
            'batch_key': before_batch['batch_key'], 'manifest_digest': before_batch['manifest_digest'],
            'complete': False, 'idempotent': True, 'object': 'hermes.room_attachment_batch'}
    else:
        assert (await put_request(case, grant=replacement))['complete'] is False
    assert batch(case)['staging_origin_json'] == before_batch['staging_origin_json']
    assert batch(case)['staging_origin_ambiguous'] == 1
    await put_request(case, grant=replacement)
    assert (await put_request(case, grant=replacement, index=1))['complete'] is True
    assert [p.read_bytes() for p in paths(case)] == case.raw
    # Repeated A cleanup cannot collect the replacement's now mixed-origin bytes.
    assert (await revoke(files_target, case, exact=exact))['result']['revoked']
    assert counts == [1, 0, 0]
    assert [p.read_bytes() for p in paths(case)] == case.raw
    # A later scope revoke can correctly deny B too; use a new post-cutoff token
    # for disposal without changing the retained attempt's identity.
    if not exact:
        monkeypatch.setattr(hosted_rooms.time, 'time', lambda: case.claims['issued_at'] + 2)
        replacement = (await invite(files_target, invitation() | {'request_id': 'post-repeat-scope'}))['grant']
    with sqlite3.connect(case.spool.db_path) as conn:
        assert conn.execute('SELECT * FROM roomlink_attachment_attempt_fences').fetchall() == fences
        assert conn.execute('SELECT SUM(size) FROM roomlink_attachment_files WHERE stored=1').fetchone()[0] == sum(map(len, case.raw))
    result = await asyncio.to_thread(case.client.discard_attachments, task_id=case.value.task_id,
                                    execution_generation=1, grant=replacement)
    assert result['removed'] == 1
    assert batch(case) is None and file_rows(case) == []
    assert not any(p.exists() for p in paths(case))
    with sqlite3.connect(case.spool.db_path) as conn:
        assert conn.execute('SELECT * FROM roomlink_attachment_attempt_fences').fetchall() == fences
    with pytest.raises(PeerRunsHTTPError) as retired:
        await manifest_request(case, grant=replacement)
    assert retired.value.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize('history', ['null', 'ambiguous', 'foreign', 'invalid', 'scope', 'duplicate', 'row-generation'])
async def test_uncertain_history_never_authorizes_reclamation(files_target, monkeypatch, history):
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    row = batch(case)
    origin = json.loads(row['staging_origin_json'])
    encoded, ambiguous = row['staging_origin_json'], 0
    if history == 'null':
        encoded = None
    elif history == 'ambiguous':
        ambiguous = 1
    elif history == 'foreign':
        encoded = json.dumps(origin | {'receiver_home': str(files_target.home / 'foreign')})
    elif history == 'scope':
        encoded = json.dumps(origin | {'member_id': 'somebody-else'})
    elif history == 'duplicate':
        encoded = encoded[:-1] + ',"version":1}'
    elif history == 'invalid':
        encoded = json.dumps(origin | {'issued_at': True})
    with sqlite3.connect(case.spool.db_path) as conn:
        if history == 'row-generation':
            conn.execute('UPDATE roomlink_attachment_batches SET execution_generation=2')
        conn.execute('UPDATE roomlink_attachment_batches SET staging_origin_json=?, staging_origin_ambiguous=?',
                     (encoded, ambiguous))
    before = batch(case)
    assert (await revoke(files_target, case))['result']['revoked']
    denied(files_target, case)
    assert batch(case) == before
    assert [p.read_bytes() for p in paths(case)] == case.raw


@pytest.mark.asyncio
@pytest.mark.parametrize('failed_store', ['shared-state.db', 'state.db'])
@pytest.mark.parametrize('exact', [False, True])
async def test_failed_deny_preserves_sibling_and_all_staging(files_target, monkeypatch, failed_store, exact):
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    with sqlite3.connect(files_target.home / failed_store) as conn:
        table = 'hosted_room_revoked_grant_tokens' if exact else 'hosted_room_revoked_grants'
        conn.execute(f"CREATE TRIGGER deny_revoke BEFORE INSERT ON {table} "
                     "BEGIN SELECT RAISE(ABORT, 'fixture store failure'); END")
    result = await revoke(files_target, case, exact=exact)
    assert 'error' in result, result
    for name in ('shared-state.db', 'state.db'):
        assert hosted_rooms.room_grant_is_revoked(files_target.home / name, claims=case.claims) == (name != failed_store)
    assert batch(case) is not None
    assert [p.read_bytes() for p in paths(case)] == case.raw


@pytest.mark.asyncio
@pytest.mark.parametrize('exact', [False, True])
async def test_accepted_native_and_document_custody_is_not_staging(files_target, monkeypatch, exact):
    from gateway.platforms import api_server_runs
    case = await case_for(files_target, monkeypatch)
    await stage(case)
    launched = []

    async def inert(adapter, launch, **kwargs):
        launched.append(launch.admission)

    monkeypatch.setattr(api_server_runs, '_execute_run', inert)
    accepted = await asyncio.to_thread(case.client.dispatch, dispatch=case.value.as_mapping(), grant=case.issued['grant'])
    await asyncio.sleep(0)
    assert len(launched) == 1
    from gateway.hosted_room_artifacts import RoomArtifactOutbox, RoomArtifactScope
    outbox = RoomArtifactOutbox(files_target.home / 'state.db', root=files_target.home / 'hosted-room-artifact-outbox')
    scope = RoomArtifactScope.from_mapping({key: getattr(case.value, key) for key in (
        'room_id', 'task_id', 'execution_generation', 'member_id', 'target_profile',
        'home_install_id', 'target_install_id', 'authority_gateway_id', 'authority_epoch')})
    output = outbox.put_bytes(scope=scope, data=b'fixture retained output', source_name='answer.txt')
    output_rows = [tuple(row) for row in files_target.db._conn.execute('SELECT * FROM hosted_room_output_artifacts')]
    before_batch, before_files = batch(case), file_rows(case)
    before = files_state(files_target, case.spool)
    admissions = [tuple(row) for row in files_target.db._conn.execute('SELECT * FROM session_admissions')]
    monkeypatch.setattr(hosted_rooms.time, 'time', lambda: case.claims['issued_at'] + 0.5)
    assert (await revoke(files_target, case, exact=exact))['result']['revoked']
    denied(files_target, case)
    monkeypatch.setattr(hosted_rooms.time, 'time', lambda: case.claims['issued_at'] + 1)
    replacement = (renewed_token(files_target, case) if exact else
                   (await invite(files_target, invitation() | {'request_id': 'accepted-scope-replacement'}))['grant'])
    assert_reclaimed(case, before_batch, before_files)
    after = files_state(files_target, case.spool)
    assert after['accepted_bytes'] == before['accepted_bytes'] and list(after['accepted_bytes'].values()) == case.raw
    assert after['custody'] == before['custody'] and after['copies'] == before['copies']
    assert outbox.read(scope, output['artifact_id']) == (output, b'fixture retained output')
    assert [tuple(row) for row in files_target.db._conn.execute('SELECT * FROM hosted_room_output_artifacts')] == output_rows
    assert (await manifest_request(case, grant=replacement))['complete'] is False
    await put_request(case, grant=replacement)
    await put_request(case, grant=replacement, index=1)
    replay = await asyncio.to_thread(case.client.dispatch, dispatch=case.value.as_mapping(), grant=replacement)
    assert replay['run_id'] == accepted['run_id'] and len(launched) == 1
    assert [tuple(row) for row in files_target.db._conn.execute('SELECT * FROM session_admissions')] == admissions
    assert all(Path(path).read_bytes() == data for path, data in after['accepted_bytes'].items())
