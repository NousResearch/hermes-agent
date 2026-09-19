"""Readiness/NEW consent fences, preserving old input-only admission."""
import asyncio
import json

import pytest
from tests.gateway.test_canonical_peer_target_setup import target, invite, request  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target, staged_batch  # noqa: F401
from tests.gateway.test_peer_output_product import register_routes
from gateway.session_peer_output import initialize_peer_output, peer_output_permissions


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['routes', 'owner', 'epoch', 'process', 'policy', 'outbox', 'named'])
async def test_real_provider_denies_unready_owner_without_storage_writes(files_target, monkeypatch, change):
    from aiohttp import web
    from gateway import hosted_rooms
    from gateway.runtime_ownership import process_ownership
    from gateway.platforms.api_server_room_grants import _local_room_catalog
    t = files_target
    # The real request/native boundary owns this profile scope; do not read
    # the import-time gateway home of another parametrized fixture.
    monkeypatch.setattr("gateway.run._hermes_home", t.home)
    register_routes(t)
    initialize_peer_output(t.adapter)
    _, catalog = _local_room_catalog(t.adapter, 'default', hosted_rooms.local_authority_gateway_id())
    assert peer_output_permissions(t.adapter, profile='default', catalog=catalog, connection=t.db._conn)
    if change == 'routes':
        t.adapter._app = web.Application()
    elif change == 'owner':
        t.runner.session_authority = None
    elif change == 'epoch':
        t.authority.epoch += 1
    elif change == 'process':
        process_ownership.release(t.home)
    elif change == 'policy':
        (t.home / 'config.yaml').write_text('approvals:\n  mode: off\n')
    elif change == 'outbox':
        t.db._execute_write(lambda conn: conn.execute('DROP TABLE hosted_room_output_artifacts'))
    changes = t.db._conn.total_changes
    schema = list(t.db._conn.execute('SELECT name,sql FROM sqlite_master'))
    assert peer_output_permissions(t.adapter, profile='named' if change == 'named' else 'default', catalog=catalog, connection=t.db._conn) == ()
    assert t.db._conn.total_changes == changes
    assert list(t.db._conn.execute('SELECT name,sql FROM sqlite_master')) == schema


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['provider', 'grant', 'epoch', 'owner', 'policy'])
async def test_output_consent_rechecked_inside_new_after_input_preparation(files_target, monkeypatch, change):
    from gateway import session_api_turn, hosted_rooms
    from tests.gateway.test_canonical_peer_text_admission import run_request
    from gateway.platforms import api_server_runs
    t = files_target
    register_routes(t)
    initialize_peer_output(t.adapter)
    issued, dispatch, claims, req, spool, manifest, raw = await staged_batch(t)
    for item, data in zip(manifest, raw):
        spool.put(claims=claims, task_id=dispatch.task_id, execution_generation=dispatch.execution_generation,
                  attachment_id=item['attachment_id'], data=data)
    original = session_api_turn.admit_session_input
    seen = []
    def before(*args, **kwargs):
        payload = kwargs['payload']
        assert payload['api_turn_v1']['output_consent']['claims']['_token_sha256'] == claims['_token_sha256']
        assert len(payload['api_turn_v1']['settings']['room_input_media']['media']) == 2
        seen.append(payload)
        if change == 'provider':
            t.adapter._room_output_invitation_permissions = None
        elif change == 'grant':
            hosted_rooms.revoke_room_grant_id(t.db.db_path, claims=claims, expires_at=claims['status_expires_at'])
        elif change == 'epoch':
            t.db._execute_write(lambda conn: conn.execute('UPDATE runtime_epoch SET epoch=epoch+1'))
        elif change == 'owner':
            t.runner.session_authority = None
        else:
            (t.home / 'config.yaml').write_text('approvals:\n  mode: off\n')
        return original(*args, **kwargs)
    monkeypatch.setattr(session_api_turn, 'admit_session_input', before)
    async def forbidden(*args, **kwargs):
        pytest.fail('failed consent must not execute')
    monkeypatch.setattr(api_server_runs, '_execute_run', forbidden)
    response = await t.adapter._handle_runs(run_request(issued['grant'], dispatch))
    assert response.status in (409, 503), response.text
    assert len(seen) == 1
    assert t.db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0
    assert t.db._conn.execute('SELECT count(*) FROM input_custody_refs').fetchone()[0] == 0
    assert t.adapter._run_idempotency_store._conn.execute('SELECT count(*) FROM run_idempotency').fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['_room_output_consent', '_room_output_authorizer', '_room_artifact_publication', 'room_artifact_publication', 'output_consent', 'api_turn_v1'])
async def test_caller_cannot_supply_private_output_consent(files_target, field):
    response = await files_target.adapter._handle_runs(request({'input': 'hello', field: True}))
    assert response.status == 400
    assert files_target.db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 0


@pytest.mark.asyncio
async def test_old_input_invitation_still_runs_without_output_binding(files_target, monkeypatch):
    from tests.gateway.test_canonical_peer_text_admission import dispatch, run_request
    from gateway.platforms import api_server_runs
    from gateway.session_hosted_output import current_output_binding
    from gateway.session_results import execution_result
    from gateway.platforms.api_server_authority_runs import run_projection
    from tools.hosted_room_artifact import share_group_file
    t = files_target
    issued = await invite(t)  # real input-only consent, before Output initialization
    register_routes(t)
    initialize_peer_output(t.adapter)
    launched = []
    async def inert(adapter, launch, **kwargs):
        launched.append(launch.admission)
    monkeypatch.setattr(api_server_runs, '_execute_run', inert)
    monkeypatch.setattr(t.authority, '_schedule', lambda ref: None)
    async def handle(event):
        assert current_output_binding() is None
        assert json.loads(share_group_file('/not-an-output-file'))['ok'] is False
        execution_result.get().update(result={'final_response': 'ordinary input work', 'messages': []}, usage={})
        return 'ordinary input work'
    t.runner._handle_message = handle
    response = await t.adapter._handle_runs(run_request(issued['grant'], dispatch(issued)))
    assert response.status == 202, response.text
    await asyncio.sleep(0)
    authority, ref, row = launched[0]
    assert 'output_consent' not in row['payload']['api_turn_v1']
    await authority._drain(ref)
    projected = run_projection(t.adapter, json.loads(response.text)['run_id'])
    assert projected['status'] == 'completed' and projected['output'] == 'ordinary input work'
    assert 'artifacts' not in projected
