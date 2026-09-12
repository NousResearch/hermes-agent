"""Current document/PNG custody survives actual root-local output and exact ACK."""
import asyncio
import base64
import json

import pytest

from gateway import hosted_room_driver as tasks
from gateway.hosted_room_artifacts import RoomArtifactOutbox, RoomArtifactScope
from gateway.hosted_room_input_reclamation import initialize_working_copies
from gateway.runtime_ownership import process_ownership
from hermes_state_runtime import get_session_admission
from tests.gateway.test_canonical_hosted_outputs import owner, execute_group_turn
from tools.hosted_room_artifact import share_group_file


@pytest.mark.asyncio
async def test_current_document_and_png_admission_produces_output_without_custody_regression(tmp_path, monkeypatch):
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        process_ownership.reserve([tmp_path])
        try:
            initialize_working_copies(authority.db, epoch=authority.epoch)
            png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII=')
            raw = [('source.txt', 'file', 'text/plain', b'private current input'),
                   ('pixel.png', 'image', 'image/png', png)]
            uploaded = [service.attachments.put(room_id='room', upload_id=f'input-{n}', name=name,
                kind=kind, mime=mime, data=data) for n, (name, kind, mime, data) in enumerate(raw)]
            manifest = [{k: item[k] for k in ('attachment_id', 'kind', 'name', 'size', 'mime')}
                        for item in uploaded]
            output = tmp_path / 'cache' / 'answer.txt'
            output.parent.mkdir(exist_ok=True)
            output.write_bytes(b'bounded result from current input')
            seen = []
            async def handle(event):
                from gateway.session_hosted_output import current_output_binding
                from gateway.session_finite import finite_turn_required
                from gateway.session_surface import _surface_turn
                binding = current_output_binding()
                assert binding is not None and binding.active
                assert finite_turn_required() is not None
                assert _surface_turn.get() is None
                row = get_session_admission(authority.db, admission_id=event.message_id)
                assert row['status'] == 'started'
                refs = [dict(r) for r in authority.db._conn.execute('SELECT * FROM input_custody_refs')]
                assert len(refs) == 1 and refs[0]['admission_id'] == row['admission_id']
                assert refs[0]['payload_digest'] == authority.db._conn.execute(
                    'SELECT payload_digest FROM session_admissions WHERE admission_id=?',
                    (row['admission_id'],)).fetchone()[0]
                from gateway.hosted_room_input_reclamation import copy_path
                copies = [copy_path(authority.db, dict(r)) for r in authority.db._conn.execute("SELECT * FROM input_custody_copies WHERE namespace='v3'")]
                assert len(copies) == 1 and copies[0].read_bytes() == raw[0][3]
                assert row['payload']['attachments_v1']['media']
                seen.append((row['admission_id'], refs))
                shared = json.loads(await asyncio.to_thread(share_group_file, str(output)))
                assert shared['ok'], shared
                return 'Shared the output.'
            runner._handle_message = handle
            from gateway.session_surface import _surface_turn, surface_turn_scope
            with surface_turn_scope({'surface': 'hud'}):
                _, _, receipt, task, binding = await execute_group_turn(
                    authority, service, input_manifest=manifest)
                assert _surface_turn.get() == {'surface': 'hud'}
            saved = tasks.get_task(service.db_path, task['identity'])
            assert saved['status'] == 'settled' and len(seen) == 1
            assert saved['result']['artifacts']['items'][0]['name'] == output.name
            scope = RoomArtifactScope.from_mapping(saved['result']['artifact_scope'])
            assert RoomArtifactOutbox(service.db_path).retirement_complete(scope)
            assert [dict(r) for r in authority.db._conn.execute('SELECT * FROM input_custody_refs')] == seen[0][1]
            service.prepare_room(binding)
            assert len(seen) == 1
            assert get_session_admission(authority.db, admission_id=receipt['admission_id'])['status'] == 'terminal'
            message, = [e for e in service._events('room') if e['kind'] == 'message.member']
            attachment = message['payload']['attachments'][0]
            assert service.attachments.read_viewer(room_id='room', event_id=message['event_id'],
                attachment_id=attachment['attachment_id'], authority_gateway_id=scope.authority_gateway_id,
                authority_epoch=scope.authority_epoch).data == output.read_bytes()
        finally:
            process_ownership.release(tmp_path)
