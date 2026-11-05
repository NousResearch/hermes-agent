"""The shared owner's /undo pointer is bounded by the rewind's own row: a turn started since cannot move it."""
import asyncio

import pytest

from tests.gateway.test_session_busy_controls import owner, settled
from tests.gateway.test_session_tui_ops import _checkpoint, _git_rpc
from tests.gateway.fixtures.local_recovery_probe import rpc, websocket


@pytest.mark.platforms("linux")
def test_undo_hint_keeps_the_rewinds_boundary_after_a_new_turn(tmp_path):
    with owner(tmp_path) as (home, _peer, desc):
        work = tmp_path / 'work'
        work.mkdir()
        (work / 'a.txt').write_text('original\n')

        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='undo-hint', source='tui', toolsets=[],
                                 cwd=str(work), checkpoints=True))['result']['session_id']
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='kept', text='KEPT')
                await settled(home)
                await asyncio.sleep(1.1)  # health: allow HX013 -- checkpoint stamps are whole seconds: the next second IS the case
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='edit', text='EDIT')
                await settled(home)
                await asyncio.sleep(1.1)  # health: allow HX013 -- whole-second stamps: the checkpoint follows the turn
                await asyncio.to_thread(_checkpoint, home, work, 'before write_file')
                (work / 'a.txt').write_text('edited\n')
                snapshot = (await rpc(ws, 'session.resume', session_id=sid))['result']
                row = next(m for m in reversed(snapshot['messages']) if m['role'] == 'user')
                receipt = await rpc(ws, 'session.mutate', session_id=sid, request_id='undo',
                                    expected_revision=snapshot['revision'],
                                    expected_generation=snapshot['execution_generation'], operation='rewind',
                                    payload={'target_message_id': row['row_id']})
                assert 'result' in receipt, receipt
                receipt = receipt['result']
                target = receipt['target_message']
                # Another viewer starts (and finishes) a turn before this client asks for the pointer.
                await asyncio.sleep(1.1)  # health: allow HX013 -- whole-second stamps: the new turn is a later second
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='other', text='OTHER')
                await settled(home)
                listed = (await _git_rpc(ws, 'rollback.list', session_id=sid, undo_since=target['timestamp'],
                                         undo_message_id=target['id']))['result']
                assert listed.get('undo_hint', {}).get('number') == 1, listed
                assert listed['undo_hint']['files'] == ['a.txt'], listed
        asyncio.run(probe())
