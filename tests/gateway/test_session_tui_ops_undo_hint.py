"""The shared owner's /undo pointer is bounded by the rewind's own row: a turn started since cannot move it."""
import asyncio

import pytest

from tests.gateway.test_session_busy_controls import owner, settled
from tests.gateway.test_session_tui_ops import _git_rpc
from tests.gateway.fixtures.local_recovery_probe import rpc, websocket


def _turn_checkpoint(home, cwd, reason, stamp):
    """A checkpoint in the owner profile's shared store, made by the turn whose user row is stamped ``stamp``
    (what the agent takes before that turn's first write)."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.checkpoint_manager import CheckpointManager
    mgr = CheckpointManager(enabled=True)
    mgr.turn_key = repr(stamp)
    token = set_hermes_home_override(str(home))
    try:
        assert mgr.ensure_checkpoint(str(cwd), reason)
    finally:
        reset_hermes_home_override(token)


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
                snapshot = (await rpc(ws, 'session.resume', session_id=sid))['result']
                row = next(m for m in reversed(snapshot['messages']) if m['role'] == 'user')
                await asyncio.to_thread(_turn_checkpoint, home, work, 'before write_file', row['timestamp'])
                (work / 'a.txt').write_text('edited\n')
                snapshot = (await rpc(ws, 'session.resume', session_id=sid))['result']
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
                hint = listed.get('undo_hint') or {}
                assert hint.get('files') == ['a.txt'], listed
                [row] = listed['checkpoints']
                assert hint.get('ref') and row['hash'].startswith(hint['ref']), listed
                restored = (await _git_rpc(ws, 'rollback.restore', session_id=sid, hash=hint['ref']))['result']
                assert restored['success'] is True and (work / 'a.txt').read_text() == 'original\n', restored
        asyncio.run(probe())
