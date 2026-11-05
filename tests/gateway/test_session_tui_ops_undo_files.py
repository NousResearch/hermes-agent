"""``/undo``'s file half on the shared owner (``rollback.undo_files``): after the rewind, the files
the undone turns wrote go back to their pre-turn state, under the session's checkpoint policy."""
import asyncio

import pytest

from tests.gateway.fixtures.local_recovery_probe import rpc, websocket
from tests.gateway.test_session_busy_controls import owner, settled
from tests.gateway.test_session_tui_ops import _git_rpc


def _turn_write(home, cwd, path, text, turn_ts, sessions):
    """What the agent loop does for a turn: begin_turn(user row ts), checkpoint, write, record."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.checkpoint_manager import CheckpointManager
    token = set_hermes_home_override(str(home))
    try:
        mgr = CheckpointManager(enabled=True)
        mgr.begin_turn(turn_ts, sessions)
        mgr.ensure_checkpoint(str(cwd), 'before write_file')
        path.write_text(text)
        mgr.record_agent_write(str(path))
    finally:
        reset_hermes_home_override(token)


def _last_user_row(home, sid):
    from hermes_state import SessionDB
    db = SessionDB(db_path=home / 'state.db')
    try:
        row = db._read_one("SELECT id, timestamp, session_id FROM messages WHERE role='user' AND active=1 "
                           "ORDER BY id DESC LIMIT 1")
        return row['id'], row['timestamp'], row['session_id']
    finally:
        db.close()


@pytest.mark.platforms("linux")
def test_undo_files_reverts_the_rewound_turns_writes_and_only_with_checkpoints_on(tmp_path):
    with owner(tmp_path, {'checkpoints': {'enabled': False}}) as (home, _peer, desc):
        work = tmp_path / 'work'
        work.mkdir()
        (work / 'a.txt').write_text('original\n')

        async def probe():
            async with websocket(home, desc) as ws:
                plain = (await rpc(ws, 'session.create', request_id='plain', source='tui', toolsets=[],
                                   cwd=str(work)))['result']['session_id']
                sid = (await rpc(ws, 'session.create', request_id='undo', source='tui', toolsets=[],
                                 cwd=str(work), checkpoints=True))['result']['session_id']
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='first', text='EDIT_IT')
                await settled(home)
                row, ts, physical = await asyncio.to_thread(_last_user_row, home, sid)
                await asyncio.to_thread(_turn_write, home, work, work / 'a.txt', 'agent\n', ts, physical)
                await asyncio.to_thread(_turn_write, home, work, work / 'new.txt', 'made\n', ts, physical)
                # Checkpoints off for this session: nothing is touched.
                off = await _git_rpc(ws, 'rollback.undo_files', session_id=plain, target_message_id=row)
                assert off['result']['enabled'] is False and (work / 'a.txt').read_text() == 'agent\n', off
                done = (await _git_rpc(ws, 'rollback.undo_files', session_id=sid, target_message_id=row))['result']
                assert sorted(p.rsplit('/', 1)[-1] for p in done['reverted']) == ['a.txt', 'new.txt'], done
                assert (work / 'a.txt').read_text() == 'original\n' and not (work / 'new.txt').exists()
                assert done['lines'] and 'a.txt' in done['lines'][0], done
                assert (await rpc(ws, 'rollback.undo_files', session_id=sid, target_message_id=0)
                        )['error']['message'] == 'invalid_params'
        asyncio.run(probe())
