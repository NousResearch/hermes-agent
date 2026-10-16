"""The classic CLI's ``/diff session`` on the shared owner (``rollback.session_diff``): the session's
frozen checkpoint policy decides, so a ``--checkpoints`` launch on a profile whose config leaves
checkpoints off still diffs, and the whole body comes back as data (no chat-sized cut)."""
import asyncio

import pytest

from tests.gateway.fixtures.local_recovery_probe import rpc, websocket
from tests.gateway.test_session_busy_controls import owner
from tests.gateway.test_session_tui_ops import _checkpoint, _git_rpc


@pytest.mark.platforms("linux")
def test_session_diff_follows_the_sessions_checkpoint_policy_and_keeps_the_whole_body(tmp_path):
    with owner(tmp_path) as (home, _peer, desc):
        work = tmp_path / 'work'
        work.mkdir()
        (work / 'big.txt').write_text(''.join(f'line {i}\n' for i in range(450)))

        async def probe():
            async with websocket(home, desc) as ws:
                plain = (await rpc(ws, 'session.create', request_id='plain', source='cli', toolsets=[],
                                   cwd=str(work)))['result']['session_id']
                sid = (await rpc(ws, 'session.create', request_id='launch', source='cli', toolsets=[],
                                 cwd=str(work), checkpoints=True))['result']['session_id']
                await asyncio.to_thread(_checkpoint, home, work, 'before write_file')
                (work / 'big.txt').write_text(''.join(f'edited {i}\n' for i in range(450)))
                # The profile's config leaves checkpoints off: only the --checkpoints session diffs.
                off = await _git_rpc(ws, 'rollback.session_diff', session_id=plain)
                assert off.get('result', {}).get('enabled') is False, off
                reply = await _git_rpc(ws, 'rollback.session_diff', session_id=sid)
                assert 'result' in reply, reply
                result = reply['result']
                assert result['enabled'] is True and result['success'] is True, result
                assert 'big.txt' in result['stat'] and '+edited 449' in result['diff'], result['stat']
                assert result['diff_lines'] == len(result['diff'].splitlines()) > 900
                for params, reason in [({'session_id': sid, 'hash': '1'}, 'invalid_params'),
                                       ({'session_id': sid, 'profile': 'foreign'}, 'profile_mismatch')]:
                    assert (await rpc(ws, 'rollback.session_diff', **params))['error']['message'] == reason
        asyncio.run(probe())
