"""The classic CLI's ``/diff session`` on the shared owner (``rollback.session_diff``): the session's
frozen checkpoint policy decides, so a ``--checkpoints`` launch on a profile whose config leaves
checkpoints off still diffs, and the whole body comes back as data (no chat-sized cut)."""
import asyncio
import json

import pytest

from gateway.session_tui_ops import _fit_frame
from tests.gateway.fixtures.local_recovery_probe import rpc, websocket
from tests.gateway.test_session_busy_controls import owner
from tests.gateway.test_session_tui_ops import _checkpoint, _git_rpc

# The classic CLI's frame cap (hermes_cli/gateway_client.py ``connect(..., max_size=...)``).
CLIENT_MAX_FRAME = 8 * 1024 * 1024


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


@pytest.mark.platforms("linux")
def test_an_escape_heavy_session_diff_fits_the_clients_frame_and_names_its_cut(tmp_path):
    """CLI-26: 2.4 Mi control characters are 14 MB JSON-escaped. The reply is cut to fit the classic
    CLI's 8 MiB frame and says how many lines the whole diff has, instead of dropping the socket."""
    with owner(tmp_path) as (home, _peer, desc):
        work = tmp_path / 'work'
        work.mkdir()
        (work / 'big.txt').write_text('original\n')

        async def probe():
            async with websocket(home, desc, max_size=CLIENT_MAX_FRAME) as ws:
                sid = (await rpc(ws, 'session.create', request_id='launch', source='cli', toolsets=[],
                                 cwd=str(work), checkpoints=True))['result']['session_id']
                await asyncio.to_thread(_checkpoint, home, work, 'before write_file')
                (work / 'big.txt').write_text(('\x01' * 1000 + '\n') * 2400)
                result = (await _git_rpc(ws, 'rollback.session_diff', session_id=sid))['result']
                assert result['success'] is True and 'big.txt' in result['stat'], result['stat']
                shown = result['diff'].splitlines()
                assert 400 < len(shown) < result['diff_lines'] and result['diff_lines'] > 2400
                assert len(json.dumps(result, ensure_ascii=False).encode()) <= CLIENT_MAX_FRAME // 2
                # The connection survives for the next call.
                assert (await _git_rpc(ws, 'rollback.list', session_id=sid))['result']['enabled'] is True
        asyncio.run(probe())


def test_fit_frame_never_turns_a_cut_over_budget_line_into_a_fitting_one():
    """The prefix slice keeps one char past the budget, so a line it cuts still breaks the loop."""
    assert _fit_frame('abcdefgh', 5) == ''
    assert _fit_frame('ab\ncdefgh', 5) == 'ab'


def test_fit_frame_cuts_only_at_a_newline():
    """A carriage return, form feed or U+2028 inside a diff line is not a line end: the cut never lands on one."""
    for sep in ('\r', '\x0c', '\u2028'):
        assert _fit_frame(f'+a\n+b{sep}cdefghij\n', 9) == '+a', repr(sep)
