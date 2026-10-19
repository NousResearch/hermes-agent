"""Ink's /save, /reload, /agents pause and /rollback on the shared owner (main's sidecar parity)."""
import asyncio
import json
import os
from pathlib import Path

import pytest

from tests.gateway.test_session_busy_controls import owner, settled
from tests.gateway.fixtures.local_recovery_probe import rpc, websocket


@pytest.mark.platforms("linux")
def test_session_save_exports_the_stored_transcript_under_the_profile_home(tmp_path):
    with owner(tmp_path) as (home, _peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='save', source='tui', toolsets=[]))['result']['session_id']
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='first', text='SAVE_ME')
                await settled(home)
                saved = await rpc(ws, 'session.save', session_id=sid)
                assert 'result' in saved, saved
                path = saved['result']['file']
                assert path.startswith(str(home / 'sessions' / 'saved' / 'hermes_conversation_')), path
                data = json.loads(await asyncio.to_thread(Path(path).read_text, encoding='utf-8'))
                contents = [m.get('content') for m in data['messages']]
                assert 'SAVE_ME' in contents and 'CONTROL_ACK' in contents, data
                for params, reason in [({'session_id': sid, 'format': 'md'}, 'invalid_params'),
                                       ({'session_id': sid, 'profile': 'foreign'}, 'profile_mismatch'),
                                       ({}, 'invalid_params')]:
                    assert (await rpc(ws, 'session.save', **params))['error']['message'] == reason
        asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_session_export_returns_the_rendered_transcript_for_the_client_to_write(tmp_path):
    """T10/T11: ``session.export`` hands the client the rendered ``/save`` transcript; the gateway
    writes nothing, so a remote Desktop and a local Ink save to the user's own disk."""
    with owner(tmp_path) as (home, _peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='exp', source='tui', toolsets=[]))['result']['session_id']
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='first',
                          text='EXPORT_ME sk-ant-api03-' + 'a' * 40)
                await settled(home)
                saved = home / 'sessions' / 'saved'
                before = sorted(saved.iterdir()) if saved.exists() else []
                as_json = (await rpc(ws, 'session.export', session_id=sid))['result']
                assert as_json['format'] == 'json' and as_json['filename'].endswith('.json'), as_json
                contents = [m.get('content') for m in json.loads(as_json['content'])['messages']]
                assert any('EXPORT_ME' in (c or '') for c in contents), contents
                md = (await rpc(ws, 'session.export', session_id=sid, format='markdown', redact=True))['result']
                assert md['format'] == 'md' and md['filename'].endswith('.md')
                assert 'EXPORT_ME' in md['content'] and 'a' * 40 not in md['content']
                html = (await rpc(ws, 'session.export', session_id=sid, format='html'))['result']
                assert html['content'].lstrip().lower().startswith('<!doctype html')
                assert (sorted(saved.iterdir()) if saved.exists() else []) == before  # nothing written
                for params, reason in [({'session_id': sid, 'format': 'pdf'}, 'invalid_params'),
                                       ({'session_id': sid, 'redact': 'yes'}, 'invalid_params'),
                                       ({'session_id': sid, 'profile': 'foreign'}, 'profile_mismatch'),
                                       ({}, 'invalid_params')]:
                    assert (await rpc(ws, 'session.export', **params))['error']['message'] == reason, params
        asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_reload_env_rereads_the_profile_dotenv_for_a_controlled_session(tmp_path):
    with owner(tmp_path) as (home, _peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='reload', source='tui', toolsets=[]))['result']['session_id']
                env = home / '.env'
                await asyncio.to_thread(env.write_text, 'TUI_RELOAD_PROBE=one\n')
                first = await rpc(ws, 'reload.env', session_id=sid)
                assert first.get('result', {}).get('updated', 0) >= 1, first
                assert (await rpc(ws, 'reload.env', session_id=sid))['result'] == {'updated': 0}
                # Same size, same inode, pinned mtime: only the bytes distinguish the versions. On a
                # kernel with coarse ctime a same-tick rewrite also kept ctime, and a stat-keyed .env
                # memo served the old value (CI run 39146191939); the memo now keys on content.
                pinned = env.stat()
                await asyncio.to_thread(env.write_text, 'TUI_RELOAD_PROBE=two\n')
                os.utime(env, ns=(pinned.st_atime_ns, pinned.st_mtime_ns))
                assert env.stat().st_size == pinned.st_size
                assert (await rpc(ws, 'reload.env', session_id=sid))['result'] == {'updated': 1}
                # A bare frame never rewrites the shared owner's environment.
                for params, reason in [({}, 'invalid_params'), ({'session_id': sid, 'profile': 'foreign'}, 'profile_mismatch'),
                                       ({'session_id': sid, 'force': True}, 'invalid_params')]:
                    assert (await rpc(ws, 'reload.env', **params))['error']['message'] == reason
        asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_delegation_pause_toggles_the_spawn_gate_and_status_reports_it(tmp_path):
    with owner(tmp_path) as (home, _peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                await rpc(ws, 'session.create', request_id='pause', source='tui', toolsets=[])
                paused = await rpc(ws, 'delegation.pause', paused=True)
                assert paused.get('result') == {'paused': True}, paused
                assert (await rpc(ws, 'delegation.status'))['result']['paused'] is True
                assert (await rpc(ws, 'delegation.pause', paused=False))['result'] == {'paused': False}
                assert (await rpc(ws, 'delegation.status'))['result']['paused'] is False
                for params, reason in [({'paused': 'yes'}, 'invalid_params'), ({'paused': True, 'profile': 'foreign'},
                                       'profile_mismatch')]:
                    assert (await rpc(ws, 'delegation.pause', **params))['error']['message'] == reason
                assert (await rpc(ws, 'delegation.status'))['result']['paused'] is False
        asyncio.run(probe())


async def _git_rpc(ws, method, **params):
    """``rpc`` with room for checkpoint git calls: each one resolves Hermes' managed git first, which
    can take a second per call on a cold test home (the shared helper waits 20 s)."""
    await ws.send(json.dumps({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params}))
    async with asyncio.timeout(120):
        while True:
            reply = json.loads(await ws.recv())
            if reply.get('id') == method:
                return reply


@pytest.mark.platforms("linux")
def test_session_export_keeps_the_assistant_reply_and_refuses_a_client_path(tmp_path):
    """T11: the JSON export carries the assistant's reply, and a client cannot steer a gateway-side
    write with ``path`` (unknown params are refused; the client alone picks where the file goes)."""
    with owner(tmp_path) as (home, _peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='export', source='tui', toolsets=[]))['result']['session_id']
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='first', text='EXPORT_ME')
                await settled(home)
                exported = (await rpc(ws, 'session.export', session_id=sid, format='json'))['result']
                contents = [m.get('content') for m in json.loads(exported['content'])['messages']]
                assert 'CONTROL_ACK' in contents, contents
                refused = await rpc(ws, 'session.export', session_id=sid, format='md', path='/etc/x')
                assert refused['error']['message'] == 'invalid_params', refused
                assert not (home / 'sessions' / 'saved').exists(), 'session.export must not write on the gateway host'
        asyncio.run(probe())


def _checkpoint(home, cwd, reason):
    """A checkpoint in the owner profile's shared store (what the agent takes before a write)."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.checkpoint_manager import CheckpointManager
    token = set_hermes_home_override(str(home))
    try:
        assert CheckpointManager(enabled=True).ensure_checkpoint(str(cwd), reason)
    finally:
        reset_hermes_home_override(token)


@pytest.mark.platforms("linux")
def test_rollback_lists_diffs_and_restores_the_session_cwd_and_rewinds_the_last_turn(tmp_path):
    with owner(tmp_path) as (home, peer, desc):
        work = tmp_path / 'work'
        work.mkdir()
        (work / 'a.txt').write_text('original\n')
        (work / 'b.txt').write_text('original b\n')

        async def probe():
            async with websocket(home, desc) as ws:
                plain = (await rpc(ws, 'session.create', request_id='plain', source='tui', toolsets=[],
                                   cwd=str(work)))['result']['session_id']
                assert (await _git_rpc(ws, 'rollback.list', session_id=plain))['result'] == {'enabled': False, 'checkpoints': []}
                sid = (await rpc(ws, 'session.create', request_id='rollback', source='tui', toolsets=[],
                                 cwd=str(work), checkpoints=True))['result']['session_id']
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='first', text='FIRST')
                await settled(home)
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='second', text='SECOND')
                await settled(home)
                await asyncio.to_thread(_checkpoint, home, work, 'before write_file')
                (work / 'a.txt').write_text('edited\n')
                (work / 'b.txt').write_text('edited b\n')
                listed = await _git_rpc(ws, 'rollback.list', session_id=sid)
                assert listed.get('result', {}).get('enabled') is True, listed
                [row] = listed['result']['checkpoints']
                assert row['message'] == 'before write_file' and row['hash'], row
                diff = (await _git_rpc(ws, 'rollback.diff', session_id=sid, hash='1'))['result']
                assert 'a.txt' in diff['stat'] and 'edited' in diff['diff'], diff
                one = (await _git_rpc(ws, 'rollback.restore', session_id=sid, hash=row['hash'], file_path='a.txt'))['result']
                assert one['success'] is True and 'history_removed' not in one, one
                assert (work / 'a.txt').read_text() == 'original\n' and (work / 'b.txt').read_text() == 'edited b\n'
                # A whole-tree restore rewinds the transcript too: refused while a turn runs (main's busy check).
                peer.blocked.clear()
                peer.release.clear()
                await rpc(ws, 'prompt.submit', session_id=sid, input_id='block', text='BLOCK_REDIRECT')
                assert await asyncio.to_thread(peer.blocked.wait, 25)
                busy = await _git_rpc(ws, 'rollback.restore', session_id=sid, hash=row['hash'])
                assert busy['error']['message'] == 'session_busy', busy
                assert (work / 'b.txt').read_text() == 'edited b\n'
                peer.release.set()
                await settled(home)
                before = (await rpc(ws, 'session.resume', session_id=sid))['result']['messages']
                # The file restore took a pre-rollback snapshot, now checkpoint 1: restore the original by hash.
                full = await _git_rpc(ws, 'rollback.restore', session_id=sid, hash=row['hash'])
                assert 'result' in full, full
                full = full['result']
                assert full['success'] is True and full['history_removed'] == 2, full
                assert (work / 'b.txt').read_text() == 'original b\n'
                after = (await rpc(ws, 'session.resume', session_id=sid))['result']['messages']
                assert [m['content'] for m in after] == [m['content'] for m in before[:-2]], (before, after)
                assert after[-1]['content'] == 'CONTROL_ACK' and after[-2]['content'] == 'SECOND', after
                for method, params, reason in [
                    ('rollback.diff', {'session_id': sid}, 'invalid_params'),
                    ('rollback.restore', {'session_id': sid, 'hash': ''}, 'invalid_params'),
                    ('rollback.diff', {'session_id': sid, 'hash': '9'}, 'invalid_checkpoint'),
                    ('rollback.list', {'session_id': sid, 'profile': 'foreign'}, 'profile_mismatch'),
                ]:
                    assert (await rpc(ws, method, **params))['error']['message'] == reason
                bad = (await _git_rpc(ws, 'rollback.restore', session_id=sid, hash='9'))['result']
                assert bad['success'] is False and 'Invalid checkpoint number' in bad['error'], bad
        asyncio.run(probe())


def _agent_write(home, path, text):
    """A file write Hermes made (and recorded in the agent-write ledger that safe restore reads)."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.checkpoint_manager import CheckpointManager
    path.write_text(text)
    token = set_hermes_home_override(str(home))
    try:
        CheckpointManager(enabled=True).record_agent_write(str(path))
    finally:
        reset_hermes_home_override(token)


@pytest.mark.platforms("linux")
def test_rollback_restore_keeps_hand_edits_unless_all(tmp_path):
    """The RPC's whole-tree restore is main's safe one (the help text's "restores keep your
    hand-edits"); ``all`` is ``/rollback <N> --all``."""
    with owner(tmp_path) as (home, _peer, desc):
        work = tmp_path / 'work'
        work.mkdir()
        (work / 'agent.txt').write_text('original\n')
        (work / 'mine.txt').write_text('original mine\n')

        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='safe', source='tui', toolsets=[],
                                 cwd=str(work), checkpoints=True))['result']['session_id']
                await asyncio.to_thread(_checkpoint, home, work, 'before write_file')
                [row] = (await _git_rpc(ws, 'rollback.list', session_id=sid))['result']['checkpoints']
                await asyncio.to_thread(_agent_write, home, work / 'agent.txt', 'agent\n')
                (work / 'mine.txt').write_text('hand edit\n')
                assert (await rpc(ws, 'rollback.restore', session_id=sid, hash=row['hash'],
                                  all='maybe'))['error']['message'] == 'invalid_params'
                safe = (await _git_rpc(ws, 'rollback.restore', session_id=sid, hash=row['hash']))['result']
                assert safe['success'] is True and safe['skipped_user_edits'] == ['mine.txt'], safe
                assert (work / 'agent.txt').read_text() == 'original\n'
                assert (work / 'mine.txt').read_text() == 'hand edit\n'
                forced = (await _git_rpc(ws, 'rollback.restore', session_id=sid, hash=row['hash'], all=True))['result']
                assert forced['success'] is True and not forced.get('skipped_user_edits'), forced
                assert (work / 'mine.txt').read_text() == 'original mine\n'
        asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_rollback_restore_reads_a_null_all_as_omitted(tmp_path):
    """The published contract declares ``all`` nullable (``RollbackRestoreParams``): a client sending
    its declared default gets the safe restore, like one sending nothing or ``false``."""
    with owner(tmp_path) as (home, _peer, desc):
        work = tmp_path / 'work'
        work.mkdir()
        (work / 'agent.txt').write_text('original\n')
        (work / 'mine.txt').write_text('original mine\n')
        out = {}

        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='null-all', source='tui', toolsets=[],
                                 cwd=str(work), checkpoints=True))['result']['session_id']
                await asyncio.to_thread(_checkpoint, home, work, 'before write_file')
                [row] = (await _git_rpc(ws, 'rollback.list', session_id=sid))['result']['checkpoints']
                await asyncio.to_thread(_agent_write, home, work / 'agent.txt', 'agent\n')
                for case, extra in (('omitted', {}), ('null', {'all': None}), ('false', {'all': False}),
                                    ('true', {'all': True})):
                    (work / 'mine.txt').write_text('hand edit\n')
                    reply = await _git_rpc(ws, 'rollback.restore', session_id=sid, hash=row['hash'], **extra)
                    out[case] = (reply.get('result', {}).get('skipped_user_edits')
                                 or reply.get('error', {}).get('message'), (work / 'mine.txt').read_text())
        asyncio.run(probe())
    kept = (['mine.txt'], 'hand edit\n')
    assert out == {'omitted': kept, 'null': kept, 'false': kept, 'true': (None, 'original mine\n')}, out


@pytest.mark.platforms("linux")
def test_config_get_voice_reports_the_sessions_own_voice_mode(tmp_path):
    """T9: Ink's /voice runs the owner's per-session handler (slash.exec) and reads the mode back
    with ``config.get key=voice``; the sidecar's process-global HERMES_VOICE is never consulted."""
    with owner(tmp_path) as (home, _peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='voice', source='tui', toolsets=[]))['result']['session_id']
                other = (await rpc(ws, 'session.create', request_id='voice2', source='tui', toolsets=[]))['result']['session_id']
                assert (await rpc(ws, 'config.get', key='voice', session_id=sid))['result']['value'] == 'off'
                on = await rpc(ws, 'slash.exec', command='voice on', session_id=sid)
                assert on['result']['type'] == 'exec', on
                assert (await rpc(ws, 'config.get', key='voice', session_id=sid))['result']['value'] == 'voice_only'
                assert (await rpc(ws, 'config.get', key='voice', session_id=other))['result']['value'] == 'off'
                await rpc(ws, 'slash.exec', command='voice off', session_id=sid)
                assert (await rpc(ws, 'config.get', key='voice', session_id=sid))['result']['value'] == 'off'
                assert (await rpc(ws, 'config.get', key='voice'))['error']['message'] == 'invalid_params'
        asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_voice_record_follows_the_sessions_gateway_voice_mode(tmp_path):
    """T9: push-to-talk (``voice.record``) gates on the session's gateway voice mode, the state
    Ink's ``/voice on`` sets through slash.exec, not the sidecar's process-global HERMES_VOICE."""
    with owner(tmp_path) as (home, _peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='rec', source='tui', toolsets=[]))['result']['session_id']
                off = await rpc(ws, 'voice.record', action='start', session_id=sid)
                assert off.get('error', {}).get('code') == 4015, off
                await rpc(ws, 'slash.exec', command='voice on', session_id=sid)
                on = await rpc(ws, 'voice.record', action='start', session_id=sid)
                # No microphone in CI: any answer but "voice mode is off" proves the gate passed.
                assert on.get('error', {}).get('code') != 4015, on
                await rpc(ws, 'voice.record', action='stop', session_id=sid)
                await rpc(ws, 'slash.exec', command='voice off', session_id=sid)
                again = await rpc(ws, 'voice.record', action='start', session_id=sid)
                assert again.get('error', {}).get('code') == 4015, again
        asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_cli_only_commands_refuse_with_what_to_use_instead(tmp_path):
    """T10: the commands a local session cannot run on the shared gateway keep the
    ``unsupported_command`` reason, and the refusal now names the alternative in ``data.detail``."""
    with owner(tmp_path) as (home, _peer, desc):
        async def probe():
            async with websocket(home, desc) as ws:
                sid = (await rpc(ws, 'session.create', request_id='refuse', source='tui', toolsets=[]))['result']['session_id']
                for command, hint in [('export', 'hermes profile export'), ('import x.tar.gz', 'hermes profile import'),
                                      ('snapshot', 'hermes backup'), ('snap list', 'hermes backup'),
                                      ('handoff telegram', 'hermes --cli'), ('hatch a cat', 'hermes --cli'),
                                      ('worktree new', 'hermes -w'), ('palette', 'Type /')]:
                    refused = (await rpc(ws, 'slash.exec', command=command, session_id=sid))['error']
                    assert refused['message'] == 'unsupported_command', (command, refused)
                    assert hint in refused['data'].get('detail', ''), (command, refused)
        asyncio.run(probe())
