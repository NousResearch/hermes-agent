"""Authority event contracts, separate from the transport's queue unit tests."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.linux_only
def test_full_observer_cannot_block_real_authority_execution(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    home, state = tmp_path / 'home', tmp_path / 'state'
    home.mkdir()
    state.mkdir()
    env = {key: os.environ[key] for key in ('PATH', 'LANG', 'TZ') if key in os.environ}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state),
               PYTHONPATH=str(repo), PYTHONUNBUFFERED='1')
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / 'fixtures' / 'authority_events_peer.py')],
        cwd=repo, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=100)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    print(result.stdout)


@pytest.mark.asyncio
async def test_replay_matches_subscription_watermark_or_requires_snapshot(tmp_path, monkeypatch):
    import asyncio
    from dataclasses import replace
    import queue
    from types import SimpleNamespace

    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import LiveSession, initialize_session_authority
    from gateway.session_contract import SessionRef, Submission
    from gateway.session_controls import AuthorityConnection
    from tui_gateway import event_replay

    # Small real rings exercise truncation and cross-session eviction without
    # a timing race or a thousand model calls. Execution/storage are real;
    # the runner's answer boundary is deliberately deterministic here.
    monkeypatch.setattr(event_replay, '_REPLAY_BUFFER_MAX', 5)
    monkeypatch.setattr(event_replay, '_REPLAY_SESSIONS_MAX', 1)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    async def answer(event):
        return event.text
    runner = SimpleNamespace(_session_db=db, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id='replay', instance_id='owner')
    ref = SessionRef('replay', 'shared')
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='replay')
    for sid in ('shared', 'other'):
        db.create_session(sid, source='telegram')
        authority.sessions[sid] = LiveSession(source, sid)
    class Peer:
        def __init__(self):
            self.frames = queue.Queue()
        def write(self, frame):
            self.frames.put(frame)
            return True
    peers = [Peer(), Peer()]
    a, b = [AuthorityConnection(authority, peer, {'user_id': 'viewer'}) for peer in peers]
    async def since(epoch, seq, sid='shared'):
        return await b.dispatch({'id': 1, 'method': 'session.events.since', 'params': {
            'session_id': sid, 'replay_epoch': epoch, 'last_sequence': seq}})
    async def turn(n, target=ref):
        await authority.submit(b.actor, Submission(str(n), target, {'text': str(n)}, 'queue'))
        await authority.sessions[target.session_id].task
    try:
        initial = await a.resume(ref, {})
        await b.resume(ref, {})
        assert 'replay_epoch' in initial and 'last_sequence' in initial, 'resume omits replay watermark'
        await turn(1)
        frames = []
        for peer in peers:
            observed = []
            while not observed or observed[-1]['params']['type'] != 'message.complete':
                observed.append(await asyncio.to_thread(peer.frames.get, True, 5))
            frames.append(observed)
        assert frames[0] == frames[1]
        replay = (await since(initial['replay_epoch'], initial['last_sequence']))['result']
        assert not replay['snapshot_required']
        assert replay['events'] == [frame['params'] for frame in frames[0]]
        current = await b.resume(ref, {})
        assert current['last_sequence'] == frames[0][-1]['params']['seq']
        assert len(authority.sessions['shared'].subscribers) == 2
        assert (await since('old-owner', 0))['result']['snapshot_required']
        assert (await since(current['replay_epoch'], current['last_sequence'] + 1))['result']['snapshot_required']
        assert (await since(current['replay_epoch'], True))['error']['data']['reason'] == 'invalid_params'
        assert (await since(current['replay_epoch'], 0, 'missing'))['error']['data']['reason'] == 'not_found'
        original = b.actor
        b.actor = replace(original, profile_id='foreign')
        assert (await since(current['replay_epoch'], 0))['error']['data']['reason'] == 'profile_mismatch'
        b.actor = original
        await turn(2)
        await turn(3)
        assert (await since(initial['replay_epoch'], 0))['result']['snapshot_required']
        await turn(4, SessionRef('replay', 'other'))
        assert (await since(current['replay_epoch'], current['last_sequence']))['result']['snapshot_required']
        refreshed = await b.resume(ref, {})
        await turn(5)
        after = (await since(refreshed['replay_epoch'], refreshed['last_sequence']))['result']
        assert not after['snapshot_required']
        assert after['events'][-1]['type'] == 'message.complete'
        assert after['events'][-1]['payload']['text'] == '5'
        before_race = await b.resume(ref, {})
        stamp = event_replay._stamp_event
        def evict_then_stamp(frame):
            stamp({'method': 'event', 'params': {'session_id': 'competing-owner'}})
            stamp(frame)
        with monkeypatch.context() as race:
            race.setattr(event_replay, '_stamp_event', evict_then_stamp)
            await turn(6)
        raced = (await since(before_race['replay_epoch'], before_race['last_sequence']))['result']
        assert raced['snapshot_required'], 'eviction during stamp reused an old epoch/sequence'
        assert raced['replay_epoch'] != before_race['replay_epoch']
    finally:
        await a.close()
        await b.close()
        store.close_all_db_handles()
