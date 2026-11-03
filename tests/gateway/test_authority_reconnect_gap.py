"""A reconnecting viewer receives what its session published between the replay answer and its resume."""
import asyncio
import queue
from types import SimpleNamespace

import pytest


def _drain(peer):
    frames = []
    while not peer.frames.empty():
        frames.append(peer.frames.get_nowait()['params'])
    return frames


@pytest.mark.asyncio
async def test_resume_after_replay_delivers_the_frames_published_in_between(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import LiveSession, initialize_session_authority
    from gateway.session_contract import SessionRef, Submission
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db

    async def answer(event):
        return f'reply to {event.text}'
    runner = SimpleNamespace(_session_db=db, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id='gap', instance_id='owner')
    ref = SessionRef('gap', 'chat')
    db.create_session('chat', source='telegram')
    authority.sessions['chat'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='gap'), 'chat')

    class Peer:
        def __init__(self):
            self.frames = queue.Queue()

        def write(self, frame):
            self.frames.put(frame)
            return True

    dropped, reconnected, fresh = Peer(), Peer(), Peer()
    old = AuthorityConnection(authority, dropped, {'user_id': 'viewer'})
    new = AuthorityConnection(authority, reconnected, {'user_id': 'viewer'})

    async def turn(text):
        await authority.submit(old.actor, Submission(text, ref, {'text': text}, 'queue'))
        await authority.sessions['chat'].task

    attached = await old.resume(ref, {})
    await turn('one')
    seen = _drain(dropped)[-1]['seq']
    # The reconnect: the replay is answered on the new socket before its resume rebinds it.
    replay = (await new.dispatch({'id': 1, 'method': 'session.events.since', 'params': {
        'session_id': 'chat', 'replay_epoch': attached['replay_epoch'], 'last_sequence': seen}}))['result']
    assert replay['events'] == []
    await turn('two')  # commits in the gap: only the dead socket is subscribed
    await new.resume(ref, {})
    caught_up = _drain(reconnected)
    assert caught_up and caught_up[0]['seq'] == seen + 1
    assert [f['seq'] for f in caught_up] == list(range(seen + 1, seen + 1 + len(caught_up)))
    complete = [f for f in caught_up if f['type'] == 'message.complete']
    assert [f['payload']['text'] for f in complete] == ['reply to two']

    # A viewer that never replayed opens from the snapshot alone: no backlog is pushed to it.
    await AuthorityConnection(authority, fresh, {'user_id': 'viewer'}).resume(ref, {})
    assert _drain(fresh) == []
