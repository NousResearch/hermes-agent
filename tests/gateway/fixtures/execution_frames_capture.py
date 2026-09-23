"""Capture real ``SessionEvents.publish`` frames for the Desktop execution-authority fence tests.

Two successive owners (runtime epochs 1 and 2) each run one turn on the same
session, then the second owner applies an idle ``rename`` mutation. The Desktop
fixture ``apps/desktop/src/lib/execution-authority.frames.json`` is this
module's output; regenerate it with::

    PYTHONPATH=. .venv/bin/python tests/gateway/fixtures/execution_frames_capture.py \
        apps/desktop/src/lib/execution-authority.frames.json
"""
import asyncio
import json
import os
import queue
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


class Peer:
    def __init__(self):
        self.frames = queue.Queue()

    def write(self, frame):
        self.frames.put(frame)
        return True

    def drain(self, until):
        """Fanout writes land from a writer thread; collect through the terminal *until* type."""
        out = []
        while not out or out[-1]['params']['type'] != until:
            out.append(self.frames.get(timeout=5))
        return out


async def capture(tmp):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import LiveSession, initialize_session_authority
    from gateway.session_contract import SessionRef, Submission
    from gateway.session_controls import AuthorityConnection

    store = SessionStore(tmp / 'sessions', GatewayConfig())
    db = store._db

    async def answer(event):
        return 'answer:' + event.text

    ref = SessionRef('p', 'shared')
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='c')
    db.create_session('shared', source='telegram')
    owners = {}
    try:
        for label, instance in (('epoch1', 'owner-1'), ('epoch2', 'owner-2')):
            runner = SimpleNamespace(_session_db=db, session_store=store, _draining=False,
                                     _handle_message=answer, _adapter_for_source=lambda s: None)
            authority = await initialize_session_authority(runner, profile_id='p', instance_id=instance)
            authority.sessions['shared'] = LiveSession(source, 'shared')
            peer = Peer()
            conn = AuthorityConnection(authority, peer, {'user_id': 'viewer'})
            await conn.resume(ref, {})
            await authority.submit(conn.actor, Submission(label, ref, {'text': label}, 'queue'))
            await authority.sessions['shared'].task
            owners[label] = {'authority_epoch': authority.epoch, 'frames': peer.drain('message.complete')}
            if label == 'epoch2':
                revision = db.get_session('shared')['runtime_revision']
                response = await conn.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
                    'session_id': 'shared', 'request_id': 'rename', 'expected_revision': revision,
                    'operation': 'rename', 'payload': {'title': 'Renamed while idle'}}})
                assert 'result' in response, response
                owners[label]['idle_mutation_frames'] = peer.drain('session.updated')
            await conn.close()
    finally:
        store.close_all_db_handles()
    return {'_generated_by': 'tests/gateway/fixtures/execution_frames_capture.py', **owners}


def main():
    tmp = Path(tempfile.mkdtemp())
    os.environ['HERMES_HOME'] = str(tmp)
    Path(sys.argv[1]).write_text(json.dumps(asyncio.run(capture(tmp)), indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
