"""A lost managed worker's message.complete carries the contract's error TurnStatus next to
outcome 'unknown' (N117), so a client that renders by status never shows a normal finish."""
import asyncio
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.session_contract import Principal, SessionRef

CHILD = """import json, os, sys, psutil
me = psutil.Process()
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
out.write((json.dumps({'type': 'hello', 'pid': me.pid, 'birth': me.create_time(),
                       'ancestors': [p.pid for p in me.parents()[:3]]}) + '\\n').encode())
for line in sys.stdin.buffer:
    out.write(b'{"type": "bogus"}\\n')
    break
"""


@pytest.mark.platforms("linux")
def test_unknown_worker_loss_publishes_error_status(tmp_path, monkeypatch):
    from gateway import session_managed_worker as managed
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Submission
    from gateway.session_policy import build_policy
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    runner = SimpleNamespace(_draining=False, _cached_agent_for=lambda route: None,
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='i', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='i'))
    live = authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human', chat_id='c'), 'route')
    published = []
    original = live.event_stream.publish
    monkeypatch.setattr(live.event_stream, 'publish', lambda sid, payload, **kw: (
        published.append((kw.get('event_type', 'message.complete'), payload)), original(sid, payload, **kw))[1])
    actor = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
    ref = SessionRef('owned', 's')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)

    class Child(subprocess.Popen):
        def __init__(self, args, **kw):
            super().__init__([sys.executable, '-c', CHILD], **kw)
    monkeypatch.setattr(managed.subprocess, 'Popen', Child)
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {}, profile_terminal=False)

    async def scenario():
        await authority.submit(actor, Submission('turn', ref, {'text': 'go'}, 'queue'))
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        with pytest.raises(managed.ManagedExecutionUnknown):
            async with asyncio.timeout(30):
                await managed.execute_managed(authority, ref, row, policy)
    with db:
        asyncio.run(scenario())
    complete = [p for kind, p in published if kind == 'message.complete' and p.get('outcome') == 'unknown']
    assert complete and complete[0]['status'] == 'error', published
