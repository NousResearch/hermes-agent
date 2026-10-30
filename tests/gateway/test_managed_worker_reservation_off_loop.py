"""The managed worker's reservation and loss writes run off the owner loop (B1-13): a state.db
writer held by another process (up to _WRITE_PATIENCE_S) must not freeze every other session,
timer, socket and Stop on the loop. Real child, real SQLite, a real competing writer."""
import asyncio
from itertools import pairwise
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from gateway.session_contract import Principal, SessionRef
from tests.waiting import wait_until

CHILD = """import json, os, sys, time, psutil
from hermes_cli.process_identity import worker_birth
me = psutil.Process()
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
def send(frame):
    out.write((json.dumps(frame) + '\\n').encode())
send({'type': 'hello', 'pid': me.pid, 'birth': worker_birth(me), 'ancestors': [p.pid for p in me.parents()[:3]]})
for line in sys.stdin.buffer:
    frame = json.loads(line)
    if 'version' in frame:
        open(sys.argv[2], 'w').close()
        time.sleep(.5)
        send({'type': 'bogus'})  # an invalid frame: the owner revokes the assignment
    elif frame == {'type': 'finish'}:
        break
"""
HOLD_S = 3.0


@pytest.mark.platforms("linux")
@pytest.mark.parametrize('held_during', ['reserve', 'lose'])
def test_reservation_and_loss_writes_never_block_the_loop(tmp_path, monkeypatch, held_during):
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
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human', chat_id='c'), 'route')
    actor = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
    ref = SessionRef('owned', 's')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    bootstrapped = tmp_path / 'bootstrapped'

    class Child(subprocess.Popen):
        def __init__(self, args, **kw):
            super().__init__([sys.executable, '-c', CHILD, held_during, str(bootstrapped)], **kw)
    monkeypatch.setattr(managed.subprocess, 'Popen', Child)
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {}, profile_terminal=False)
    held = threading.Event()

    def competing_writer():
        if held_during == 'lose':
            wait_until(bootstrapped.exists, timeout=30, what='the worker to receive its bootstrap')
        rival = sqlite3.connect(tmp_path / 'state.db', isolation_level=None)
        rival.execute('BEGIN IMMEDIATE')
        held.set()
        time.sleep(HOLD_S)  # health: allow HX013 -- the held writer's duration IS the behaviour under test
        rival.execute('ROLLBACK')
        rival.close()

    async def scenario():
        await authority.submit(actor, Submission('turn', ref, {'text': 'go'}, 'queue'))
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        writer = threading.Thread(target=competing_writer, daemon=True)
        writer.start()
        if held_during == 'reserve':
            assert await asyncio.to_thread(held.wait, 10)
        ticks = []

        async def ticker():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(.05)  # health: allow HX013 -- the loop heartbeat probe; its gaps are the measurement
        beat = asyncio.create_task(ticker())
        try:
            async with asyncio.timeout(60):
                await managed.execute_managed(authority, ref, row, policy)
        except managed.ManagedExecutionUnknown:
            pass
        finally:
            beat.cancel()
        await asyncio.to_thread(writer.join, 10)
        return row, max(b - a for a, b in pairwise(ticks))
    with db:
        row, gap = asyncio.run(scenario())
        status = db._read_one('SELECT status FROM session_admissions WHERE admission_id=?', (row['admission_id'],))[0]
    assert held.is_set()
    assert status == 'unknown'  # the bogus frame still revokes the assignment once the writer frees
    assert gap < HOLD_S / 2, f'the owner loop stalled {gap:.2f}s behind a held state.db writer'
