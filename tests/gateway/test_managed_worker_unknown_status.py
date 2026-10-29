"""A lost managed worker's message.complete carries the contract's error TurnStatus next to
outcome 'unknown' (N117), so a client that renders by status never shows a normal finish."""
import asyncio
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.session_contract import Principal, SessionRef

CHILD = """import json, os, sys, psutil
from hermes_cli.process_identity import worker_birth
me = psutil.Process()
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
out.write((json.dumps({'type': 'hello', 'pid': me.pid, 'birth': worker_birth(me),
                       'ancestors': [p.pid for p in me.parents()[:3]]}) + '\\n').encode())
for line in sys.stdin.buffer:
    out.write(b'{"type": "bogus"}\\n')
    break
"""


def _authority_with_a_lost_worker(tmp_path, monkeypatch):
    """A real authority whose next managed turn's child says hello, then sends a bogus frame."""
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

    async def claim():
        await authority.submit(actor, Submission('turn', ref, {'text': 'go'}, 'queue'))
        return claim_session_input(db, epoch=authority.epoch, session_id='s')
    return authority, ref, policy, claim, published, db


@pytest.mark.platforms("linux")
def test_unknown_worker_loss_publishes_error_status(tmp_path, monkeypatch):
    from gateway import session_managed_worker as managed
    authority, ref, policy, claim, published, db = _authority_with_a_lost_worker(tmp_path, monkeypatch)

    async def scenario():
        row = await claim()
        with pytest.raises(managed.ManagedExecutionUnknown):
            async with asyncio.timeout(30):
                await managed.execute_managed(authority, ref, row, policy)
    with db:
        asyncio.run(scenario())
    complete = [p for kind, p in published if kind == 'message.complete' and p.get('outcome') == 'unknown']
    assert complete and complete[0]['status'] == 'error', published


def _kanban_loss_recorder(monkeypatch, policy, started, release, write_s):
    """Record the order of the board's worker_lost write (held until ``release``) and the
    admission's settlement."""
    import dataclasses
    from gateway import session_kanban, session_worker_reservation
    order = []

    def record(*_, **__):
        started.set()
        release.wait(write_s)
        order.append('worker_lost')
    settle = session_worker_reservation.lose_admission_worker
    monkeypatch.setattr(session_kanban, 'record_worker_loss', record)
    monkeypatch.setattr(session_worker_reservation, 'lose_admission_worker',
                        lambda *a, **kw: (order.append('settled'), settle(*a, **kw))[1])
    return dataclasses.replace(policy, kanban_json='{"task_id": "t"}'), order


@pytest.mark.platforms("linux")
def test_a_lost_card_worker_is_on_the_board_before_its_admission_settles(tmp_path, monkeypatch):
    """The settled 'unknown' lets the submitter exit and the crash sweep book the run, so the
    worker_lost note must already be there."""
    from gateway import session_managed_worker as managed
    import threading
    authority, ref, policy, claim, _published, db = _authority_with_a_lost_worker(tmp_path, monkeypatch)
    # ``release`` is never set: the write takes its full 1 s, slower than the settlement.
    started, release = threading.Event(), threading.Event()
    policy, order = _kanban_loss_recorder(monkeypatch, policy, started, release, write_s=1)

    async def scenario():
        row = await claim()
        authority.executing[ref.session_id] = row['generation']  # as the drain stamps it
        with pytest.raises(managed.ManagedExecutionUnknown):
            async with asyncio.timeout(30):
                await managed.execute_managed(authority, ref, row, policy)
    with db:
        asyncio.run(scenario())
    assert order == ['worker_lost', 'settled']


@pytest.mark.platforms("linux")
def test_a_cancellation_while_the_board_is_busy_still_settles_the_admission(tmp_path, monkeypatch):
    import threading
    from tests.waiting import async_wait_until
    from gateway import session_managed_worker as managed
    authority, ref, policy, claim, _published, db = _authority_with_a_lost_worker(tmp_path, monkeypatch)
    started, busy_board_frees = threading.Event(), threading.Event()
    policy, order = _kanban_loss_recorder(monkeypatch, policy, started, busy_board_frees, write_s=30)

    async def scenario():
        row = await claim()
        authority.executing[ref.session_id] = row['generation']
        drain = asyncio.create_task(managed.execute_managed(authority, ref, row, policy))
        await async_wait_until(started.is_set, timeout=30)
        drain.cancel()  # while the loss handler waits on the board write
        with pytest.raises(asyncio.CancelledError):
            await drain
        busy_board_frees.set()
        await asyncio.gather(*authority._mutation_tasks)
        return row
    with db:
        row = asyncio.run(scenario())
    assert order == ['settled', 'worker_lost']
    import sqlite3
    with sqlite3.connect(tmp_path / 'state.db') as ledger:
        status = ledger.execute('SELECT status FROM session_admissions WHERE admission_id=?',
                                (row['admission_id'],)).fetchone()[0]
    assert status == 'unknown'  # settled, not left 'started' by the cancellation
