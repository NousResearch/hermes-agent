"""Managed-worker Stop parity and per-frame fence (N84 / B1-11 follow-ups).

A Stop reaching the worker hard-cancels its agent like the in-process canonical Stop
(request_hard_interrupt), both when the agent exists and when it latched during construction; and the
owner's frame loop fences on the executing generation instead of a ledger read per worker frame."""
import asyncio
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from gateway.session_contract import Principal, SessionRef
from tests.gateway.test_managed_turn_contract import assignment


class _Agent:
    calls = []

    def __init__(self, **kwargs):
        self.session_prompt_tokens = self.session_completion_tokens = 0

    def interrupt(self, message=None, **kwargs):
        self.calls.append('soft')

    def hard_interrupt(self, message=None, **kwargs):
        self.calls.append('hard')

    def run_conversation(self, text, **kwargs):
        return {'final_response': '', 'interrupted': True}


def test_worker_stop_hard_cancels_the_agent_before_and_after_construction(tmp_path, monkeypatch):
    from agent import managed_worker as worker
    _Agent.calls = []
    controls = worker.WorkerControls.__new__(worker.WorkerControls)  # no stdin reader thread
    controls.agent, controls.stopped, controls.lock, controls.clarifications = _Agent(), threading.Event(), threading.Lock(), {}
    controls.stop()
    frame, _ = assignment(tmp_path, monkeypatch)

    class Store:
        def __init__(self, *args): pass
        def get_messages_as_conversation(self, sid): return []
        def flush_token_counts(self): pass
        def finish(self): pass
        def close(self): pass

    class Latched:  # a Stop that arrived while the agent was being constructed
        def __init__(self, *args):
            self.stopped, self.finish = threading.Event(), threading.Event()
            self.stopped.set()
            self.finish.set()
        def approval(self, data): pass
        def clarify(self, questions): pass

    monkeypatch.setattr(worker, 'bind_worker_policy', lambda frame: None)
    monkeypatch.setattr(worker, 'discover_profile_mcp', lambda policy: None)
    monkeypatch.setattr(worker, 'retire_agent', lambda agent: None)
    monkeypatch.setattr(worker, 'WorkerControls', Latched)
    monkeypatch.setattr('agent.runtime_session_store.WorkerRPC', lambda home: lambda *a, **k: {'owner_epoch': 1})
    monkeypatch.setattr('agent.runtime_session_store.RuntimeSessionStore', Store)
    monkeypatch.setattr('tools.process_registry.process_registry.recover_from_checkpoint', lambda: None)
    monkeypatch.setattr('run_agent.AIAgent', _Agent)
    monkeypatch.setattr('agent.title_generator.wait_for_title_upgrades', lambda: None)
    worker.execute(frame, SimpleNamespace(send=lambda kind, **payload: None))
    assert _Agent.calls == ['hard', 'hard'], 'a managed worker Stop must hard-cancel like the canonical Stop'


CHILD = """import json, os, sys, psutil
from hermes_cli.process_identity import worker_birth
me = psutil.Process()
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
send = lambda f: out.write((json.dumps(f) + '\\n').encode())
send({'type': 'hello', 'pid': me.pid, 'birth': worker_birth(me), 'ancestors': [p.pid for p in me.parents()[:3]]})
for line in sys.stdin.buffer:
    for i in range(20):
        send({'type': 'delta', 'text': str(i)})
    send({'type': 'bogus'})
    break
"""


@pytest.mark.platforms("linux")
def test_worker_frames_read_no_ledger_per_frame(tmp_path, monkeypatch):
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
    actor = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
    ref = SessionRef('owned', 's')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)

    class Child(subprocess.Popen):
        def __init__(self, args, **kw):
            super().__init__([sys.executable, '-c', CHILD], **kw)
    monkeypatch.setattr(managed.subprocess, 'Popen', Child)
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {}, profile_terminal=False)
    log = []
    real_rows, real_publish = authority._pending_rows, authority.publish_execution
    monkeypatch.setattr(authority, '_pending_rows', lambda *a, **k: log.append('ledger') or real_rows(*a, **k))
    monkeypatch.setattr(authority, 'publish_execution', lambda *a, **k: log.append('delta') or real_publish(*a, **k))

    async def scenario():
        await authority.submit(actor, Submission('turn', ref, {'text': 'go'}, 'queue'))
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        authority._stamp_claim(ref, live, row)  # the drain's claim stamp
        log.clear()
        with pytest.raises(managed.ManagedExecutionUnknown):  # the closing bogus frame revokes the worker
            async with asyncio.timeout(30):
                await managed.execute_managed(authority, ref, row, policy)
    with db:
        asyncio.run(scenario())
    assert log.count('delta') == 20, log
    first, last = log.index('delta'), len(log) - 1 - log[::-1].index('delta')
    frames = log[first:last + 1]
    assert 'ledger' not in frames, f"{frames.count('ledger')} ledger reads for 20 worker frames"
