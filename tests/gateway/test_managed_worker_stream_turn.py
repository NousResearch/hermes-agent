"""A managed turn's live deltas and its stored rows carry one stream identity (#144609).

The owner mints the execution's presentation ``turn_id``, hands it to the worker on the verified
bootstrap and stamps the deltas it relays with it; the worker stamps the same id on every assistant /
tool row it writes. A viewer that drew the turn live can then tell that turn's committed rows before
its receipt, as for an in-process turn. Real owner (SessionAuthority over SQLite, a child process
relaying the frames) and real worker ``execute``; only the model is scripted."""
import asyncio
import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionActor, SessionRef
from tests.gateway.fixtures.storage_handles import storage_handle

# Introduces itself, hands the bootstrap it received to the test's worker, relays that worker's frames.
CHILD = """import json, os, sys, time, psutil
from pathlib import Path
from hermes_cli.process_identity import worker_birth
me, turn = psutil.Process(), Path(sys.argv[1])
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
send = lambda f: out.write((json.dumps(f) + '\\n').encode())
send({'type': 'hello', 'pid': me.pid, 'birth': worker_birth(me), 'ancestors': [p.pid for p in me.parents()[:3]]})
for line in sys.stdin.buffer:
    frame = json.loads(line)
    if 'version' in frame:
        (turn / 'boot.json').write_text(line.decode())
        while not (turn / 'frames.json').exists():
            time.sleep(0.02)
        for relayed in json.loads((turn / 'frames.json').read_text()):
            send(relayed)
    elif frame == {'type': 'finish'}:
        send({'type': 'finished'})
        break
"""


def _run_worker(frame, home, monkeypatch_state, tool_first=False):
    """The worker's real ``execute`` on the owner's bootstrap; its agent writes real rows and streams
    (``tool_first``: the turn calls its tool before it writes any text)."""
    from agent import managed_worker as worker
    from agent.session_persistence import SessionPersistenceMixin
    from persistence_assembly import open_persistence, profile_scope

    sent = []

    class Agent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.session_id, self.model, self.provider, self.context_compressor = 's', 'm', 'custom', None
            self.session_prompt_tokens = self.session_completion_tokens = 0

        def run_conversation(self, text, conversation_history=None, **_kwargs):
            history = list(conversation_history or [])
            messages = history + [
                {'role': 'user', 'content': text},
                {'role': 'assistant', 'content': None if tool_first else 'Checking', 'tool_calls': [
                    {'id': 'call_0', 'type': 'function', 'function': {'name': 'terminal', 'arguments': '{}'}}]},
                {'role': 'tool', 'content': 'ok', 'tool_call_id': 'call_0'},
                {'role': 'assistant', 'content': 'Done.'}]
            if not tool_first:
                self.kwargs['stream_delta_callback']('Chec')
                self.kwargs['stream_delta_callback']('king')
            self.kwargs['tool_start_callback']('call_0', 'terminal', {})
            self.kwargs['tool_complete_callback']('call_0', 'terminal', {}, 'ok')
            self.kwargs['stream_delta_callback']('Done.')
            with open_persistence(profile_scope(home=home)) as store:
                self.persistence = self._session_db = store
                self._session_db_created = True
                assert SessionPersistenceMixin._flush_messages_to_session_db_unlocked(self, messages, history)
            return {'final_response': 'Done.', 'messages': messages}

    class Store:
        def __init__(self, *args): pass
        def get_messages_as_conversation(self, sid): return []
        def flush_token_counts(self): pass
        def finish(self): pass
        def close(self): pass

    class Controls:
        def __init__(self, *args):
            self.stopped, self.finish = threading.Event(), threading.Event()
            self.finish.set()
        def approval(self, data): pytest.fail('unexpected approval')
        def clarify(self, questions): pytest.fail('unexpected question')

    for target, value in (('bind_worker_policy', lambda frame: None), ('discover_profile_mcp', lambda policy: None),
                          ('retire_agent', lambda agent: None), ('WorkerControls', Controls)):
        monkeypatch_state.setattr(worker, target, value)
    monkeypatch_state.setattr('agent.runtime_session_store.WorkerRPC', lambda home: lambda *a, **k: {
        'owner_epoch': frame['scope']['epoch']})
    monkeypatch_state.setattr('agent.runtime_session_store.RuntimeSessionStore', Store)
    monkeypatch_state.setattr('tools.process_registry.process_registry.recover_from_checkpoint', lambda: None)
    monkeypatch_state.setattr('run_agent.AIAgent', Agent)
    monkeypatch_state.setattr('agent.title_generator.wait_for_title_upgrades', lambda: None)
    worker.execute(frame, SimpleNamespace(send=lambda kind, **payload: sent.append({'type': kind, **payload})))
    # The test's child relays the worker's turn frames; it introduces itself (its own pid) and closes.
    return [f for f in sent if f['type'] not in ('ready', 'finished')]


@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize('tool_first', [False, True], ids=['text-first', 'tool-first'])
def test_managed_turn_rows_and_relayed_deltas_share_one_identity_per_execution(tmp_path, monkeypatch, tool_first):
    from gateway import session_managed_worker as managed
    from gateway.session_authority import LiveSession, SessionAuthority, admission_row
    from gateway.session_contract import Submission
    from gateway.session_local import _bypass_policy
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_read import transcript_rows
    from tests.fixtures.storage_seed import seed_session
    import hermes_cli.config

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_GATEWAY_SESSION', raising=False)  # the worker's execute sets it
    # The bootstrap shape test_managed_turn_contract's worker runs: a bypass session, no launch key.
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda *args: None)
    monkeypatch.setattr(hermes_cli.config, 'load_config_readonly', dict)
    db = open_persistence(profile_scope(home=tmp_path))
    seed_session(db, 's', 'test')
    runner = SimpleNamespace(_draining=False, _cached_agent_for=lambda route: None,
                             _adapter_for_source=lambda source: None)
    profile = str(tmp_path)  # a profile id is its home (the worker's bootstrap checks it)
    authority = SessionAuthority(runner, profile_id=profile, instance_id='i',
                                 epoch=db.runtime.begin_epoch(instance_id='i'), persistence=storage_handle(tmp_path))
    live = authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human', chat_id='c'),
                                                 'managed-stream-turn')
    actor = SessionActor('human', profile, frozenset({'session:submit', 'session:control'}), 'cli')
    ref = SessionRef(profile, 's')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    published = []
    live.event_stream.observers.add(lambda frame: published.append(frame['params']))
    turns = [tmp_path / 'turn0', tmp_path / 'turn1']
    spawned = []

    class Child(subprocess.Popen):
        def __init__(self, args, **kw):
            turns[len(spawned)].mkdir()
            super().__init__([sys.executable, '-c', CHILD, str(turns[len(spawned)])], **kw)
            spawned.append(self)
    monkeypatch.setattr(managed.subprocess, 'Popen', Child)
    policy = _bypass_policy({'cwd': str(tmp_path), 'model': 'fixture', 'provider': 'custom',
                             'base_url': 'http://127.0.0.1:9/v1', 'ignore_user_config': True}, private_secrets={})
    boots, claims, failures = [], [], []

    def worker_side():
        try:
            for turn in turns:
                deadline = time.monotonic() + 30
                while not (turn / 'boot.json').exists():
                    assert time.monotonic() < deadline, 'the owner never bootstrapped the worker'
                    time.sleep(0.02)
                frame = json.loads((turn / 'boot.json').read_text())
                boots.append(frame)
                with pytest.MonkeyPatch.context() as patch:
                    relayed = _run_worker(frame, tmp_path, patch, tool_first)
                (turn / 'frames.json').write_text(json.dumps(relayed))
        except BaseException as exc:  # surfaced on the test thread below
            failures.append(exc)

    async def scenario():
        rows = []
        for n in range(2):
            await authority.submit(actor, Submission(f'turn-{n}', ref, {'text': f'go {n}'}, 'queue'))
            row = admission_row(db.admissions.claim(epoch=authority.epoch, session_id='s'))
            authority._stamp_claim(ref, live, row)  # the drain's claim stamp
            async with asyncio.timeout(40):
                assert await managed.execute_managed(authority, ref, row, policy) == 'Done.'
            published.append({'type': '<settled>'})
            captured = authority.pending_results.pop(row['admission_id'])
            db.admissions.settle(epoch=authority.epoch, admission_id=row['admission_id'], generation=row['generation'],
                                 outcome='completed', result={'result': captured['result'], 'usage': captured['usage']})
            claims.append(row)
            rows.append(transcript_rows(db, 's'))
        return rows

    thread = threading.Thread(target=worker_side, daemon=True)
    thread.start()
    with db, authority.persistence:
        stored = asyncio.run(scenario())
    thread.join(10)
    assert not failures, failures
    assert claims[0]['generation'] != claims[1]['generation']

    minted = [json.loads(boot['policy']['request_json'])['stream_turn_v1']['turn_id'] for boot in boots]
    assert all(isinstance(turn, str) and turn for turn in minted) and minted[0] != minted[1]
    # Live (owner → viewer): each claim's message.start already names the id its worker got, before any
    # output (a tool before the first text included); the relayed deltas name it too, in order.
    per_turn = [[]]
    for params in published:
        if params['type'] == '<settled>':
            per_turn.append([])
        elif params['type'] in ('message.start', 'message.delta', 'tool.start'):
            per_turn[-1].append((params['type'], params['payload']))
    for frames, turn in zip(per_turn[:2], minted, strict=True):
        assert frames[0] == ('message.start', {'turn_id': turn})
        assert [kind for kind, _ in frames].index('tool.start') == (1 if tool_first else 3)
        deltas = [payload for kind, payload in frames if kind == 'message.delta']
        assert [(d['text'], d['turn_id'], d['segment'], d['offset']) for d in deltas] == (
            [('Done.', turn, 0, 0)] if tool_first
            else [('Chec', turn, 0, 0), ('king', turn, 0, 4), ('Done.', turn, 0, 8)])
    # Stored (worker → persistence): each turn's assistant/tool rows carry the same id; the prompt none.
    def stamps(rows):
        return [(r['role'], (r['display_metadata'] or {}).get('stream_turn_id')) for r in rows]

    first_rows, all_rows = stored[0], stored[1]
    for rows, turn in ((first_rows, minted[0]), (all_rows[len(first_rows):], minted[1])):
        assert stamps(rows) == [('user', None), ('assistant', turn), ('tool', turn), ('assistant', turn)]
    # No alias across generations: the first turn's rows keep the first id after the second turn ran.
    assert stamps(all_rows[:len(first_rows)]) == stamps(first_rows)
    # The private reservation (execution id, secret) never reaches the viewer stream or the rows.
    public = json.dumps([published, stored])
    for boot in boots:
        assert boot['scope']['secret'] not in public and boot['scope']['execution_id'] not in public
        assert boot['scope']['execution_id'] not in minted
