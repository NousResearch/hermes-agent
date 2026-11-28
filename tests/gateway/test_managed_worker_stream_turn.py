"""A managed turn's live frames name its stream turn and the stored rows they are (#144609).

The owner mints the execution's presentation ``turn_id``, names it on the claim's ``message.start`` and
stamps the deltas it relays with it. The worker's tool frames and result name the stored rows of its
round and its reply (agent/live_row_identity.py), and the owner publishes them, so a viewer that drew the
turn live can tell its pieces in a later transcript read although a worker sends no receipt. Real owner
(SessionAuthority over SQLite, a child process relaying the frames) and real worker ``execute``; only the
model is scripted."""
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
    from agent.live_row_identity import live_assistant_row
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
            with open_persistence(profile_scope(home=home)) as store:
                self.persistence = self._session_db = store
                self._session_db_created = True
                # As a tool round does (agent/turn_tool_round.py): the assistant row is stored before
                # its calls run, and the calls run naming that row.
                round_rows = len(history) + 2
                assert SessionPersistenceMixin._flush_messages_to_session_db_unlocked(self, messages[:round_rows], history)
                with live_assistant_row(self, messages[round_rows - 1]):
                    self.kwargs['tool_start_callback']('call_0', 'terminal', {})
                    self.kwargs['tool_complete_callback']('call_0', 'terminal', {}, 'ok')
                self.kwargs['stream_delta_callback']('Done.')
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
def test_managed_turn_frames_name_their_stream_turn_and_stored_rows(tmp_path, monkeypatch, tool_first):
    from gateway import session_managed_worker as managed
    from gateway.session_authority import LiveSession, SessionAuthority, admission_row
    from gateway.session_contract import Submission
    from gateway.session_local import _bypass_policy
    from gateway.session_settlement_recovery import completion_payload
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
    boots, claims, failures, completions = [], [], [], []

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
            completions.append(completion_payload(row, {'outcome': 'completed'}, 'Done.', captured))
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

    # The owner asks for the stored-row fields its frame validator accepts (agent/managed_worker.live_rows_asked).
    assert all(json.loads(boot['policy']['request_json'])['live_rows_v1'] is True for boot in boots)
    # Live (owner → viewer): each claim's message.start names its turn before any output (a tool before
    # the first text included); the relayed deltas name it too, in order.
    per_turn = [[]]
    for params in published:
        if params['type'] == '<settled>':
            per_turn.append([])
        elif params['type'] in ('message.start', 'message.delta', 'tool.start', 'tool.complete'):
            per_turn[-1].append((params['type'], params['payload']))
    minted = [frames[0][1].get('turn_id') for frames in per_turn[:2]]
    assert all(isinstance(turn, str) and turn for turn in minted) and minted[0] != minted[1]
    for frames, turn in zip(per_turn[:2], minted, strict=True):
        assert frames[0] == ('message.start', {'turn_id': turn})
        assert [kind for kind, _ in frames].index('tool.start') == (1 if tool_first else 3)
        deltas = [payload for kind, payload in frames if kind == 'message.delta']
        assert [(d['text'], d['turn_id'], d['segment'], d['offset']) for d in deltas] == (
            [('Done.', turn, 0, 0)] if tool_first
            else [('Chec', turn, 0, 0), ('king', turn, 0, 4), ('Done.', turn, 0, 8)])
    first_rows, all_rows = stored[0], stored[1]
    assert [row['role'] for row in all_rows] == ['user', 'assistant', 'tool', 'assistant'] * 2
    # The tool events and the completion name the stored rows they are (agent/live_row_identity.py): the
    # round's assistant row, the call's own uid, and the reply's row, though a worker sends no receipt.
    for frames, rows, complete in zip(per_turn[:2], (first_rows, all_rows[len(first_rows):]), completions, strict=True):
        _prompt, round_row, _tool, reply = rows
        named = {'row_id': round_row['id'], 'message_uid': round_row['message_uid'],
                 'tool_call_uid': json.loads(round_row['tool_call_uids'])['call_0']}
        tools = [payload for kind, payload in frames if kind in ('tool.start', 'tool.complete')]
        assert [{key: payload.get(key) for key in named} for payload in tools] == [named, named]
        assert (complete['row_id'], complete['message_uid']) == (reply['id'], reply['message_uid'])
        assert 'persisted_turn' not in complete
    # The private reservation (execution id, secret) never reaches the viewer stream or the rows.
    public = json.dumps([published, stored])
    for boot in boots:
        assert boot['scope']['secret'] not in public and boot['scope']['execution_id'] not in public
        assert boot['scope']['execution_id'] not in minted
