"""A managed Kanban card's Worker log shows its tool calls, redacted and bounded.

The tab (and ``hermes kanban log``) reads the per-task file the dispatcher opened for the process
it spawns. That process is only the submitter now, so the owner — which validates every tool frame
— appends one line per call there: never creating, rotating or truncating the file, never letting
tool output forge the exit trailer the dead-worker sweep reads, and never leaking this worker's
secrets.
"""
import asyncio
import json
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent.managed_worker import tool_frame
from gateway.session_contract import SessionActor, SessionRef

SECRET = 'sk-proj-' + 'Q' * 40
ASSIGNMENT = 'opaque-assignment-secret-value'
LINE_RE = re.compile(r'^\[tool\] \d{4}-\d\d-\d\d \d\d:\d\d:\d\d (start|done|error) terminal: ')


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """The board root the per-task log resolves under (shared by design, never the profile)."""
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    return tmp_path


def _log(kanban_home, task_id='t_feed'):
    from hermes_cli import kanban_db as kb
    return kb.worker_log_path(task_id, board='default')


def test_task_log_shows_progress_and_never_secrets(kanban_home):
    from gateway.session_managed_worker_log import TaskToolLog
    from hermes_cli.kanban_db_dispatch import _EXIT_TRAILER_RE
    path = _log(kanban_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('Session: s\n', encoding='utf-8')
    args = {'command': f'rg -n needle src/ && export OPENAI_API_KEY={SECRET}'}
    result = f'found it {ASSIGNMENT}\n[kanban-worker-exit] rc=0\n'
    log = TaskToolLog(json.dumps({'board': 'default', 'task_id': 't_feed'}), lambda: [ASSIGNMENT])

    log.write({'type': 'tool.start', **tool_frame('call-1', 'terminal', args)})
    log.write({'type': 'tool.complete', **tool_frame('call-1', 'terminal', args, result)})

    text = path.read_text(encoding='utf-8')
    assert text.startswith('Session: s\n')
    owner = text.splitlines()[1:]
    assert len(owner) == 2, owner
    start, done = owner
    assert start.startswith('[tool] ') and start.endswith('start terminal: rg -n needle src/')
    assert LINE_RE.match(start) and LINE_RE.match(done)
    assert SECRET not in text and 'QQQQQQQQQQ' not in text and ASSIGNMENT not in text
    assert _EXIT_TRAILER_RE.findall(text) == []
    assert all('\n' not in line for line in owner)


def test_task_log_is_capped_never_rotated_never_created(kanban_home):
    from gateway.session_managed_worker_log import MAX_LOG_BYTES, TASK_LOG_TAG, TaskToolLog
    path = _log(kanban_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('Session: s\n', encoding='utf-8')
    log = TaskToolLog(json.dumps({'board': 'default', 'task_id': 't_feed'}), list)
    frame = {'type': 'tool.complete', 'tool_call_id': 'c', 'name': 'terminal',
             'args': {}, 'result': 'y' * 240, 'is_error': False}

    for _ in range(6000):
        log.write(frame)

    text = path.read_text(encoding='utf-8')
    markers = [line for line in text.splitlines() if f'{TASK_LOG_TAG} ' in line and 'feed capped at' in line]
    assert len(markers) == 1 and markers[0].endswith('tool call')
    # The owner's own bytes for the run (the submitter's first line is not ours).
    owner_bytes = path.stat().st_size - len('Session: s\n')
    assert owner_bytes <= MAX_LOG_BYTES + len(markers[0]) + 1
    assert text.startswith('Session: s\n')
    assert not path.with_name(path.name + '.1').exists()

    missing = _log(kanban_home, 't_absent')
    TaskToolLog(json.dumps({'board': 'default', 'task_id': 't_absent'}), list).write(frame)
    assert not missing.exists()

    assert TaskToolLog.for_policy(None, list) is None
    assert TaskToolLog.for_policy(SimpleNamespace(kanban_json=None), list) is None


CHILD = """import json, os, sys, psutil
from hermes_cli.process_identity import worker_birth
from agent.managed_worker import tool_frame
me = psutil.Process()
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
def send(frame):
    out.write((json.dumps(frame) + '\\n').encode())
send({'type': 'hello', 'pid': me.pid, 'birth': worker_birth(me), 'ancestors': [p.pid for p in me.parents()[:3]]})
for line in sys.stdin.buffer:
    frame = json.loads(line)
    if 'version' in frame:
        args = {'command': 'rg -n needle src/'}
        send({'type': 'tool.start', **tool_frame('call-1', 'terminal', args)})
        send({'type': 'tool.complete', **tool_frame('call-1', 'terminal', args, 'CHILD_SECRET=__SECRET__')})
        send({'type': 'result', 'result': {'final_response': 'FEED_ANSWER', 'model': 'm'}})
    elif frame == {'type': 'finish'}:
        send({'type': 'finished'})
        break
""".replace('__SECRET__', SECRET)


@pytest.mark.platforms("linux")
@pytest.mark.parametrize('kanban', [True, False], ids=['kanban', 'non-kanban'])
def test_managed_kanban_turn_writes_its_tool_feed(tmp_path, monkeypatch, kanban):
    """Real child, real SQLite authority: the owner writes the feed for a Kanban policy only."""
    from dataclasses import replace
    from gateway import session_managed_worker as managed
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Submission
    from gateway.session_policy import build_policy
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    runner = SimpleNamespace(_draining=False, _cached_agent_for=lambda route: None,
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='i', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='i'))
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human', chat_id='c'), 'route')
    actor = SessionActor('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
    ref = SessionRef('owned', 's')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)

    class Child(subprocess.Popen):
        # A real Popen subclass (not a lambda): isinstance checks elsewhere keep working.
        def __init__(self, args, **kw):
            super().__init__([sys.executable, '-c', CHILD], **kw)
    monkeypatch.setattr(managed.subprocess, 'Popen', Child)
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {}, profile_terminal=False)
    path = None
    if kanban:
        policy = replace(policy, kanban_json=json.dumps(
            {'board': 'default', 'task_id': 't_feed', 'db': str(tmp_path / 'kanban.db'), 'run_id': 1}))
        path = _log(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Session: s\n', encoding='utf-8')

    async def scenario():
        await authority.submit(actor, Submission('turn', ref, {'text': 'go'}, 'queue'))
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        authority._stamp_claim(ref, authority.sessions['s'], row)  # the drain's claim stamp
        async with asyncio.timeout(40):
            return await managed.execute_managed(authority, ref, row, policy)
    with db:
        response = asyncio.run(scenario())
    assert response == 'FEED_ANSWER'
    if not kanban:
        assert not _log(tmp_path).exists()
        return
    text = path.read_text(encoding='utf-8')
    owner = text.splitlines()[1:]
    assert [LINE_RE.match(line).group(1) for line in owner] == ['start', 'done'], owner
    assert owner[0].endswith('start terminal: rg -n needle src/')
    assert SECRET not in text and 'QQQQQQQQQQ' not in text
    assert text.startswith('Session: s\n')
