"""A managed worker's turn carries its author so memory providers attribute it like the owner path."""
import json
import os
from types import SimpleNamespace

import pytest


def _policy(tmp_path):
    from gateway.session_local import _bypass_policy
    return _bypass_policy({'cwd': str(tmp_path), 'model': 'safe-fixture', 'provider': 'custom',
                           'base_url': 'http://127.0.0.1:9/v1', 'ignore_user_config': True}, private_secrets={})


def test_bootstrap_frame_carries_the_admitted_author(tmp_path, monkeypatch):
    from gateway import session_managed_worker as smw
    from agent.managed_worker import validate_bootstrap

    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    live = SimpleNamespace(source=SimpleNamespace(user_id='u', chat_id='c'), route='route')
    authority = SimpleNamespace(sessions={'sid': live}, runner=None, profile_id=str(tmp_path))
    ref = SimpleNamespace(session_id='sid')
    scope = {'profile_id': str(tmp_path), 'session_id': 'sid', 'execution_id': 'x', 'generation': 1,
             'pid': os.getpid(), 'birth': 0, 'secret': 's', 'epoch': 1}
    author = {'id': 'bot-7', 'name': 'Seven', 'is_bot': True}
    row = {'payload': {'text': 'hi', 'local_automation_v1': {'turn_author': author}}}
    frame = smw._bootstrap(authority, ref, row, _policy(tmp_path), scope)
    assert frame['turn_author'] == author
    wire = json.loads(json.dumps(frame))
    assert validate_bootstrap(wire) is wire
    plain = smw._bootstrap(authority, ref, {'payload': {'text': 'hi'}}, _policy(tmp_path), scope)
    assert plain['turn_author'] is None
    assert validate_bootstrap(json.loads(json.dumps(plain)))['turn_author'] is None
    for bad in ('bot-7', {}, {'is_bot': True}):
        with pytest.raises(ValueError):
            validate_bootstrap({**wire, 'turn_author': bad})


def test_worker_turn_passes_the_author_to_run_conversation():
    from gateway.session_kanban import run_worker_turns

    seen = []
    agent = SimpleNamespace(run_conversation=lambda text, **kw: seen.append((text, kw)) or {'final_response': 'ok'})
    author = {'id': 'alice', 'name': 'alice', 'is_bot': True}
    run_worker_turns(agent, {'policy': {'kanban_json': None}, 'text': 'hello', 'turn_author': author}, [])
    run_worker_turns(agent, {'policy': {'kanban_json': None}, 'text': 'again', 'turn_author': None}, [])
    assert seen == [('hello', {'conversation_history': [], 'turn_author': author}),
                    ('again', {'conversation_history': []})]
