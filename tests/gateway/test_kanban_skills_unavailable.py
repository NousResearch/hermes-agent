"""A card naming skills its profile lacks dies with a self-describing run error.

Regression for the 2026-11-02 Wyze Scale card (t_9e85fa3f): ``skills: [home-assistant-operations]``
exists only in the default profile's tree, so every fabro dispatch died ~7 s into boot. The board
recorded four ``crashed`` runs whose only explanation was ``Kanban owner refused: Kanban execution
unknown``; the real ``ValueError: kanban_skills_unavailable`` never left the profile log.

The boot failure must name every missing skill and the assignee profile, and it must be recorded on
the run's ``worker_result`` so the owner-side submitter can surface it.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing


def _claimed(tmp_path, monkeypatch, *, skills, assignee='fabro'):
    """A running task claimed by this process, with the context ``build_kanban_policy`` builds."""
    home = tmp_path / assignee
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.delenv('HERMES_KANBAN_TASK', raising=False)
    kb.init_db()
    path = kb.kanban_db_path()
    with connect_closing(path) as conn:
        task_id = kb.create_task(conn, title='skills probe', assignee=assignee, skills=skills)
        task = kb.claim_task(conn, task_id, claimer='private-host:owner')
        assert task is not None
        context = {'db': str(path), 'task_id': task_id, 'run_id': task.current_run_id,
                   'claim_lock': task.claim_lock, 'profile': assignee, 'context': '',
                   'skills': list(skills), 'goal_mode': False}
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, 'worker_bound',
                             {'pid': os.getpid(), 'claim_lock': task.claim_lock},
                             run_id=task.current_run_id)
    return path, context


def _frame(context, text='work kanban task'):
    return {'policy': {'kanban_json': json.dumps(context)}, 'text': text}


def test_unresolvable_skills_fail_boot_with_skill_and_profile_named(tmp_path, monkeypatch):
    path, context = _claimed(tmp_path, monkeypatch, skills=['nonexistent-skill'])

    with pytest.raises(ValueError) as failure:
        session_kanban._run_worker_turns(SimpleNamespace(session_id="s1"), _frame(context), [])

    message = str(failure.value)
    assert 'nonexistent-skill' in message, message
    assert "'fabro'" in message or 'fabro' in message, message
    assert message != 'kanban_skills_unavailable', 'the bare token explains nothing'
    # The board can read the same text without the profile log.
    assert session_kanban.worker_result(path, context)['error'] == message


def test_every_missing_skill_is_named(tmp_path, monkeypatch):
    _, context = _claimed(tmp_path, monkeypatch, skills=['alpha-missing', 'beta-missing'])

    with pytest.raises(ValueError) as failure:
        session_kanban._run_worker_turns(SimpleNamespace(session_id="s1"), _frame(context), [])

    assert 'alpha-missing' in str(failure.value) and 'beta-missing' in str(failure.value)


def test_empty_skill_list_boots_normally(tmp_path, monkeypatch):
    path, context = _claimed(tmp_path, monkeypatch, skills=[])
    agent = SimpleNamespace(session_id="s1", run_conversation=lambda *_a, **_k: {'final_response': 'ran', 'completed': True})
    monkeypatch.setattr('agent.kanban_turn_recovery.recover_failed_kanban_turns', lambda *_a, **_k: None)

    assert session_kanban._run_worker_turns(agent, _frame(context), [])['final_response'] == 'ran'
    assert session_kanban.worker_result(path, context).get('error') is None


def test_a_resolvable_skill_still_boots_with_its_prompt(tmp_path, monkeypatch):
    path, context = _claimed(tmp_path, monkeypatch, skills=['demo-skill'])
    skill = tmp_path / 'fabro' / 'skills' / 'demo-skill'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\nname: demo-skill\ndescription: Demo.\n---\nBODY_SENTINEL\n',
                                    encoding='utf-8')
    seen = {}

    def run(prompt, conversation_history=None, **_kwargs):
        seen['prompt'] = prompt
        return {'final_response': 'ran', 'completed': True}

    agent = SimpleNamespace(session_id="s1", run_conversation=run)
    monkeypatch.setattr('agent.kanban_turn_recovery.recover_failed_kanban_turns', lambda *_a, **_k: None)

    assert session_kanban._run_worker_turns(agent, _frame(context), [])['final_response'] == 'ran'
    assert 'BODY_SENTINEL' in seen['prompt']
    assert session_kanban.worker_result(path, context).get('error') is None
