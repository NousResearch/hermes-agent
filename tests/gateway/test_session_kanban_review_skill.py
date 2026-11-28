"""A review-lane attempt must survive an unavailable lane skill.

Regression for the live failure behind ``Kanban execution unknown; retained by owner``: a card
handed to ``kanban_request_review`` gives every reviewer spawn the bundled ``sdlc-review`` skill,
and the worker aborted the whole attempt when that skill could not be resolved in the reviewer's
profile (its copy of the bundled skill had gone missing). No turn ran, the owner settled the
admission ``unknown`` — nobody can know whether a turn that started without a result had effects —
and the dispatcher spent the card's failure budget on repeat spawns of the same abort.

The lane default is a default: an attempt without it must still reach the model. A skill the CARD
asked for that cannot load keeps aborting the attempt, which is what the original guard is for.
"""
import json
import logging
from contextlib import closing
from types import SimpleNamespace

import pytest


class RecordingAgent:
    """The two things ``_run_task_turns`` reads from a worker agent."""

    session_id = 'local-review-lane-test'

    def __init__(self):
        self.prompts = []

    def run_conversation(self, prompt, conversation_history=None, turn_author=None):
        self.prompts.append(prompt)
        return {'final_response': 'reviewed', 'completed': True, 'messages': []}


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env',
                        lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    # The live profile had lost its copy of the bundled skill; nothing in this home resolves it.
    (tmp_path / 'skills').mkdir(exist_ok=True)
    workspace = tmp_path / 'workspace'
    workspace.mkdir(exist_ok=True)
    return workspace


def _claimed_context(tmp_path, monkeypatch, *, skills=None, review):
    """A real board, a claimed run, and the policy a worker for it receives."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import SessionActor
    from gateway.session_kanban import build_kanban_policy
    workspace = _isolate(tmp_path, monkeypatch)
    kb.create_board('owned')  # explicit creation; connect() never creates a board (#43243)
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Lane default skill', body='Acceptance: review', assignee='default',
                                 workspace_kind='dir', workspace_path=str(workspace), skills=skills)
        kb.recompute_ready(conn)
        claimed = kb.claim_task(conn, task_id)
        if review:
            assert kb.request_review(conn, task_id, summary='ready', expected_run_id=claimed.current_run_id)
            claimed = kb.claim_review_task(conn, task_id)
        else:
            claimed = kb.get_task(conn, task_id)
    actor = SessionActor('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), actor=actor, native_owner=True)
    params = dict(board='owned', task_id=claimed.id, run_id=claimed.current_run_id, claim_lock=claimed.claim_lock)
    config = {'model': {'default': 'loop-model', 'provider': 'custom'},
              'platform_toolsets': {'cli': ['terminal', 'file']}, 'agent': {'coding_context': 'focus'}}
    policy, _secrets = build_kanban_policy(connection, params, config)
    return json.loads(policy.kanban_json)


def test_review_lane_runs_when_the_lane_default_skill_is_unavailable(tmp_path, monkeypatch, caplog):
    """The reviewer's attempt reaches the model, degraded, instead of dying before its first turn."""
    from gateway.session_kanban import _run_task_turns
    context = _claimed_context(tmp_path, monkeypatch, skills=None, review=True)
    assert context['skills'] == ['sdlc-review'], context['skills']
    agent = RecordingAgent()
    with caplog.at_level(logging.WARNING):
        result = _run_task_turns(agent, {'text': 'work kanban task t_owned'}, [], context)
    assert result['final_response'] == 'reviewed'
    assert agent.prompts and 'work kanban task t_owned' in agent.prompts[0]
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any('sdlc-review' in message for message in warnings), warnings


def test_card_skill_that_cannot_load_still_aborts_the_attempt(tmp_path, monkeypatch):
    """The original guard stands: a skill the card asked for is not optional."""
    from gateway.session_kanban import _run_task_turns
    context = _claimed_context(tmp_path, monkeypatch, skills=['owned-missing-skill'], review=False)
    assert context['skills'] == ['owned-missing-skill']
    with pytest.raises(ValueError, match='kanban_skills_unavailable'):
        _run_task_turns(RecordingAgent(), {'text': 'work kanban task t_owned'}, [], context)


def test_review_skill_the_card_asked_for_is_still_required(tmp_path, monkeypatch):
    """A card that names the review skill itself keeps the strict rule; only the lane default is optional."""
    from gateway.session_kanban import _run_task_turns
    context = _claimed_context(tmp_path, monkeypatch, skills=['sdlc-review'], review=True)
    assert context['skills'] == ['sdlc-review']
    with pytest.raises(ValueError, match='kanban_skills_unavailable'):
        _run_task_turns(RecordingAgent(), {'text': 'work kanban task t_owned'}, [], context)
