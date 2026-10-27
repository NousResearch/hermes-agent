"""Owner policy rebuilding must preserve the dispatcher's disabled-review-skill decision."""
import json
from types import SimpleNamespace

import pytest

from gateway.session_contract import Principal
from gateway.session_kanban import _run_task_turns, build_kanban_policy
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing


@pytest.mark.parametrize('disabled', [True, False])
def test_owner_review_policy_respects_disabled_automatic_skill(tmp_path, monkeypatch, disabled):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path))
    config = {'model': {'default': 'loop-model', 'provider': 'custom'},
              'platform_toolsets': {'cli': ['terminal']},
              'skills': {'disabled': ['sdlc-review'] if disabled else []}}
    (tmp_path / 'config.yaml').write_text(json.dumps(config), encoding='utf-8')
    skill = tmp_path / 'skills' / 'sdlc-review'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text(
        '---\nname: sdlc-review\ndescription: Fixture review\n---\nReview fixture guidance.\n',
        encoding='utf-8')
    kb.create_board('owned')
    with connect_closing(board='owned') as conn:
        tid = kb.create_task(conn, title='Review work', assignee='default',
                             workspace_kind='dir', workspace_path=str(tmp_path))
        impl = kb.claim_task(conn, tid)
        assert kb.request_review(conn, tid, summary='Ready', expected_run_id=impl.current_run_id)
        task = kb.claim_review_task(conn, tid)
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)),
                                 actor=actor, native_owner=True)
    params = dict(board='owned', task_id=tid, run_id=task.current_run_id, claim_lock=task.claim_lock)
    policy, _ = build_kanban_policy(connection, params, config)
    context = json.loads(policy.kanban_json)
    prompts = []

    def run(prompt, **kwargs):
        prompts.append(prompt)
        return {'completed': True, 'final_response': 'Review finished'}

    agent = SimpleNamespace(session_id='fixture-session', run_conversation=run)
    assert _run_task_turns(agent, {'text': 'Review the handoff'}, [], context)['completed']
    assert context['skills'] == ([] if disabled else ['sdlc-review'])
    assert len(prompts) == 1
    assert ('Review fixture guidance.' in prompts[0]) is not disabled
