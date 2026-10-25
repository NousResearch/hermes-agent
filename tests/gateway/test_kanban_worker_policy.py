"""Assignee policy survives the persisted-card owner handoff."""
import json
import os
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture
def claimed_card(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import Principal

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda _: str(tmp_path))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    kb.create_board('owned')
    with closing(connect(board='owned')) as conn:
        tid = kb.create_task(conn, title='Policy', assignee='default', workspace_kind='dir',
                             workspace_path=str(workspace), skills=['card-skill'],
                             model_override='card-model', provider_override='openai', reasoning_effort='high')
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, tid)
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), actor=actor, native_owner=True)
    params = dict(board='owned', task_id=tid, run_id=task.current_run_id, claim_lock=task.claim_lock)
    return tmp_path, connection, params


def load_owner_policy(claimed_card, worker_policy, *, review=False):
    from gateway.session_kanban import build_kanban_policy
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect

    home, connection, params = claimed_card
    if review:
        with closing(connect(board='owned')) as conn:
            assert kb.request_review(conn, params['task_id'], expected_run_id=params['run_id'])
            task = kb.claim_review_task(conn, params['task_id'])
            params = dict(params, run_id=task.current_run_id, claim_lock=task.claim_lock)
    config = {'model': {'default': 'profile-model', 'provider': 'custom'},
              'platform_toolsets': {'cli': ['hermes-cli']}, 'kanban': {'worker_policy': worker_policy}}
    path = home / 'config.yaml'
    path.write_text(json.dumps(config), encoding='utf-8')
    return build_kanban_policy(connection, params, load_user_config_effective(path, fail_closed=True))[0]


@pytest.mark.parametrize('review_skill,expected', [('none', []), ('custom-review', ['custom-review'])])
def test_owner_refuses_overrides_and_keeps_review_control_separate(claimed_card, review_skill, expected):
    policy = load_owner_policy(claimed_card, dict(task_skills='none', review_skill=review_skill,
        lifecycle_toolset='none', task_model_override='deny', task_reasoning_override='deny'), review=True)
    context = json.loads(policy.kanban_json)
    assert context['skills'] == expected
    assert policy.model == 'profile-model'
    assert policy.provider == 'custom'
    assert 'reasoning' not in json.loads(policy.request_json)
    assert context['lifecycle_toolset'] == 'none'
    from hermes_cli.kanban_db_connect import connect
    with closing(connect(board='owned')) as conn:
        events = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='worker_policy_refused'",
                              (context['task_id'],)).fetchall()
    refused = json.loads(events[-1][0])['refused']
    assert {'task_skills', 'task_model_override', 'task_reasoning_override', 'lifecycle_toolset'} <= set(refused)


@pytest.mark.parametrize('value', [None, [], 'deny', {'task_skills': True}, {'task_skills': ['ok', 1]},
    {'task_skills': 'typo'}, {'review_skill': False}, {'lifecycle_toolset': []},
    {'task_model_override': None}, {'task_reasoning_override': 'deni'},
    *({field: None} for field in ('task_skills', 'review_skill', 'lifecycle_toolset')),
    {'task_skills': ['']}, {'review_skill': '  '}, {'lifecycle_toolset': ''},
    {'task_model_override': True}, {'task_reasoning_override': 1}])
def test_malformed_policy_never_grants_capabilities(claimed_card, value):
    with pytest.raises(ValueError, match='worker_policy'):
        load_owner_policy(claimed_card, value)


def test_omitted_policy_fields_keep_legacy_defaults(claimed_card):
    policy = load_owner_policy(claimed_card, {}, review=True)
    assert json.loads(policy.kanban_json)['skills'] == ['card-skill', 'sdlc-review']
    assert policy.model == 'card-model'
    assert policy.provider == 'openai'
    assert json.loads(policy.request_json)['reasoning'] == 'high'
    assert 'kanban' in policy.toolsets


@pytest.mark.parametrize('toolsets', [['hermes-cli'], ['kanban', 'terminal'], None])
def test_lifecycle_denial_after_composite_schema_expansion(monkeypatch, toolsets):
    from model_tools import get_tool_definitions
    monkeypatch.setenv('HERMES_KANBAN_TASK', 'owned')
    monkeypatch.setenv('HERMES_KANBAN_LIFECYCLE_TOOLSET', 'none')
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    names = {t['function']['name'] for t in get_tool_definitions(toolsets, quiet_mode=True,
                                                               skip_tool_search_assembly=True)}
    assert 'terminal' in names
    assert not any(name.startswith('kanban_') for name in names)


def owner_schema(policy, monkeypatch, *, disabled=()):
    """Consume the frozen owner frame through the real worker binding and schema assembly."""
    from gateway.session_kanban import bind_worker_context
    from model_tools import get_tool_definitions
    with monkeypatch.context() as scope, patch.dict(os.environ):
        scope.setattr('gateway.session_kanban._watch_submitter', lambda *_: None)
        scope.chdir(Path.cwd())
        bind_worker_context({'policy': asdict(policy)})
        return {td['function']['name'] for td in get_tool_definitions(
            list(policy.toolsets), disabled_toolsets=list(disabled), quiet_mode=True,
            skip_tool_search_assembly=True)}


def test_owner_binding_enforces_denial_and_preserves_other_tools(claimed_card, monkeypatch):
    policy = load_owner_policy(claimed_card, {'lifecycle_toolset': 'none'})
    names = owner_schema(policy, monkeypatch)
    assert 'terminal' in names
    assert not any(name.startswith('kanban_') for name in names)


def test_custom_names_and_review_policy_remain_independent(claimed_card, monkeypatch):
    from toolsets import TOOLSETS
    monkeypatch.setitem(TOOLSETS, 'custom-worker', {'tools': ['terminal', 'kanban_complete']})
    policy = load_owner_policy(claimed_card, dict(task_skills=['card-skill', 'plugin/skill'],
        review_skill='plugin/review', lifecycle_toolset='custom-worker'), review=True)
    assert json.loads(policy.kanban_json)['skills'] == ['card-skill', 'plugin/review']
    assert 'custom-worker' in policy.toolsets
    assert 'kanban_complete' in owner_schema(policy, monkeypatch)
    names = owner_schema(policy, monkeypatch, disabled=['kanban'])
    assert 'terminal' in names
    assert 'kanban_complete' not in names


def test_lifecycle_choice_partitions_schema_cache(claimed_card, monkeypatch):
    policy = load_owner_policy(claimed_card, {})
    from model_tools import get_tool_definitions
    monkeypatch.setenv('HERMES_KANBAN_TASK', 'owned')
    for choice in ('kanban', 'none', 'kanban'):
        monkeypatch.setenv('HERMES_KANBAN_LIFECYCLE_TOOLSET', choice)
        names = {td['function']['name'] for td in get_tool_definitions(
            ['hermes-cli'], quiet_mode=True, skip_tool_search_assembly=True)}
        assert 'terminal' in names
        assert ('kanban_complete' in names) == (choice == 'kanban')
    assert 'kanban_complete' in owner_schema(policy, monkeypatch)


def test_owner_refusals_are_durable_and_deduplicated(claimed_card):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    for _ in range(2):
        policy = load_owner_policy(claimed_card, {'task_skills': 'none', 'lifecycle_toolset': 'none'})
    context = json.loads(policy.kanban_json)
    with closing(connect(board='owned')) as conn:
        events = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND run_id=? AND kind='worker_policy_refused'",
                             (context['task_id'], context['run_id'])).fetchall()
        assert [json.loads(row[0])['refused'] for row in events] == [['task_skills', 'lifecycle_toolset']]
        assert kb.get_task(conn, context['task_id']).skills == ['card-skill']


@pytest.mark.parametrize('kanban', [None, [], 'invalid'])
def test_malformed_kanban_section_is_rejected(claimed_card, kanban):
    from gateway.session_kanban import build_kanban_policy
    from hermes_cli.config_effective import load_user_config_effective
    home, connection, params = claimed_card
    path = home / 'config.yaml'
    path.write_text(json.dumps({'kanban': kanban}), encoding='utf-8')
    with pytest.raises(ValueError, match='worker_policy'):
        build_kanban_policy(connection, params, load_user_config_effective(path, fail_closed=True))


@pytest.mark.parametrize('contents', [
    'kanban:\n  worker_policy: [',
    'kanban:\n  worker_policy: null',
    '- kanban:\n    worker_policy:\n      task_skills: none\n      lifecycle_toolset: none\n'
    '      task_model_override: deny\n      task_reasoning_override: deny\n',
    '[]', 'null', 'false', '42', 'invalid',
])
def test_owner_rejects_malformed_file_before_session_creation(claimed_card, monkeypatch, contents):
    import asyncio
    from ruamel.yaml.error import YAMLError
    from gateway.session_kanban import run_task
    from hermes_cli.config_effective import load_user_config_effective

    home, connection, params = claimed_card
    connection.authority.db = SimpleNamespace(db_path=str(home / 'state.db'), get_session=lambda _: None)
    path = home / 'config.yaml'
    # Prime last-good recovery with permissive defaults, then simulate a torn policy edit.
    path.write_text('{}', encoding='utf-8')
    load_user_config_effective(path, fail_closed=True)
    path.write_text(contents, encoding='utf-8')
    # Runtime readers may already have cached a tolerant interpretation of this file.
    load_user_config_effective(path)
    # An ambient launch profile must not supply defaults for this owner.
    launch_home = home / 'launch'
    launch_home.mkdir()
    (launch_home / 'config.yaml').write_text(json.dumps({
        'model': {'default': 'launch-model', 'provider': 'custom'},
        'platform_toolsets': {'cli': ['terminal']}}), encoding='utf-8')
    monkeypatch.setattr('gateway.run._gateway_config_home', lambda: launch_home)

    def unexpected_session(*args, **kwargs):
        pytest.fail('malformed configuration reached session creation')

    monkeypatch.setattr('gateway.session_local.create_local_session', unexpected_session)
    with pytest.raises((YAMLError, ValueError)):
        asyncio.run(run_task(connection, params))
