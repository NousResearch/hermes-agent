"""Skill receipts reach the real turn footer without enrolling checkpoint/stop gates."""
import json
import logging
from unittest.mock import Mock

import pytest

from agent.turn_explainers import TurnExplainersMixin
from agent.turn_finalizer import _append_file_mutation_footer
from tools import skill_manager_tool as skills


@pytest.fixture
def agent():
    obj = TurnExplainersMixin()
    obj._turn_failed_file_mutations = {}
    obj._turn_file_mutation_paths = set()
    obj._checkpoint_mgr = Mock(enabled=True)
    obj._file_mutation_verifier_enabled_cache = True
    return obj


@pytest.fixture
def skill(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    root = tmp_path / 'skills' / 'testing' / 'sample'
    root.mkdir(parents=True)
    (root / 'SKILL.md').write_text('---\nname: sample\ndescription: Test a skill.\n---\nOriginal\n', encoding='utf8')
    return root


def record(agent, args, result=None):
    if result is None:
        from tools.registry import registry
        result = registry.dispatch('skill_manage', args)
    parsed = json.loads(result)
    agent._record_file_mutation_result('skill_manage', args, result, bool(parsed.get('error')))
    return parsed


def footer(agent):
    return _append_file_mutation_footer(agent, 'Finished.', logging.getLogger(__name__))


def test_real_validation_failure_reaches_footer(agent, skill):
    args = {'operations': [{'name': 'sample', 'action': 'write_file', 'file_path': 'scripts/run.py'}]}
    result = record(agent, args)
    assert 'file_content' in result['error']
    assert 'skill_manage' in footer(agent)
    assert 'sample' in footer(agent)
    assert 'scripts/run.py' in footer(agent)
    assert not agent._turn_file_mutation_paths
    agent._checkpoint_mgr.record_agent_write.assert_not_called()


def test_real_retry_clears_only_same_file_and_accepts_category_alias(agent, skill):
    record(agent, {'operations': [{'name': 'sample', 'action': 'write_file', 'file_path': 'scripts/run.py'}]})
    record(agent, {'operations': [{'name': 'sample', 'action': 'write_file', 'file_path': 'references/note.md'}]})
    args = {'name': 'testing/sample', 'action': 'write_file', 'file_path': 'scripts/run.py', 'file_content': 'print(1)\n'}
    assert record(agent, args)['success'] is True
    assert (skill / 'scripts/run.py').read_text() == 'print(1)\n'
    assert 'scripts/run.py' not in footer(agent)
    assert 'references/note.md' in footer(agent)
    assert not agent._turn_file_mutation_paths
    agent._checkpoint_mgr.record_agent_write.assert_not_called()


@pytest.mark.parametrize('receipt', [
    {'success': True, 'staged': True}, {}, {'success': False}, {'success': True, 'error': 'failed'},
])
def test_unproven_or_staged_retry_does_not_clear_failure(agent, skill, receipt):
    args = {'name': 'sample', 'action': 'patch', 'old_string': 'missing', 'new_string': 'new'}
    record(agent, args, json.dumps({'error': 'not found'}))
    record(agent, args, json.dumps(receipt))
    assert 'skill_manage' in footer(agent)


def test_batch_failure_tracks_all_rolled_back_targets(agent, skill):
    args = {'operations': [
        {'name': 'sample', 'action': 'patch', 'old_string': 'Original', 'new_string': 'Changed'},
        {'name': 'sample', 'action': 'patch', 'old_string': 'Not present', 'new_string': 'Oops'},
    ]}
    result = record(agent, args)
    assert result['success'] is False
    assert 'Original' in (skill / 'SKILL.md').read_text()
    assert 'skill_manage' in footer(agent)


def test_missing_action_is_reported_and_display_can_be_disabled(agent, skill):
    record(agent, {'operations': [{'name': 'sample'}]})
    assert 'sample' in footer(agent)
    agent._file_mutation_verifier_enabled_cache = False
    assert footer(agent) == 'Finished.'


def test_successful_batch_clears_failures_without_enrolling_stop_gates(agent, skill):
    args = {'operations': [{'name': 'sample', 'action': 'write_file', 'file_path': 'scripts/x.py'}]}
    record(agent, args)
    args['operations'][0]['file_content'] = 'x = 1\n'
    assert record(agent, args)['success'] is True
    assert footer(agent) == 'Finished.'
    assert not agent._turn_file_mutation_paths
    agent._checkpoint_mgr.record_agent_write.assert_not_called()


def test_lookup_exception_retains_named_failure(agent, skill, monkeypatch):
    def unavailable(*args):
        raise OSError('unavailable')
    monkeypatch.setattr(skills, '_find_skill', unavailable)
    record(agent, {'name': 'sample', 'action': 'patch'}, json.dumps({'error': 'cannot write'}))
    assert 'sample' in footer(agent)
    assert 'cannot write' in footer(agent)


def test_custom_create_directory_is_used(agent, tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    external = tmp_path / 'custom-skills'
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('skills:\n  create_dir: ' + external.as_posix() + '\n')
    args = {'name': 'fresh', 'action': 'create', 'category': 'testing'}
    record(agent, args, json.dumps({'error': 'invalid content'}))
    assert (external / 'testing/fresh/SKILL.md').as_posix() in footer(agent)


def test_profile_a_b_a_does_not_clear_another_profiles_failure(agent, tmp_path, monkeypatch):
    args = {'name': 'sample', 'action': 'patch'}
    for home in [tmp_path / 'a', tmp_path / 'b']:
        monkeypatch.setenv('HERMES_HOME', str(home))
        record(agent, args, json.dumps({'error': 'not found'}))
    assert len(agent._turn_failed_file_mutations) == 2
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'a'))
    record(agent, args, json.dumps({'success': True}))
    assert len(agent._turn_failed_file_mutations) == 1
    assert (tmp_path / 'b').as_posix() in footer(agent)


def test_empty_batch_error_is_not_silenced(agent, skill):
    record(agent, {'operations': []})
    assert 'skill_manage' in footer(agent)


@pytest.mark.parametrize('batch', [False, True])
def test_categorized_delete_retry_clears_failure_after_directory_disappears(agent, skill, batch):
    op = {'name': 'sample', 'action': 'delete', 'absorbed_into': 'missing-umbrella'}
    args = {'operations': [op]} if batch else op
    assert record(agent, args)['success'] is False
    assert 'sample' in footer(agent)
    op.pop('absorbed_into')
    assert record(agent, args)['success'] is True
    assert not skill.exists()
    assert footer(agent) == 'Finished.'


def test_malformed_result_does_not_erase_failure(agent, skill):
    args = {'name': 'sample', 'action': 'patch'}
    record(agent, args, json.dumps({'error': 'not found'}))
    agent._record_file_mutation_result('skill_manage', args, 'not JSON', False)
    assert 'skill_manage' in footer(agent)

