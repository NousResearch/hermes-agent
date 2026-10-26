"""Real file + SQLite receipts across cached turns and recreated agents."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import prompt_builder as pb
from agent.turn_context import build_turn_context
from run_agent import AIAgent
from agent.instruction_refresh import refresh_instructions
from agent.runtime_cwd import reset_session_cwd, set_session_cwd
from hermes_state import SessionDB


@pytest.fixture
def context(tmp_path):
    home = Path(os.environ['HERMES_HOME'])
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text('instruction_refresh:\n  enabled: true\n  paths: []\n')
    (home / 'SOUL.md').write_text('Initial operator instruction.')
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'AGENTS.md').write_text('Initial repository instruction.')
    token = set_session_cwd(str(project))
    db = SessionDB(home / 'state.db')
    db.create_session(session_id='existing', source='slack', model='test-model')
    db.set_session_title('existing', 'Existing task')
    db.set_session_title_source('existing', 'user')
    db.append_message('existing', 'user', 'Keep my existing task.')
    try:
        yield home, project, db
    finally:
        db.close()
        reset_session_cwd(token)


def make_agent(home, project, db):
    agent = SimpleNamespace(
        load_soul_identity=True, skip_context_files=False, platform='slack',
        context_compressor=None, _session_db=db, session_id='existing',
        _cached_system_prompt=db.get_session('existing')['system_prompt'] or 'LEGACY',
        build_count=0, pinned_tools=('terminal',),
    )

    def build(_system_message):
        agent.build_count += 1
        return '\n\n'.join([
            pb.load_soul_md(home_override=home) or '',
            (pb.build_context_files_prompt(cwd=str(project), skip_soul=True, home_override=home)
             if not agent.skip_context_files else ''),
        ])
    agent._build_system_prompt = build
    return agent


def test_changed_files_refresh_cached_and_sqlite_restored_agents(context):
    home, project, db = context
    agent = make_agent(home, project, db)
    assert refresh_instructions(agent)
    first = agent._cached_system_prompt
    assert not refresh_instructions(agent)
    assert agent._cached_system_prompt == first
    assert agent.build_count == 1

    # Editing during an active turn does not mutate the frozen request.
    (home / 'SOUL.md').write_text('New operator instruction.')
    assert agent._cached_system_prompt == first
    assert db.get_session('existing')['system_prompt'] == first
    # The next turn boundary adopts the change exactly once.
    assert refresh_instructions(agent)
    assert 'New operator instruction.' in agent._cached_system_prompt
    assert not refresh_instructions(agent)

    # A gateway eviction/restart must not restore the pre-change SQLite prompt.
    restored = make_agent(home, project, db)
    assert not refresh_instructions(restored)
    (project / 'AGENTS.md').write_text('New repository instruction.')
    assert refresh_instructions(restored)
    assert 'New repository instruction.' in restored._cached_system_prompt
    assert restored.pinned_tools == ('terminal',)
    assert db.get_session('existing')['message_count'] == 1
    assert db.get_session('existing')['system_prompt'] == restored._cached_system_prompt


def test_configured_sources_outside_cwd_and_failed_reads_do_not_advance_receipt(context):
    home, project, db = context
    shared = home / 'shared.md'
    shared.write_text('Standing shared instruction.')
    (home / 'config.yaml').write_text(json.dumps({'instruction_refresh': {'enabled': True, 'paths': [str(shared)]}}))
    agent = make_agent(home, project, db)
    assert refresh_instructions(agent)
    assert 'Standing shared instruction.' in agent._cached_system_prompt
    before = agent._cached_system_prompt
    shared.unlink()
    with pytest.raises(FileNotFoundError):
        refresh_instructions(agent)
    assert agent._cached_system_prompt == before
    assert db.get_session('existing')['system_prompt'] == before


def test_disabled_and_skipped_context_do_not_load_project_files(context):
    home, project, db = context
    agent = make_agent(home, project, db)
    (home / 'config.yaml').write_text('instruction_refresh:\n  enabled: false\n')
    assert not refresh_instructions(agent)
    assert agent._cached_system_prompt == 'LEGACY'
    (home / 'config.yaml').write_text('instruction_refresh:\n  enabled: true\n')
    agent.skip_context_files = True
    assert refresh_instructions(agent)
    before = agent._cached_system_prompt
    (project / 'AGENTS.md').write_text('Do not load this source.')
    assert not refresh_instructions(agent)
    assert agent._cached_system_prompt == before


def test_real_turn_setup_refreshes_cached_agent(context):
    home, project, db = context
    agent = AIAgent(api_key='test-key', base_url='https://openrouter.ai/api/v1',
                    model='test-model', quiet_mode=True, skip_memory=True,
                    session_db=db, session_id='existing')
    agent._session_db_created = True
    agent._cached_system_prompt = 'LEGACY'
    agent._skip_mcp_refresh = True
    # Use real file discovery/SQLite and the real turn phase; no model/provider calls.
    agent._build_system_prompt = make_agent(home, project, db)._build_system_prompt
    kwargs = dict(agent=agent, user_message='Continue', system_message=None,
                  conversation_history=None, task_id=None, stream_callback=None,
                  persist_user_message=None, restore_or_build_system_prompt=lambda *a: None,
                  install_safe_stdio=lambda: None, sanitize_surrogates=lambda s: s,
                  summarize_user_message_for_log=lambda s: s, set_session_context=lambda s: None,
                  set_current_write_origin=lambda s: None,
                  ra=lambda: SimpleNamespace(_set_interrupt=lambda *a, **k: None))
    with patch('agent.auxiliary_client.set_runtime_main', lambda *a, **k: None):
        build_turn_context(**kwargs)
        assert 'Initial operator instruction.' in agent._cached_system_prompt
        first = agent._cached_system_prompt
        build_turn_context(**kwargs)
        assert agent._cached_system_prompt == first
        (project / 'AGENTS.md').write_text('Changed between turns.')
        build_turn_context(**kwargs)
        assert 'Changed between turns.' in agent._cached_system_prompt


def test_normal_context_precedence_and_file_removal(context):
    home, project, db = context
    (project / '.hermes.md').write_text('Higher priority project instruction.')
    agent = make_agent(home, project, db)
    assert refresh_instructions(agent)
    before = agent._cached_system_prompt
    (project / 'AGENTS.md').write_text('This lower priority source is not selected.')
    assert not refresh_instructions(agent)
    assert agent._cached_system_prompt == before
    (project / '.hermes.md').unlink()
    assert refresh_instructions(agent)
    assert 'This lower priority source is not selected.' in agent._cached_system_prompt


def test_receipts_and_sources_stay_in_the_owning_profile(context, tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    home, project, db = context
    primary = make_agent(home, project, db)
    assert refresh_instructions(primary)
    primary_prompt = primary._cached_system_prompt
    other = tmp_path / 'other-profile'
    other.mkdir()
    (other / 'config.yaml').write_text('instruction_refresh:\n  enabled: true\n')
    (other / 'SOUL.md').write_text('Other profile instruction.')
    other_db = SessionDB(other / 'state.db')
    other_db.create_session(session_id='existing', source='slack', model='test-model')
    token = set_hermes_home_override(other)
    try:
        secondary = make_agent(other, project, other_db)
        assert refresh_instructions(secondary)
        assert 'Other profile instruction.' in secondary._cached_system_prompt
        assert 'Initial operator instruction.' not in secondary._cached_system_prompt
        assert db.get_session('existing')['system_prompt'] == primary_prompt
        assert not refresh_instructions(secondary)
    finally:
        reset_hermes_home_override(token)
        other_db.close()
