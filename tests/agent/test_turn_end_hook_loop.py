"""External evidence-check consumer through native discovery and actual loop."""
import json
from unittest.mock import Mock
import pytest

from tests.agent.test_text_turn_incremental_persistence import loop_agent  # noqa: F401
from tests.agent.test_run_agent import _mock_response
from hermes_cli import plugins


@pytest.mark.parametrize('transform', [False, True])
def test_external_hook_reworks_without_persisting_rejected_candidate(loop_agent, tmp_path, monkeypatch, transform):
    from hermes_state import SessionDB
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    package = tmp_path / 'plugins/evidence-example'
    package.mkdir(parents=True)
    (package / 'plugin.yaml').write_text('name: evidence-example\nversion: 1.0.0\n')
    (package / '__init__.py').write_text('''from pathlib import Path
calls = []
def register(ctx):
    def check(final_response, turn_id, already_blocked, can_continue):
        calls.append((final_response, turn_id))
        if not already_blocked and can_continue and not Path(ctx.get_config('target')).exists():
            return {'action': 'block', 'message': 'Produce the missing evidence file.'}
    ctx.register_hook('before_turn_end', check)
''')
    file = tmp_path / 'result.txt'
    (tmp_path / 'config.yaml').write_text(json.dumps({'plugins': {'enabled': ['evidence-example'],
        'entries': {'evidence-example': {'settings': {'target': str(file)}}}}}))
    manager = plugins.PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins, 'get_plugin_manager', lambda: manager)
    assert manager.has_hook('before_turn_end')
    plugin = manager._plugins['evidence-example'].module
    suffix = ' (transformed)' if transform else ''
    if transform:
        manager._hooks['transform_llm_output'] = [lambda response_text: response_text + suffix]
    a = loop_agent
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session(a.session_id, source='cli', model=a.model)
    a._session_db = db
    a._intent_ack_continuation = False
    a._stall_guards = False
    a.stream_delta_callback = Mock()
    a.interim_assistant_callback = Mock()
    turns = []
    def create(**kw):
        turns.append(kw)
        if len(turns) == 1:
            a._fire_stream_delta('file ready, unverified')
            return _mock_response(content='file ready, unverified', finish_reason='stop')
        file.write_text('real artefact')
        return _mock_response(content='file ready, corrected', finish_reason='stop')
    a.client.chat.completions.create.side_effect = create
    try:
        result = a.run_conversation('Create the result file', task_id='completion-proof')
        assert result['final_response'] == 'file ready, corrected' + suffix
        assert [c[0] for c in plugin.calls] == ['file ready, unverified' + suffix, 'file ready, corrected' + suffix]
        assert plugin.calls[0][1] and plugin.calls[0][1] == plugin.calls[1][1]
        assert len(turns) == 2
        persisted = db.get_messages(a.session_id)
        assert any(m.get('content') == 'file ready, corrected' + suffix for m in persisted)
        assert not any('unverified' in (m.get('content') or '') for m in persisted)
        assert not any('Produce the missing evidence' in (m.get('content') or '') for m in persisted)
        a.stream_delta_callback.assert_not_called()
        a.interim_assistant_callback.assert_not_called()
    finally:
        a._session_db = None
        db.close()
