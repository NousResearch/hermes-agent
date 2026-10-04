from types import SimpleNamespace
import pytest
from gateway.config import PlatformConfig, Platform
from gateway.platforms.base import resolve_channel_project
from gateway.run_turn_runner import TurnRunner
from hermes_cli import projects_db
from agent.runtime_cwd import scoped_session_cwd, set_session_cwd, reset_session_cwd


def test_typed_from_dict_overrides():
    config = PlatformConfig.from_dict({'channel_overrides': {'20': {'project': 'exact'}, 'chat': {'project': 'parent'}}})
    assert 'channel_overrides' not in config.extra
    assert resolve_channel_project(config, '20', 'chat') == 'exact'
    assert resolve_channel_project(config, 'other', 'chat') == 'parent'


@pytest.mark.parametrize('reverse', [True, False])
def test_exact_topic_before_default(reverse):
    topics = [{'name': 'general'}, {'thread_id': 20, 'project': 'exact'}]
    if reverse: topics.reverse()
    config = PlatformConfig.from_dict({'group_topics': [{'chat_id': 'chat', 'project': 'default', 'topics': topics}]})
    assert resolve_channel_project(config, '20', 'chat') == 'exact'
    assert resolve_channel_project(config, 'missing', 'chat') == 'default'


def test_legacy_and_cross_chat():
    config = PlatformConfig.from_dict({'group_topics': {'chat': [{'thread_id': 20, 'project': 'legacy'}], 'other': [{'thread_id': 20, 'project': 'wrong'}]}})
    assert resolve_channel_project(config, '20', 'chat') == 'legacy'
    assert resolve_channel_project(config, '20', 'unknown') is None


@pytest.mark.parametrize('raises', [False, True])
def test_production_turn_binds_real_project_and_restores(tmp_path, monkeypatch, raises):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    with projects_db.connect_closing() as conn:
        projects_db.create_project(conn, name='Work', slug='work', primary_path=str(workspace))
    turn = TurnRunner.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id='chat', thread_id='20'))
    turn._runner = SimpleNamespace(config=SimpleNamespace(platforms={Platform.TELEGRAM: PlatformConfig.from_dict({'channel_overrides': {'20': {'project': 'work'}}})}))
    def body(self):
        assert scoped_session_cwd() == str(workspace)
        if raises: raise RuntimeError('test')
        return 'ran'
    monkeypatch.setattr(TurnRunner, '_run_sync_with_project', body)
    token = set_session_cwd('before')
    try:
        if raises:
            with pytest.raises(RuntimeError): turn.run_sync()
        else:
            assert turn.run_sync() == 'ran'
        assert scoped_session_cwd() == 'before'
    finally:
        reset_session_cwd(token)
