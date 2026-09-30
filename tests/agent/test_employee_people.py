from types import SimpleNamespace
import pytest
from agent.people import bind_turn, select_store
from agent.turn_context import compose_user_api_content, substitute_api_content
from tools.memory_tool_store import MemoryStore


def test_people_across_sessions_and_profiles_preserve_historical_bytes(tmp_path, monkeypatch):
    for profile in ('a', 'b', 'a'):
        monkeypatch.setenv('HERMES_HOME', str(tmp_path / profile))
        agent = SimpleNamespace(platform='telegram', session_id='dm', _memory_store=MemoryStore())
        bind_turn(agent, {'id': '123', 'name': 'Sasha'})
        store, target = select_store(agent, {})
        store.add(target, 'Prefers short answers.')
        bind_turn(agent, {'id': '123', 'name': 'Sasha'})
        before = compose_user_api_content('hello', 'recalled facts', 'plugin', agent._personal_context)
        assert before.index('hello') < before.index('<user-profile-context>') < before.index('<memory-context>') < before.index('plugin')
        agent.session_id = 'group'
        bind_turn(agent, {'id': '123', 'name': 'Sasha'})
        assert 'Prefers short answers.' in agent._personal_context
        first = agent._current_person
        bind_turn(agent, {'id': '456', 'name': 'Sasha'})
        assert agent._current_person != first
        assert 'Prefers short answers.' not in agent._personal_context
        with pytest.raises(ValueError, match='requires'):
            select_store(agent, {'target': 'user'})
        with pytest.raises(ValueError, match='Ambiguous'):
            select_store(agent, {'target': 'user', 'user': 'Sasha'})
        historical = {'role': 'user', 'content': 'hello', 'api_content': before}
        assert substitute_api_content(historical) == before


def test_review_freezes_participants_before_worker_starts(monkeypatch):
    from agent import background_review
    agent = SimpleNamespace(_person_participants={'Sasha #p1': ['p1']})
    captured = {}
    monkeypatch.setattr(background_review, '_run_review_in_thread', lambda *args, **kwargs: captured.update(kwargs))
    target, _ = background_review.spawn_background_review_thread(agent, [], task_cfg={})
    agent._person_participants['Sasha #p1'].append('p2')
    target()
    assert captured['person_snapshot'] == {'Sasha #p1': ['p1']}
    review = SimpleNamespace(_is_knowledge_review=True, _person_participants=captured['person_snapshot'],
                             _current_person=None, _personal_context='')
    bind_turn(review, {'id': '456', 'name': 'Sasha'})
    assert review._person_participants == {'Sasha #p1': ['p1']}
    assert review._current_person is None


def test_bot_cli_turn_cannot_inherit_owner(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr('hermes_cli.config.load_config_readonly', lambda: {'employee': {'owner': 'telegram:123'}})
    agent = SimpleNamespace(platform='cli', session_id='local', _memory_store=MemoryStore())
    bind_turn(agent, None)
    assert agent._current_person
    bind_turn(agent, {'id': 'another-agent', 'name': 'Bot', 'is_bot': True})
    assert agent._current_person is None and agent._personal_context == ''
    _, target = select_store(agent, {})
    assert target == 'memory'


def test_personal_memory_retry_budget_lasts_for_the_turn(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    agent = SimpleNamespace(platform='telegram', session_id='dm', _memory_store=MemoryStore())
    author = {'id': '123', 'name': 'Alex'}
    bind_turn(agent, author)
    for _ in range(MemoryStore._MAX_CONSOLIDATION_FAILURES_PER_TURN + 1):
        store, target = select_store(agent, {})
        result = store.replace(target, 'missing fact', 'new fact')
    assert result['done'] is True and not result['success']
    bind_turn(agent, author)
    store, target = select_store(agent, {})
    assert not store.replace(target, 'missing fact', 'new fact').get('done')


def test_participants_survive_compression_and_fresh_child_resume(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('parent', source='telegram')
        agent = SimpleNamespace(platform='telegram', session_id='parent', _memory_store=MemoryStore(), _session_db=db)
        bind_turn(agent, {'id': 'one', 'name': 'Alex'})
        first_label = agent._current_person_label
        bind_turn(agent, {'id': 'two', 'name': 'Sam'})
        db.create_session('child', source='telegram', parent_session_id='parent')
        resumed = SimpleNamespace(platform='telegram', session_id='child', _memory_store=MemoryStore(), _session_db=db)
        bind_turn(resumed, {'id': 'two', 'name': 'Sam'})
        assert first_label in resumed._person_participants
        with pytest.raises(ValueError, match='multiple participants'):
            select_store(resumed, {'target': 'user'})
        store, target = select_store(resumed, {'target': 'user', 'user': first_label})
        assert store.add(target, 'Owns sales.')['success']
        agent.session_id = 'child'
        bind_turn(agent, {'id': 'one', 'name': 'Alex'})
        assert 'Owns sales.' in agent._personal_context
    finally:
        db.close()


def test_memory_tool_routes_person_and_organization_writes(tmp_path, monkeypatch):
    import json
    from agent.inline_tool_executors import _memory, InlineToolContext

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    agent = SimpleNamespace(platform='telegram', session_id='group',
                            _memory_store=MemoryStore(), _memory_manager=None)
    agent._memory_store.load_from_disk()
    bind_turn(agent, {'id': '123', 'name': 'Alex'})
    label = agent._current_person_label
    bind_turn(agent, {'id': '456', 'name': 'Blair'})
    ctx = InlineToolContext(effective_task_id='memory-routing')
    personal = _memory(agent, {'action': 'add', 'target': 'user', 'user': label,
                              'content': 'Prefers concise replies.'}, ctx)
    shared = _memory(agent, {'action': 'add', 'target': 'memory',
                            'content': 'Organization uses EUR for billing.'}, ctx)
    assert json.loads(personal)['success']
    assert json.loads(shared)['success']
    bind_turn(agent, {'id': '123', 'name': 'Alex'})
    assert 'Prefers concise replies.' in agent._personal_context
    assert 'Organization uses EUR' not in agent._personal_context
    bind_turn(agent, {'id': '456', 'name': 'Blair'})
    assert 'Prefers concise replies.' not in agent._personal_context
    assert agent._memory_store._entries_for('memory') == ['Organization uses EUR for billing.']
