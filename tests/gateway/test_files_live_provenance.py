"""Normal SessionStore/cache allocator/TurnRunner path and negative provenance.

Real temporary SessionDB; only the SDK/client construction is inert. No gateway
listener, executor, native process or session coordinator is started.
"""
import copy
import json
import threading
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from agent.session_persistence import files_user_message_persistence
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionStore
from gateway.turn_context import TurnContext
from hermes_state import SessionDB
from tests.agent.files_persistence_fixtures import inert_agent

SAFE = 'same caption\n\n[Attached file: "same.txt"]'


@pytest.fixture
def live(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / 'owned.db')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db = db
    agent, sent, _ = inert_agent(monkeypatch, db, 'owned')
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._session_db = SimpleNamespace(_db=db)
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = OrderedDict()
    runner._refresh_fallback_model = lambda: None
    runner._apply_fallback_chain_to_agent = lambda *a: None
    runner._consume_pending_native_image_paths = lambda key: []
    source = SimpleNamespace(user_id='user', user_id_alt=None, user_name='User',
                             chat_id='owned', platform='api_server')
    ctx = TurnContext(source=source, session_id='owned', session_key='owned-key',
        history=[], user_config={}, enabled_toolsets=[], message='followup', inbound_message_id='next')
    turn = TurnRunner(runner, ctx)
    # Inert route policy lookup only; the cache allocator itself is real.
    monkeypatch.setattr(turn, '_skip_context_files', lambda platform: True)
    route = {'model': agent.model, 'runtime': {}}
    sig = runner._agent_config_signature(agent.model, {}, [], '',
        cache_keys=runner._extract_cache_busting_config(ctx.user_config),
        user_id='user', user_id_alt=None, skip_context_files=True)
    def admit(payload, admission):
        with files_user_message_persistence(agent, SAFE, admission_id=admission) as transcript:
            result = agent.run_conversation(payload, conversation_history=agent._session_messages,
                persist_user_message=transcript, persist_user_platform_id=admission)
        count = db.get_session('owned')['message_count']
        runner._agent_cache['owned-key'] = (agent, sig, count, 'owned')
        return result
    yield SimpleNamespace(agent=agent, sent=sent, db=db, store=store, runner=runner,
        ctx=ctx, turn=turn, route=route, admit=admit, path=tmp_path)
    db.close()


def test_normal_loader_allocator_and_two_identical_files_admissions(live, record_property):
    live.admit('/private/first.txt', 'admission-one')
    # Live memory on a second Files admission is not result-message aliasing.
    live.admit('/private/second.txt', 'admission-two')
    previous = copy.deepcopy(live.sent[-1])
    for number in range(3):
        live.ctx.history = live.store.load_transcript('owned')
        agent, reused = live.turn._resolve_turn_agent(live.route, 'api_server', '', 3, None, {})
        assert reused and agent is live.agent
        history, observed, _ = live.turn._load_turn_history(agent, reused)
        assert len(history) == len(agent._session_messages)
        result = live.turn._run_conversation_with_approval(agent, history, observed, None, None)
        assert live.sent[-1][:len(previous)] == previous
        assert '/private/' not in json.dumps(result)
        previous = copy.deepcopy(live.sent[-1])
        live.runner._agent_cache['owned-key'] = (agent, live.runner._agent_cache['owned-key'][1],
            live.db.get_session('owned')['message_count'], 'owned')
        assert len(agent._files_live_entries) == 2
        assert len({id(e.row) for e in agent._files_live_entries}) == 2
    users = [row['content'] for row in previous if row['role'] == 'user']
    assert users[:2] == ['/private/first.txt', '/private/second.txt']
    record_property('normal_cache_reuse', json.dumps({'provider': previous,
        'sql': live.db.get_messages('owned'), 'safe_result': result}, default=str))


@pytest.mark.parametrize('fault', ['missing_id', 'forged_row', 'duplicate_source', 'duplicate_sql',
    'inactive_original', 'compacted_clone', 'wrong_db', 'wrong_session', 'read_failure',
    'changed_content', 'mirror', 'lost_history'])
def test_gateway_provenance_failure_never_expands(live, monkeypatch, fault):
    live.admit('/private/first.txt', 'admission-one')
    live.ctx.history = live.store.load_transcript('owned')
    row_id = live.db.get_messages('owned')[0]['id']
    other = None
    if fault == 'missing_id':
        live.ctx.history[0].pop('message_id')
    elif fault == 'forged_row':
        live.ctx.history[0]['_row_id'] = row_id + 100
    elif fault == 'duplicate_source':
        live.ctx.history.extend(copy.deepcopy(live.ctx.history))
    elif fault == 'duplicate_sql':
        live.db.append_message('owned', 'user', content=SAFE, platform_message_id='admission-one')
    elif fault == 'inactive_original':
        live.db._conn.execute('UPDATE messages SET active=0 WHERE id=?', (row_id,))
        live.db._conn.commit()
    elif fault == 'compacted_clone':
        rows = live.db.get_messages('owned')
        live.db.archive_and_compact('owned', rows)
    elif fault == 'wrong_db':
        other = SessionDB(live.path / 'other.db')
        other.create_session(session_id='owned', source='api_server')
        other.append_message('owned', 'user', content=SAFE, platform_message_id='admission-one')
        assert other.get_messages('owned')[0]['id'] == row_id
        live.agent._session_db = other
    elif fault == 'wrong_session':
        live.agent.session_id = 'different-physical-session'
    elif fault == 'read_failure':
        monkeypatch.setattr(live.db, 'get_messages', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('read failed')))
    elif fault == 'changed_content':
        live.ctx.history[0]['content'] += ' rewritten'
    elif fault == 'mirror':
        live.ctx.history[0]['mirror'] = True
    elif fault == 'lost_history':
        live.ctx.history = []
    try:
        history, _, _ = live.turn._load_turn_history(live.agent, True)
        assert not live.agent._files_live_entries
        from agent.turn_context import build_api_messages
        wire, _ = build_api_messages(live.agent, history, current_turn_user_idx=None,
            ext_prefetch_cache='', plugin_user_context='', moa_config=None, active_system_prompt='system')
        assert '/private/first.txt' not in json.dumps(wire)
    finally:
        if other:
            other.close()


@pytest.mark.parametrize('empty', [False, True])
def test_explicit_api_history_cannot_authorize_overlay(live, empty):
    from gateway.session_api_turn import api_execution
    live.admit('/private/first.txt', 'admission-one')
    supplied = [] if empty else live.store.load_transcript('owned')
    token = api_execution.set({'history': supplied})
    try:
        history, _, _ = live.turn._load_turn_history(live.agent, True)
        assert history is supplied
        assert not live.agent._files_live_entries
    finally:
        api_execution.reset(token)


@pytest.mark.parametrize('fault', ['inactive_original', 'read_failure'])
def test_longer_unpersisted_selection_still_requires_live_sql_provenance(live, monkeypatch, fault):
    live.admit('/private/first.txt', 'admission-one')
    live.ctx.history = live.store.load_transcript('owned')
    live.agent._session_messages.append({'role': 'user', 'content': 'unpersisted ordinary'})
    if fault == 'inactive_original':
        live.db._conn.execute("UPDATE messages SET active=0 WHERE role='user'")
        live.db._conn.commit()
    else:
        monkeypatch.setattr(live.db, 'get_messages', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('read failed')))
    history, _, _ = live.turn._load_turn_history(live.agent, True)
    assert history[-1]['content'] == 'unpersisted ordinary'
    assert not live.agent._files_live_entries
