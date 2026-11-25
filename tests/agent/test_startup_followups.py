"""Startup follow-ups reach the first request without rewriting earlier turns (#154296)."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.startup_followups import StartupFollowups
from tests.agent.test_run_agent import _mock_response
from tests.agent.test_steer import TestSteerSurvivesRedirectRebuild as _SteerFixture


def test_first_request_contains_a_ticketed_startup_followup(monkeypatch):
    agent = _SteerFixture()._loop_agent()
    agent.session_id = "startup-test"
    writer = MagicMock(return_value=True)
    agent._session_db = SimpleNamespace(set_message_api_content=writer)
    monkeypatch.setattr("agent.agent_init_session.agent_interface", lambda _agent: None)
    delivered, requests = [], []
    startup = StartupFollowups(lambda entries: delivered.extend(item.ticket for item in entries))
    assert startup.offer("followup-1", "also include item B", None)
    agent._gateway_startup_followups = startup

    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "old", "type": "function", "function": {"name": "probe", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "old", "content": "old result"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    before = deepcopy(history)

    def persist(messages, *_args):
        for row in reversed(messages):
            if row.get("role") == "user":
                row["_row_id"] = 101
                break

    def model_call(kwargs):
        requests.append(deepcopy(kwargs["messages"]))
        return _mock_response(content="done", finish_reason="stop")

    agent._interruptible_api_call = model_call
    with (
        patch.object(agent, "_ensure_db_session"),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_persist_session", side_effect=persist),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("work on item A", conversation_history=history)

    assert result["completed"] is True
    assert len(requests) == 1
    assert requests[0][-1]["role"] == "user"
    assert "work on item A" in requests[0][-1]["content"]
    assert "also include item B" in requests[0][-1]["content"]
    assert history == before
    assert "also include item B" in writer.call_args.args[-1]
    assert delivered == ["followup-1"]
    assert startup.withdraw("followup-1") is False


def test_codex_responses_first_request_takes_the_followup(monkeypatch):
    from tests.agent.test_run_agent_codex_responses import _build_agent, _codex_message_response

    agent = _build_agent(monkeypatch)
    assert agent.api_mode == 'codex_responses'
    agent.session_id = 'startup-codex'
    agent._session_db = SimpleNamespace(set_message_api_content=MagicMock(return_value=True))
    monkeypatch.setattr('agent.agent_init_session.agent_interface', lambda _agent: None)
    startup = StartupFollowups(lambda _entries: None)
    assert startup.offer('b', 'also include item B', None)
    agent._gateway_startup_followups = startup
    requests = []

    def persist(messages, *_args):
        messages[-1]['_row_id'] = 101

    def model(kwargs):
        requests.append(deepcopy(kwargs['input']))
        return _codex_message_response('done')

    agent._interruptible_api_call = model
    with (patch.object(agent, '_ensure_db_session'), patch.object(agent, '_flush_messages_to_session_db'),
          patch.object(agent, '_persist_session', side_effect=persist), patch.object(agent, '_save_trajectory'),
          patch.object(agent, '_cleanup_task_resources')):
        result = agent.run_conversation('work on item A')
    assert result['completed'] is True
    assert len(requests) == 1
    assert 'work on item A' in str(requests[0])
    assert 'also include item B' in str(requests[0])
    assert startup.withdraw('b') is False


@pytest.mark.parametrize('caching', [False, True])
def test_anthropic_first_request_takes_the_followup(monkeypatch, caching):
    from tests.agent.test_anthropic_mid_tool_call_drop import _make_anthropic_agent

    agent = _make_anthropic_agent()
    if caching:
        agent.provider = 'anthropic'
        agent.base_url = 'https://api.anthropic.com'
        agent._use_prompt_caching = agent._use_native_cache_layout = True
        agent._cache_ttl = '5m'
        agent._cache_ttl_auto = False
    agent.session_id = 'startup-anthropic'
    agent._session_db = SimpleNamespace(set_message_api_content=MagicMock(return_value=True))
    monkeypatch.setattr('agent.agent_init_session.agent_interface', lambda _agent: None)
    monkeypatch.setattr('agent.turn_api_call._should_stream', lambda _agent: False)
    startup = StartupFollowups(lambda _entries: None)
    assert startup.offer('b', 'also include item B', None)
    agent._gateway_startup_followups = startup
    requests = []

    def persist(messages, *_args):
        messages[-1]['_row_id'] = 101

    def model(kwargs):
        requests.append(deepcopy(kwargs['messages']))
        return SimpleNamespace(content=[SimpleNamespace(type='text', text='done')], stop_reason='end_turn',
                               model='synthetic', usage=SimpleNamespace(input_tokens=10, output_tokens=1))

    agent._interruptible_api_call = model
    with (patch.object(agent, '_ensure_db_session'), patch.object(agent, '_flush_messages_to_session_db'),
          patch.object(agent, '_persist_session', side_effect=persist), patch.object(agent, '_save_trajectory'),
          patch.object(agent, '_cleanup_task_resources')):
        result = agent.run_conversation('work on item A')
    assert result['completed'] is True
    assert len(requests) == 1
    assert 'work on item A' in str(requests[0])
    assert 'also include item B' in str(requests[0])
    assert startup.withdraw('b') is False


def test_staged_input_returns_to_its_queue_if_no_request_starts():
    startup = StartupFollowups(lambda _entries: None)
    assert startup.offer("b", "follow-up", None)
    row = {"role": "user", "content": "first", "_row_id": 1}
    wire = {"role": "user", "content": "first"}
    assert startup.stage([row], [wire], 0)
    assert "follow-up" in wire["content"]
    startup.finish()
    assert wire["content"] == "first"
    assert "api_content" not in row
    assert startup.withdraw("b") is True


def test_stop_after_staging_prevents_the_stale_request():
    startup = StartupFollowups(lambda _entries: pytest.fail('stopped input must not be sent'))
    assert startup.offer('b', 'follow-up', None)
    row = {'role': 'user', 'content': 'first', '_row_id': 1}
    wire = {'role': 'user', 'content': 'first'}
    assert startup.stage([row], [wire], 0)
    assert startup.cancel_unread() == ('b',)
    assert wire['content'] == 'first'
    with pytest.raises(InterruptedError, match='stopped'):
        startup.commit_request(SimpleNamespace(), {'messages': [wire]})


@pytest.mark.parametrize('stop', [False, True])
def test_generation_ends_during_transcript_write_without_sending(monkeypatch, stop):
    monkeypatch.setattr("agent.agent_init_session.agent_interface", lambda _agent: None)
    current = [True]
    startup = StartupFollowups(lambda _entries: pytest.fail('no request was sent'), lambda: current[0])
    assert startup.offer('b', 'follow-up', None)
    row = {'role': 'user', 'content': 'first', '_row_id': 1}
    wire = {'role': 'user', 'content': 'first'}
    assert startup.stage([row], [wire], 0)
    calls = []

    def write(session, row_id, old, new_api, *, new_content):
        calls.append((old, new_api, new_content))
        if len(calls) == 1:
            if stop:
                assert startup.cancel_unread() == ()  # the write has already started
            current[0] = False
        return True

    agent = SimpleNamespace(session_id='s', _session_db=SimpleNamespace(set_message_api_content=write))
    with pytest.raises(InterruptedError, match='generation ended'):
        startup.commit_request(agent, {'messages': [wire]})
    assert calls == [('first', 'first\n\nfollow-up', 'first\n\nfollow-up'),
                     ('first\n\nfollow-up', 'first', 'first')]
    assert row['content'] == 'first' and wire['content'] == 'first'
    assert startup.withdraw('b') is (not stop)


def test_large_followup_stays_queued_when_it_would_cross_the_preflight_budget():
    startup = StartupFollowups(lambda _entries: None)
    assert startup.offer('b', 'more text than the remaining budget', None)
    row = {'role': 'user', 'content': 'first', '_row_id': 1}
    wire = {'role': 'user', 'content': 'first'}
    assert not startup.stage([row], [wire], 0, max_extra_tokens=1)
    assert wire['content'] == 'first'
    assert startup.withdraw('b') is True


def test_followup_does_not_rewrite_a_wire_tail_merged_with_older_history():
    startup = StartupFollowups(lambda _entries: None)
    assert startup.offer('b', 'follow-up', None)
    previous = {'role': 'user', 'content': 'previous'}
    current = {'role': 'user', 'content': 'first', '_row_id': 1}
    wire = {'role': 'user', 'content': 'previous\n\nfirst'}
    assert not startup.stage([previous, current], [wire], 1)
    assert wire['content'] == 'previous\n\nfirst'
    assert startup.withdraw('b') is True


def test_text_block_keeps_its_cache_marker_and_restores_on_no_request():
    startup = StartupFollowups(lambda _entries: None)
    assert startup.offer('b', 'follow-up', None)
    row = {'role': 'user', 'content': 'first', '_row_id': 1}
    wire = {'role': 'user', 'content': [
        {'type': 'text', 'text': 'first', 'cache_control': {'type': 'ephemeral'}}]}
    assert startup.stage([row], [wire], 0)
    assert wire['content'] == [
        {'type': 'text', 'text': 'first\n\nfollow-up', 'cache_control': {'type': 'ephemeral'}}]
    startup.finish()
    assert wire['content'] == [{'type': 'text', 'text': 'first', 'cache_control': {'type': 'ephemeral'}}]
    assert startup.withdraw('b') is True


def test_non_atomic_worker_store_leaves_followup_queued(monkeypatch):
    agent = _SteerFixture()._loop_agent()
    agent.session_id = 'startup-worker'

    def legacy_writer(_session, _row, _content, _api_content):
        return True

    agent._session_db = SimpleNamespace(set_message_api_content=legacy_writer)
    monkeypatch.setattr('agent.agent_init_session.agent_interface', lambda _agent: None)
    startup = StartupFollowups(lambda _entries: pytest.fail('follow-up must stay queued'))
    assert startup.offer('b', 'follow-up', None)
    agent._gateway_startup_followups = startup
    requests = []

    def persist(messages, *_args):
        messages[-1]['_row_id'] = 101

    def model(kwargs):
        requests.append(deepcopy(kwargs['messages']))
        return _mock_response(content='done', finish_reason='stop')

    agent._interruptible_api_call = model
    with (patch.object(agent, '_ensure_db_session'), patch.object(agent, '_flush_messages_to_session_db'),
          patch.object(agent, '_persist_session', side_effect=persist), patch.object(agent, '_save_trajectory'),
          patch.object(agent, '_cleanup_task_resources')):
        assert agent.run_conversation('first')['completed'] is True
    assert requests[0][-1]['content'] == 'first'
    assert startup.withdraw('b') is True


def test_first_request_is_replayed_from_real_persistence(tmp_path, monkeypatch):
    from persistence_assembly import open_persistence, profile_scope
    from run_agent import AIAgent
    from tests.fixtures.storage_read import conversation

    monkeypatch.setattr('agent.title_generator.maybe_auto_title', lambda *args, **kwargs: None)
    requests = []
    db = open_persistence(profile_scope(home=tmp_path))
    try:
        agent = AIAgent(
            api_key='synthetic', base_url='http://127.0.0.1:1/v1', provider='openai-compat',
            model='synthetic-model', api_mode='chat_completions', enabled_toolsets=[],
            quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
            session_db=db, persistence=db, session_id='startup-real-db',
        )
        agent._cached_system_prompt = 'Synthetic system prompt.'
        agent._disable_streaming = True
        startup = StartupFollowups(lambda _entries: None)
        assert startup.offer('b', 'also include item B', None)
        agent._gateway_startup_followups = startup

        def model(kwargs):
            requests.append(deepcopy(kwargs['messages']))
            return _mock_response(content='done', finish_reason='stop')

        agent._interruptible_api_call = model
        first = agent.run_conversation('work on item A', conversation_history=[])
        assert first['completed'] is True
        stored = conversation(db, 'startup-real-db')
        assert 'also include item B' in stored[0]['content']
        assert 'also include item B' in stored[0]['api_content']
        agent._gateway_startup_followups = None
        second = agent.run_conversation('next', conversation_history=stored)
        assert second['completed'] is True
        assert requests[1][:len(requests[0])] == requests[0]
        assert startup.withdraw('b') is False
    finally:
        db.close()


def test_anthropic_cache_text_is_replayed_from_real_persistence(tmp_path, monkeypatch):
    from persistence_assembly import open_persistence, profile_scope
    from run_agent import AIAgent
    from tests.fixtures.storage_read import conversation

    monkeypatch.setattr('agent.title_generator.maybe_auto_title', lambda *args, **kwargs: None)
    monkeypatch.setattr('agent.turn_api_call._should_stream', lambda _agent: False)
    requests = []
    db = open_persistence(profile_scope(home=tmp_path))
    try:
        agent = AIAgent(
            api_key='synthetic', base_url='https://api.anthropic.com', provider='anthropic',
            model='claude-sonnet-4-6', api_mode='anthropic_messages', enabled_toolsets=[],
            quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
            session_db=db, persistence=db, session_id='startup-anthropic-db',
        )
        agent._cached_system_prompt = 'Synthetic system prompt.'
        startup = StartupFollowups(lambda _entries: None)
        assert startup.offer('b', 'also include item B', None)
        agent._gateway_startup_followups = startup

        def model(kwargs):
            requests.append(deepcopy(kwargs['messages']))
            return SimpleNamespace(content=[SimpleNamespace(type='text', text='done')], stop_reason='end_turn',
                                   model='synthetic', usage=SimpleNamespace(input_tokens=10, output_tokens=1))

        agent._interruptible_api_call = model
        assert agent.run_conversation('work on item A', conversation_history=[])['completed'] is True
        stored = conversation(db, 'startup-anthropic-db')
        assert 'also include item B' in stored[0]['content']
        assert 'also include item B' in stored[0]['api_content']
        agent._gateway_startup_followups = None
        assert agent.run_conversation('next', conversation_history=stored)['completed'] is True

        def text_of(message):
            content = message['content']
            return content if isinstance(content, str) else ''.join(part['text'] for part in content)

        assert text_of(requests[1][0]) == text_of(requests[0][0])
        assert startup.withdraw('b') is False
    finally:
        db.close()
