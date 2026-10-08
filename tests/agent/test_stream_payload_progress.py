from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.chat_completion_helpers import _StreamingCall


def chunk(content=None,reasoning=None,tools=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content,reasoning_content=reasoning,tool_calls=tools),finish_reason=None)])


def call():
    agent=MagicMock();agent.provider='local';agent.model='fixture';agent.base_url='http://127.0.0.1';agent.api_mode='chat_completions'
    return _StreamingCall(agent,{'model':'fixture'},None)


def test_empty_control_chunks_do_not_reset_the_progress_deadline(monkeypatch):
    unit=call();unit.last_chunk_time['t']=10
    diag={'bytes':0,'chunks':0}
    monkeypatch.setattr('agent.chat_completion_helpers.time.time',lambda:20)
    unit._count_chunk(diag,chunk())
    unit._count_chunk(diag,SimpleNamespace(choices=[]))
    assert unit.last_chunk_time['t']==10
    assert diag['chunks']==2 and diag['payload_chunks']==0
    assert diag['last_transport_chunk_at']==20
    unit.agent._touch_activity.assert_not_called()


def test_text_reasoning_and_tool_arguments_are_real_progress(monkeypatch):
    unit=call();unit.last_chunk_time['t']=10
    monkeypatch.setattr('agent.chat_completion_helpers.time.time',lambda:20)
    for payload in (chunk(content='答案'),chunk(reasoning='thinking'),chunk(tools=[SimpleNamespace(function=SimpleNamespace(name='x',arguments='{}'))])):
        unit.last_chunk_time['t']=10;diag={'bytes':0,'chunks':0}
        unit._count_chunk(diag,payload)
        assert unit.last_chunk_time['t']==20 and diag['bytes']>0 and diag['payload_chunks']==1


def test_relay_acceptance_does_not_reintroduce_empty_chunk_progress(monkeypatch):
    unit=call();unit.last_chunk_time['t']=10
    monkeypatch.setattr(unit,'_stream_attempt_is_active',lambda _:True)
    monkeypatch.setattr(unit,'_writer_still_current',lambda _:True)
    monkeypatch.setattr('agent.chat_completion_helpers.time.time',lambda:20)
    assert unit._accept_chat_chunk(1,chunk())
    assert unit.last_chunk_time['t']==10
    assert unit._accept_chat_chunk(1,chunk(content='x'))
    assert unit.last_chunk_time['t']==20


def test_structured_reasoning_text_is_progress_but_opaque_replay_is_not(monkeypatch):
    unit = call()
    unit.last_chunk_time['t'] = 10
    monkeypatch.setattr('agent.chat_completion_helpers.time.time', lambda: 20)
    def detail_chunk(detail):
        delta = SimpleNamespace(content=None, reasoning_content=None, tool_calls=None,
                                model_extra={'reasoning_details': [detail]})
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)])
    unit._count_chunk({}, detail_chunk({'type': 'reasoning.encrypted', 'data': 'opaque'}))
    assert unit.last_chunk_time['t'] == 10
    unit._count_chunk({}, detail_chunk({'type': 'reasoning.text', 'text': 'thinking'}))
    assert unit.last_chunk_time['t'] == 20


def test_anthropic_thinking_delta_remains_real_progress(monkeypatch):
    unit = call()
    unit.last_chunk_time['t'] = 10
    monkeypatch.setattr('agent.chat_completion_helpers.time.time', lambda: 20)
    unit._count_chunk({}, SimpleNamespace(delta=SimpleNamespace(thinking='step 1')))
    assert unit.last_chunk_time['t'] == 20
