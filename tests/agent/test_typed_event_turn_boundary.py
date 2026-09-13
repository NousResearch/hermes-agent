from types import SimpleNamespace
from unittest.mock import Mock
from agent import conversation_loop


def test_typed_event_does_not_export_matching_historical_user_as_current(monkeypatch):
    text='same text as earlier user'
    typed=object()
    result={'messages':[{'role':'user','content':text},{'role':'assistant','content':'old'},{'role':'developer','content':text}]}
    inner=Mock(return_value=result)
    monkeypatch.setattr(conversation_loop,'_run_conversation_turn',inner)
    actual=conversation_loop.run_conversation(SimpleNamespace(),text,gateway_system_event=typed)
    assert actual is result
    assert 'current_turn_user_idx' not in actual
    assert inner.call_args.kwargs['gateway_system_event'] is typed
