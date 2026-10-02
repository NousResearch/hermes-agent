"""Provider call IDs are local to their session/request, not global hook identities."""
import threading

import pytest

from hermes_cli.plugins import PluginManager


@pytest.mark.parametrize('changed', ['session_id', 'turn_id', 'api_request_id'])
def test_same_provider_call_id_in_distinct_contexts_runs_both_guards(monkeypatch, changed):
    monkeypatch.setattr('hermes_cli.plugins._resolve_hook_callback_timeout', lambda: 10.0)
    first_started, second_started, release = threading.Event(), threading.Event(), threading.Event()
    starts, results = [], []
    lock = threading.Lock()
    def guard(**kwargs):
        with lock:
            starts.append(kwargs)
            (first_started if len(starts) == 1 else second_started).set()
        release.wait(8)
        return {'action': 'allow'}
    manager = PluginManager()
    manager._hooks['pre_tool_call'] = [guard]
    original = {'session_id': 'session', 'turn_id': 'turn', 'api_request_id': 'request', 'tool_call_id': 'owned'}
    def fire(context):
        results.append(manager.invoke_hook('pre_tool_call', tool_name='terminal', **context))
    first = threading.Thread(target=fire, args=(original,), daemon=True)
    second = threading.Thread(target=fire, args=({**original, changed: 'other'},), daemon=True)
    try:
        first.start()
        assert first_started.wait(3)
        second.start()
        assert second_started.wait(2), 'a different session/request was misclassified as a running duplicate'
    finally:
        release.set()
        first.join(3)
        if second.ident is not None: second.join(3)
    assert len(starts) == 2
    assert results == [[{'action': 'allow'}], [{'action': 'allow'}]]


@pytest.mark.parametrize('context', [
    {'session_id': 'session', 'turn_id': 'turn', 'api_request_id': 'request', 'tool_call_id': 'owned'},
    {},
])
def test_same_context_and_no_identity_still_refuse_duplicate_running_guard(monkeypatch, context):
    monkeypatch.setattr('hermes_cli.plugins._resolve_hook_callback_timeout', lambda: 10.0)
    started, release = threading.Event(), threading.Event()
    calls = []
    def guard(**kwargs):
        calls.append(kwargs)
        started.set()
        release.wait(8)
    manager = PluginManager()
    manager._hooks['pre_tool_call'] = [guard]
    first = threading.Thread(target=lambda: manager.invoke_hook('pre_tool_call', **context), daemon=True)
    try:
        first.start()
        assert started.wait(3)
        denied, = manager.invoke_hook('pre_tool_call', **context)
        assert denied['action'] == 'block' and 'still running' in denied['message']
        assert len(calls) == 1
    finally:
        release.set(); first.join(3)
    assert manager.invoke_hook('pre_tool_call', **context) == []
    assert len(calls) == 2
