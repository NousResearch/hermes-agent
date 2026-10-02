"""Real registry dispatch: guards settle before request evidence observation."""
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('blocked', [False, True])
def test_dispatch_observer_follows_all_guards(monkeypatch, blocked):
    import model_tools
    from agent.tool_execution_observer import observe_tool_execution
    from tools.registry import registry
    events, calls = [], []
    name = 'request_evidence_fixture'
    registry.register(name=name, toolset='fixture', schema={'name': name, 'parameters': {'type': 'object'}},
                      handler=lambda args, **kwargs: calls.append(args) or json.dumps({'ok': True}))
    monkeypatch.setattr(model_tools, '_pre_dispatch_guards', lambda *args: (args[1],
        (json.dumps({'error': 'blocked'}), 'test_guard', 'blocked') if blocked else None))
    try:
        with observe_tool_execution(lambda **event: events.append(event)):
            result = model_tools.handle_function_call(name, {}, skip_tool_request_middleware=True,
                                                      skip_tool_execution_middleware=True)
        assert len(calls) == (0 if blocked else 1)
        assert [e['stage'] for e in events] == ([] if blocked else ['started', 'completed'])
        if not blocked:
            assert events[-1]['result'] == result
    finally:
        registry._tools.pop(name, None)


def test_execution_middleware_block_emits_no_evidence(monkeypatch):
    import model_tools
    from agent.tool_execution_observer import observe_tool_execution
    monkeypatch.setattr('hermes_cli.middleware.run_tool_execution_middleware', lambda *args, **kwargs: '{"error":"denied"}')
    events = []
    with observe_tool_execution(lambda **event: events.append(event)):
        model_tools.handle_function_call('read_file', {}, skip_pre_tool_call_hook=True,
                                        skip_tool_request_middleware=True)
    assert events == []


def test_observer_failure_does_not_change_tool_result():
    from agent.tool_execution_observer import observe_tool_execution, emit_tool_execution
    def failed(**kwargs):
        raise ValueError('plugin failure')
    with observe_tool_execution(failed):
        emit_tool_execution(stage='started', tool_name='fixture', args={})


def test_latest_detail_formats_failure_before_stream_seal(monkeypatch):
    from gateway.run_turn_runner import TurnRunner
    ctx = SimpleNamespace(session_key='session', run_generation=1, result_holder=[None])
    runner = SimpleNamespace()
    consumer = SimpleNamespace(finish=lambda *args: None)
    formatted = []
    monkeypatch.setattr('gateway.request_lifecycle.request_for_run', lambda *args: SimpleNamespace(state={}))
    monkeypatch.setattr('gateway.request_lifecycle.final_response',
        lambda *args, **kwargs: formatted.append(kwargs) or 'Known facts; provider failed; unknown checks.')
    TurnRunner(runner, ctx)._finish_stream_consumer({'failed': True, 'error': 'provider down',
        'final_response': 'diagnostic', 'messages': []}, [], consumer)
    assert formatted[0]['failure'] == 'provider'
    assert ctx.result_holder[0]['final_response'].startswith('Known facts')


@pytest.mark.asyncio
async def test_tool_completion_keeps_original_owner_after_successful_correction(tmp_path):
    from tests.gateway.test_request_lifecycle import runner, Transport, event
    from gateway.request_lifecycle import admit_request, begin_request, fold_request, tool_observer_for_run
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.plugins import get_plugin_manager, PluginContext
    from hermes_cli.plugins_manifest import PluginManifest
    token = set_hermes_home_override(tmp_path)
    manager = get_plugin_manager()
    manager._discovered = True
    ctx = PluginContext(PluginManifest(name='evidence-owner', version='1', path=tmp_path), manager)
    observed = []
    ctx.register_hook('gateway_request_tool', lambda request, stage, **kwargs: observed.append((request, stage)))
    try:
        obj = runner(Transport())
        original_event = event('Friday')
        original = await admit_request(obj, original_event, 'session')
        begin_request(obj, original_event, 4)
        old_callback = tool_observer_for_run(obj, 'session', 4)
        old_callback(stage='started', tool_name='fixture', args={})
        corrected_event = event('Actually Saturday', message='correction')
        corrected = await admit_request(obj, corrected_event, 'session')
        fold_request(obj, corrected_event, 'session')
        old_callback(stage='completed', tool_name='fixture', args={}, result={'ok': True})
        new_callback = tool_observer_for_run(obj, 'session', 4)
        new_callback(stage='started', tool_name='fixture', args={})
        new_callback(stage='completed', tool_name='fixture', args={}, result={'ok': True})
        assert observed == [(original, 'started'), (corrected, 'started'), (corrected, 'completed')]
        assert not original.active
    finally:
        reset_hermes_home_override(token)
