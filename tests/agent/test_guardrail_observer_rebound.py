"""Independent boundary tests; uses real conversation/guard/consumer paths."""
import json

import pytest
from unittest.mock import patch

from agent.tool_guardrails import ToolCallSignature

from tests.agent.test_tool_call_guardrail_runtime import (
    _hard_stop_config, _make_agent, _mock_response, _mock_tool_call,
    _run_guardrail_conversation, _search_round,
)


def _skill_round(name="synthetic-qa-skill"):
    return _mock_response(content='', finish_reason='tool_calls', tool_calls=[
        _mock_tool_call('skill_view', json.dumps({'name': name}))])


@pytest.mark.parametrize('sequential', [True, False])
def test_block_must_not_mask_sibling_identical_streak_halt(sequential):
    agent = _make_agent('web_search', 'skill_view', config=_hard_stop_config(hard_stop_after={
        'exact_failure': 1, 'same_tool_failure': 99, 'idempotent_no_progress': 3}))
    # Fail search once, then read a non-mutating skill twice. The third successful
    # identical skill read reaches the existing terminal observer threshold.
    calls = [_mock_tool_call('web_search', json.dumps({'query': 'same'})),
             _mock_tool_call('skill_view', json.dumps({'name': 'synthetic-qa-skill'}))]
    batch = _mock_response(content='', finish_reason='tool_calls', tool_calls=calls)
    def handler(name, args, _task_id, **kwargs):
        return json.dumps({'error': 'synthetic failure'}) if name == 'web_search' else 'Synthetic stable skill body'
    execute = agent._execute_tool_calls_sequential if sequential else agent._execute_tool_calls
    with patch.object(agent, '_execute_tool_calls', side_effect=execute):
        result, dispatch = _run_guardrail_conversation(agent, [
            _search_round('same'), _skill_round(), _skill_round(), batch,
            _mock_response('MUST NOT RUN AFTER TERMINAL STREAK'),
        ], handler)
    observed = {'sequential': sequential, 'exit_reason': result['turn_exit_reason'],
                'final_response': result['final_response'], 'guardrail': result.get('guardrail'),
                'rebound_used': agent._tool_guardrail_rebound_used,
                'api_calls': agent.client.chat.completions.create.call_count,
                'dispatch_calls': dispatch.call_count,
                'controller_streak': agent._tool_guardrails._identical_streak_count}
    print('QA_OBSERVATION=' + json.dumps(observed))
    assert result['turn_exit_reason'] == 'guardrail_halt', observed
    assert result['guardrail']['code'] == 'identical_call_streak_halt', observed
    assert agent._tool_guardrail_rebound_used is False, observed
    assert agent.client.chat.completions.create.call_count == 4, observed


@pytest.mark.parametrize('sequential', [True, False])
def test_identical_streak_control_without_prior_block(sequential):
    agent = _make_agent('skill_view', config=_hard_stop_config(hard_stop_after={
        'exact_failure': 1, 'same_tool_failure': 99, 'idempotent_no_progress': 3}))
    execute = agent._execute_tool_calls_sequential if sequential else agent._execute_tool_calls
    with patch.object(agent, '_execute_tool_calls', side_effect=execute):
        result, dispatch = _run_guardrail_conversation(agent, [
            _skill_round(), _skill_round(), _skill_round(), _mock_response('MUST NOT RUN'),
        ], lambda *a, **kw: 'Synthetic stable skill body')
    assert result['turn_exit_reason'] == 'guardrail_halt'
    assert result['guardrail']['code'] == 'identical_call_streak_halt'
    assert agent._tool_guardrail_rebound_used is False
    assert agent.client.chat.completions.create.call_count == 3
    assert dispatch.call_count == 3


def test_real_rebound_retains_progress_reset_semantics():
    agent = _make_agent('web_search', 'patch', config=_hard_stop_config(hard_stop_after={
        'exact_failure': 2, 'same_tool_failure': 99, 'idempotent_no_progress': 99}))
    edit = _mock_response(content='', finish_reason='tool_calls', tool_calls=[
        _mock_tool_call('patch', json.dumps({'path': 'synthetic.py', 'old_string': 'a', 'new_string': 'b'}))])
    def handler(name, *a, **kw):
        return json.dumps({'success': True, 'diff': 'synthetic change'}) if name == 'patch' else json.dumps({'error': 'synthetic failure'})
    result, dispatch = _run_guardrail_conversation(agent, [
        _search_round('same'), _search_round('same'), _search_round('same'),
        edit, _search_round('same'), _mock_response('progress retry allowed'),
    ], handler)
    assert result['final_response'] == 'progress retry allowed'
    assert dispatch.call_count == 4
    assert agent._tool_guardrail_rebound_used is True
    assert 'guardrail' not in result


def test_rate_limit_retries_reach_terminal_halt_without_rebound():
    tool = 'mcp_slack_history'
    agent = _make_agent(tool, config=_hard_stop_config(hard_stop_after={
        'exact_failure': 1, 'same_tool_failure': 3, 'idempotent_no_progress': 99}))
    def round_():
        return _mock_response(content='', finish_reason='tool_calls', tool_calls=[
            _mock_tool_call(tool, json.dumps({'channel': 'synthetic-channel'}))])
    result, dispatch = _run_guardrail_conversation(agent, [
        round_(), round_(), round_(), _mock_response('MUST NOT RUN'),
    ], lambda *a, **kw: json.dumps({'error': 'Slack API error: ratelimited. Retry after 1 seconds.'}))
    assert result['turn_exit_reason'] == 'guardrail_halt'
    assert result['guardrail']['code'] == 'same_tool_failure_halt'
    assert agent._tool_guardrail_rebound_used is False
    assert agent.client.chat.completions.create.call_count == 3
    assert dispatch.call_count == 3


def _cycle_rounds():
    return [_skill_round(name) for name in ("alpha", "beta", "alpha", "beta", "alpha")]


def _observer_handler(name, _args, _task_id, **_kwargs):
    return json.dumps({"error": "synthetic failure"}) if name == "web_search" else "Synthetic stable skill body"


@pytest.mark.parametrize("sequential", [True, False])
def test_block_must_not_mask_sibling_cycle_halt(sequential):
    agent = _make_agent("web_search", "skill_view", config=_hard_stop_config(hard_stop_after={
        "exact_failure": 1, "same_tool_failure": 99, "idempotent_no_progress": 3}))
    batch = _mock_response(content="", finish_reason="tool_calls", tool_calls=[
        _mock_tool_call("web_search", json.dumps({"query": "same"})),
        _mock_tool_call("skill_view", json.dumps({"name": "beta"})),
    ])
    execute = agent._execute_tool_calls_sequential if sequential else agent._execute_tool_calls
    with patch.object(agent, "_execute_tool_calls", side_effect=execute):
        result, dispatch = _run_guardrail_conversation(agent, [
            _search_round("same"), *_cycle_rounds(), batch,
            _mock_response("MUST NOT RUN AFTER TERMINAL CYCLE"),
        ], _observer_handler)

    assert result["turn_exit_reason"] == "guardrail_halt"
    assert result["guardrail"]["code"] == "identical_cycle_halt"
    assert result["guardrail"]["count"] == 3
    assert agent._tool_guardrail_rebound_used is False
    assert agent.client.chat.completions.create.call_count == 7
    assert dispatch.call_count == 7
    assert len(agent._tool_guardrails._call_history) == 7
    assert agent._tool_guardrails._turn_web_search_count == 2
    assert agent._tool_guardrails._same_tool_failure_counts["web_search"] == 1


@pytest.mark.parametrize("sequential", [True, False])
@pytest.mark.parametrize("cycle", [True, False])
def test_first_observer_terminal_survives_later_sibling_block(sequential, cycle):
    agent = _make_agent("web_search", "skill_view", config=_hard_stop_config(hard_stop_after={
        "exact_failure": 1, "same_tool_failure": 99, "idempotent_no_progress": 3}))
    # The sequential path produces a terminal first, then refused siblings. The
    # planner may preflight the refusal first; both must retain that first halt.
    prior = _cycle_rounds() if cycle else [_skill_round("alpha"), _skill_round("alpha")]
    first_name, last_name = ("beta", "alpha") if cycle else ("alpha", "alpha")
    batch = _mock_response(content="", finish_reason="tool_calls", tool_calls=[
        _mock_tool_call("skill_view", json.dumps({"name": first_name})),
        _mock_tool_call("web_search", json.dumps({"query": "same"})),
        _mock_tool_call("skill_view", json.dumps({"name": last_name})),
    ])
    execute = agent._execute_tool_calls_sequential if sequential else agent._execute_tool_calls
    with patch.object(agent, "_execute_tool_calls", side_effect=execute):
        result, dispatch = _run_guardrail_conversation(agent, [
            _search_round("same"), *prior, batch, _mock_response("MUST NOT RUN"),
        ], _observer_handler)

    expected = "identical_cycle_halt" if cycle else "identical_call_streak_halt"
    assert result["turn_exit_reason"] == "guardrail_halt"
    assert result["guardrail"]["code"] == expected
    terminal = agent._tool_guardrails.halt_decision
    assert terminal is agent._tool_guardrail_halt_decision
    assert terminal.action == "halt"
    assert terminal.code == expected
    assert terminal.count == 3
    assert terminal.signature == ToolCallSignature.from_call("skill_view", {"name": first_name})
    assert agent._tool_guardrail_rebound_used is False
    assert agent.client.chat.completions.create.call_count == len(prior) + 2
    assert dispatch.call_count == len(prior) + 2  # The last read is itself no-progress-blocked.
    assert agent._tool_guardrails._turn_web_search_count == 2
    assert agent._tool_guardrails._same_tool_failure_counts["web_search"] == 1
