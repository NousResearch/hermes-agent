"""A chat whose own route key changed since it started is refused by design (a session never sends a
different key than it started with). The refusal must name the way out (/new), never /retry."""
import json
from types import SimpleNamespace


def test_turn_on_a_chat_whose_route_key_changed_names_new_chat_not_retry(tmp_path, monkeypatch):
    """The refusal at turn entry used to escape as an exception: every surface showed "Something went
    wrong ... Use /retry", which can never succeed, plus a false "some actions may have run" notice."""
    from agent.error_surface import build_error_surface_from_result
    from gateway import session_authorities as authorities, session_policy
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_results import execution_result
    from tests.gateway.test_config_policy_recovery import _bound_session

    cfg = {'model': {'provider': 'custom', 'api_key': 'route-token', 'base_url': 'http://127.0.0.1:1234/v1'}}
    policy, owner, config_path = _bound_session(tmp_path, monkeypatch, cfg)
    cfg['model']['api_key'] = 'rotated-token'
    config_path.write_text(json.dumps(cfg))
    turn = TurnRunner(SimpleNamespace(), SimpleNamespace(source=None, session_id='session', session_key=None))
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policy)
    monkeypatch.setattr(authorities, 'active_authority', lambda runner: owner())  # a restarted owner

    token = execution_result.set({})
    try:
        result = turn.run_sync()
        captured = execution_result.get()
    finally:
        execution_result.reset(token)

    assert result['failed'] and not result['completed'] and result['api_calls'] == 0, result
    assert result['failure_reason'] == 'launch_credentials_unavailable' and result['failure_retryable'] is False
    text = result['final_response']
    assert '/new' in text and 'model.api_key' in text, text
    assert 'Use /retry to try again' not in text and 'Some actions may already have run' not in text, text
    assert 'route-token' not in text and 'rotated-token' not in text
    assert captured['result'] == result  # what the classic CLI viewer reads off the receipt
    surface = build_error_surface_from_result(result)
    assert result['error_surface'] == surface  # committed as the transcript's refusal card on settlement
    assert (surface['layer'], surface['code'], surface['retryable']) == ('gateway', 'launch_credentials_unavailable', False)


def test_cli_refusal_of_a_control_on_that_chat_names_new_chat():
    """The classic CLI's /model (and /compress, /worktree) printed the bare reason code for the same
    refusal; any retry or model pick repeats it."""
    from hermes_cli.cli_chat_error_copy import refusal_text
    from hermes_cli.gateway_client import GatewayRPCError

    refused = GatewayRPCError('launch_credentials_unavailable', code=4001, data={
        'reason': 'launch_credentials_unavailable', 'field': 'model.api_key',
        'detail': 'changed in config.yaml since this session started'})
    text = refusal_text(refused)
    assert '/new' in text and 'model.api_key' in text, text
    assert refusal_text(GatewayRPCError('invalid_params', data={'field': 'toolsets', 'detail': 'bad'})) == \
        'invalid_params (toolsets: bad)'  # every other refusal reads as before
