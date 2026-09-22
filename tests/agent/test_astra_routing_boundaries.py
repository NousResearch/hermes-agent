"""Additional vision, compression fallback, and native probe routing contracts."""
from types import SimpleNamespace as NS

import pytest

from hermes_cli.routing_policy import RoutingPolicyError


def deny(monkeypatch, **rules):
    from hermes_cli import routing_policy
    monkeypatch.setattr(routing_policy, 'current_routing_policy', lambda: {
        'enabled': True, 'deny': rules,
    })


def fail_policy(*args, **kwargs):
    raise RoutingPolicyError('policy denied')


@pytest.mark.parametrize('rule,value', [('providers', 'openrouter'), ('models', 'forbidden'), ('base_url_hosts', 'blocked.invalid')])
def test_vision_denial_precedes_discovery(monkeypatch, rule, value):
    from agent import auxiliary_client as aux
    deny(monkeypatch, **{rule: [value]})
    monkeypatch.setattr(aux, 'resolve_vision_provider_client', lambda **kw: pytest.fail('vision discovery'))
    with pytest.raises(RoutingPolicyError):
        aux._resolve_call_client('vision',
            provider='openrouter', model='forbidden', base_url='https://blocked.invalid/v1', api_key=None,
            resolved_provider='openrouter', resolved_model='forbidden', resolved_base_url='https://blocked.invalid/v1',
            resolved_api_key=None, resolved_api_mode=None, main_runtime=None, async_mode=False)


def test_aux_recovery_rejects_policy_before_classification(monkeypatch):
    from agent import auxiliary_client as aux
    monkeypatch.setattr(aux, '_ladder_parameter_rungs', lambda *a: pytest.fail('recovery entered'))
    with pytest.raises(RoutingPolicyError):
        next(aux._aux_recovery_ladder(RoutingPolicyError('denied'), client=None, kwargs={}, task=None,
            async_mode=False, base_info='', resolved_provider='', resolved_model=None, resolved_base_url=None,
            resolved_api_key=None, resolved_api_mode=None, final_model=None, max_tokens=None, main_runtime=None, route_info=None))


def test_compression_stall_fallback_policy_denial_propagates(monkeypatch):
    from agent import conversation_compression as c
    monkeypatch.setattr(c, 'run_compress_context_with_progress_timeout', fail_policy)
    with pytest.raises(RoutingPolicyError):
        c._run_pinned_compression_retry({'label': 'fallback', 'model': 'allowed'},
            worker=lambda fence: ([], ''), messages=[], system_prompt_fallback='', idle_timeout_seconds=1,
            total_ceiling_seconds=2, on_commit_overrun=None, on_timeout_cause=None,
            telemetry_agent=NS(), new_fence=c.CompressionCommitFence)


@pytest.mark.parametrize('wire_denial', [False, True])
def test_bedrock_context_probe_policy_is_terminal(monkeypatch, wire_denial):
    from agent import bedrock_adapter as b
    calls = []
    def send(**kw):
        calls.append(kw)
        fail_policy()
    client = NS(meta=NS(endpoint_url='https://blocked.invalid'), converse=send)
    monkeypatch.setattr(b, '_get_bedrock_runtime_client', lambda region: client)
    monkeypatch.setattr(b, '_BEDROCK_PROBE_TIERS', (10, 20))
    deny(monkeypatch, base_url_hosts=['blocked.invalid'] if wire_denial else [])
    with pytest.raises(RoutingPolicyError):
        b.probe_bedrock_context_length('allowed', 'us-east-1')
    assert len(calls) == (0 if wire_denial else 1)
