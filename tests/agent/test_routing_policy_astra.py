"""Final-wire and terminal routing contracts from the Astra routing review."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from hermes_cli.routing_policy import RoutingPolicyError


@pytest.fixture
def policy(monkeypatch, tmp_path):
    # Exercise the real policy/config reader in the runner's isolated home.
    import yaml
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    value = {'enabled': True, 'deny': {'models': ['forbidden'],
             'providers': ['forbidden'], 'base_url_hosts': ['forbidden.invalid']}}
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'routing_policy': value}))
    return value


def denied(*a, **kw):
    raise RoutingPolicyError('forbidden policy request')


def client(base_url='https://forbidden.invalid/v1'):
    return NS(base_url=base_url, chat=NS(completions=NS(create=Mock(return_value=NS(choices=[])))))


@pytest.mark.parametrize('async_mode', [False, True])
@pytest.mark.parametrize('override', [False, True])
def test_anthropic_translated_wire_guard(policy, monkeypatch, async_mode, override):
    from agent import auxiliary_client as aux, anthropic_adapter
    wire = Mock(return_value=NS(content=[], usage=None, stop_reason='end_turn'))
    monkeypatch.setattr(anthropic_adapter, 'create_anthropic_message', wire)
    if not override:
        monkeypatch.setattr(anthropic_adapter, 'build_anthropic_kwargs', lambda **kw: {'model': 'forbidden', 'messages': []})
    wrapper = aux.AnthropicAuxiliaryClient(NS(base_url='https://allowed.invalid'), 'allowed', 'test', 'https://allowed.invalid')
    if async_mode:
        wrapper = aux.AsyncAnthropicAuxiliaryClient(wrapper)
    kwargs = dict(model='allowed', messages=[], extra_body={'model': 'forbidden'} if override else {})
    with pytest.raises(RoutingPolicyError):
        result = wrapper.chat.completions.create(**kwargs)
        if async_mode:
            asyncio.run(result)
    wire.assert_not_called()


@pytest.mark.parametrize('entry', ['provider', 'call'])
@pytest.mark.parametrize('field', ['provider', 'model', 'endpoint'])
def test_aux_denies_before_discovery(policy, monkeypatch, entry, field):
    from agent import auxiliary_client as aux
    provider = 'forbidden' if field == 'provider' else 'openrouter'
    model = 'forbidden' if field == 'model' else 'allowed'
    url = 'https://forbidden.invalid' if field == 'endpoint' else None
    probe = Mock(return_value=(None, None))
    monkeypatch.setattr(aux, '_try_openrouter', probe)
    monkeypatch.setattr(aux, '_get_cached_client', probe)
    with pytest.raises(RoutingPolicyError):
        if entry == 'provider':
            aux.resolve_provider_client(provider, model=model, explicit_base_url=url)
        else:
            aux._resolve_call_client(None, provider=provider, model=model, base_url=url, api_key=None,
                resolved_provider=provider, resolved_model=model, resolved_base_url=url,
                resolved_api_key=None, resolved_api_mode=None, main_runtime=None, async_mode=False)
    probe.assert_not_called()


def test_main_config_endpoint_denied_before_credentials(policy, monkeypatch):
    from hermes_cli import runtime_provider as rp
    monkeypatch.setattr(rp, '_get_model_config', lambda: {'provider': 'openrouter', 'default': 'allowed', 'base_url': 'https://forbidden.invalid'})
    probe = Mock(side_effect=RuntimeError('credential discovery reached'))
    monkeypatch.setattr(rp, 'resolve_provider', probe)
    with pytest.raises(RoutingPolicyError):
        rp.resolve_runtime_provider(requested='openrouter', target_model='allowed')
    probe.assert_not_called()


def test_nonstream_uses_actual_request_endpoint(policy):
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    wire = client()
    agent = NS(provider='openrouter', model='allowed', base_url='https://allowed.invalid', api_mode='chat_completions')
    with pytest.raises(RoutingPolicyError):
        _dispatch_nonstreaming_api_request(agent, {'model': 'allowed'}, make_client=lambda *a, **kw: wire)
    wire.chat.completions.create.assert_not_called()


@pytest.mark.parametrize('entry', ['direct', 'stream', 'helper', 'helper_stream'])
def test_bedrock_sdk_endpoint_override(policy, monkeypatch, entry):
    from agent import bedrock_adapter as bed, chat_completion_helpers as helpers
    # Real boto client: SDK endpoint overrides must be observed, not inferred from region.
    import boto3
    wire = boto3.client('bedrock-runtime', region_name='us-east-1', endpoint_url='https://forbidden.invalid',
                        aws_access_key_id='test', aws_secret_access_key='test')
    wire.converse = Mock(return_value={})
    wire.converse_stream = Mock(return_value={})
    monkeypatch.setattr(bed, '_get_bedrock_runtime_client', lambda region: wire)
    with pytest.raises(RoutingPolicyError):
        if entry.startswith('helper'):
            helpers._bedrock_converse_call({'modelId': 'allowed', '__bedrock_region__': 'us-east-1'}, stream=entry.endswith('stream'))
        else:
            fn = bed.call_converse_stream if entry == 'stream' else bed.call_converse
            fn(region='us-east-1', model='allowed', messages=[])
    wire.converse.assert_not_called()
    wire.converse_stream.assert_not_called()


@pytest.mark.parametrize('recovery,stream', [(kind, stream) for kind in ('cache', 'redacted', 'iam', 'policy')
                                          for stream in (False, True) if kind != 'iam' or stream])
def test_bedrock_recovery_is_guarded(policy, monkeypatch, stream, recovery):
    from agent import bedrock_adapter as bed
    wire = NS(meta=NS(endpoint_url='https://allowed.invalid'))
    calls = []
    def send(**kwargs):
        calls.append(kwargs)
        wire.meta.endpoint_url = 'https://forbidden.invalid'
        if recovery == 'policy':
            denied()
        raise RuntimeError('provider failure')
    wire.converse = wire.converse_stream = send
    monkeypatch.setattr(bed, '_get_bedrock_runtime_client', lambda region: wire)
    monkeypatch.setattr(bed, 'recover_from_cache_point_rejection', lambda exc, kw: dict(kw) if recovery in ('cache', 'policy') else None)
    monkeypatch.setattr(bed, 'recover_from_redacted_reasoning_rejection', lambda exc, kw: dict(kw) if recovery == 'redacted' else None)
    monkeypatch.setattr(bed, 'is_streaming_access_denied_error', lambda exc: recovery == 'iam')
    with pytest.raises(RoutingPolicyError):
        (bed.call_converse_stream if stream else bed.call_converse)(region='us-east-1', model='allowed', messages=[])
    assert len(calls) == 1


def test_existing_codex_thread_compaction_fails_closed(policy):
    from agent.conversation_compression import _compress_context_via_codex_app_server
    compact = Mock(return_value=NS(error='blocked'))
    agent = NS(_codex_session=NS(compact_thread=compact), session_id='test', _emit_status=Mock())
    with pytest.raises(RoutingPolicyError):
        _compress_context_via_codex_app_server(agent, [], '', force=True)
    compact.assert_not_called()


@pytest.mark.parametrize('async_mode', [False, True])
def test_stream_denial_cannot_retry_plain(policy, monkeypatch, async_mode):
    from agent import auxiliary_client as aux
    wire = client('https://allowed.invalid')
    calls = []
    def send(**kw):
        calls.append(kw)
        denied()
    async def asend(**kw):
        return send(**kw)
    wire.chat.completions.create = asend if async_mode else send
    monkeypatch.setattr(aux, '_aux_progress_active', lambda: True)
    with pytest.raises(RoutingPolicyError):
        if async_mode:
            asyncio.run(aux._acreate_with_progress(wire, {'model': 'allowed'}))
        else:
            aux._create_with_progress(wire, {'model': 'allowed'})
    assert len(calls) == 1


@pytest.mark.parametrize('entry', ['probe', 'revalidate', 'ensure'])
def test_feasibility_outer_denial_propagates(policy, monkeypatch, entry):
    from agent import auxiliary_client as aux, conversation_compression as comp
    monkeypatch.setattr(aux, 'get_text_auxiliary_client', denied)
    agent = NS(compression_enabled=True, context_compressor=NS(), _current_main_runtime=lambda: {})
    with pytest.raises(RoutingPolicyError):
        if entry == 'probe':
            comp.check_compression_model_feasibility(agent)
        elif entry == 'revalidate':
            comp.revalidate_compression_feasibility(agent)
        else:
            comp.ensure_compression_feasibility_checked(agent, 1000000)
    assert not getattr(agent, '_compression_feasibility_checked', False)


def test_review_worker_denial_propagates(policy, monkeypatch):
    from agent import background_review as review
    monkeypatch.setattr(review, '_run_review_fork', denied)
    monkeypatch.setattr(review, '_set_thread_approval_callback', lambda cb: None)
    monkeypatch.setattr(review, '_parent_can_emit_tool_calls', lambda agent: True)
    with pytest.raises(RoutingPolicyError):
        review._run_review_in_thread(NS(_emit_auxiliary_failure=Mock()), [], 'review')


def test_main_fallback_denial_stops_chain(policy, monkeypatch):
    from agent import chat_completion_helpers as helpers
    agent = NS(_fallback_index=0, _fallback_chain=[{'provider': 'openrouter', 'model': 'allowed'},
              {'provider': 'openrouter', 'model': 'second'}])
    monkeypatch.setattr('agent.fallback_cooldown.switch_deferred_by_reset', lambda *a: False)
    monkeypatch.setattr('agent.fallback_cooldown._arm_rate_limit_cooldown', lambda *a, **kw: 0)
    monkeypatch.setattr(helpers, '_should_skip_fallback_candidate', lambda *a: False)
    monkeypatch.setattr(helpers, '_fallback_api_mode_hint', lambda *a: (False, 'chat_completions'))
    monkeypatch.setattr('hermes_cli.fallback_config.resolve_entry_api_key', lambda entry: None)
    resolve = Mock(side_effect=RoutingPolicyError('denied fallback'))
    monkeypatch.setattr('agent.auxiliary_client.resolve_provider_client', resolve)
    with pytest.raises(RoutingPolicyError):
        helpers.try_activate_fallback(agent)
    assert resolve.call_count == 1
    assert agent._fallback_index == 1


@pytest.mark.parametrize('async_mode', [False, True])
def test_fallback_parameter_recovery_never_retries_policy_error(policy, monkeypatch, async_mode):
    from agent import auxiliary_client as aux, auxiliary_fallback_recovery as recovery
    calls = []
    # Policy exception text is not a transport classification signal.
    def send(c, kw):
        calls.append(kw)
        raise RoutingPolicyError('unsupported_parameter temperature')
    async def asend(c, kw):
        return send(c, kw)
    with pytest.raises(RoutingPolicyError):
        if async_mode:
            asyncio.run(recovery.send_with_parameter_rungs_async(asend, client(), {'model': 'allowed', 'temperature': 1}, task=None))
        else:
            recovery.send_with_parameter_rungs(send, client(), {'model': 'allowed', 'temperature': 1}, task=None)
    assert len(calls) == 1


def test_summary_uses_actual_client_endpoint(policy, monkeypatch):
    from agent import chat_completion_helpers as helpers
    wire = client()
    agent = NS(provider='openrouter', model='allowed', base_url='https://allowed.invalid',
               _build_api_kwargs=lambda msgs: {'model': 'allowed', 'messages': msgs},
               _ensure_primary_openai_client=lambda **kw: wire)
    monkeypatch.setattr(helpers, 'sanitize_outbound_kwargs', lambda *a: None)
    monkeypatch.setattr(helpers, '_summary_text', lambda *a: '')
    with pytest.raises(RoutingPolicyError):
        helpers._chat_summary_attempt(agent, [], 'test')(0)
    wire.chat.completions.create.assert_not_called()


def test_aux_main_model_fallback_denial_propagates(policy, monkeypatch):
    from agent import auxiliary_client as aux
    monkeypatch.setattr(aux, '_read_main_provider', lambda: 'openrouter')
    monkeypatch.setattr(aux, '_read_main_model', lambda: 'allowed')
    monkeypatch.setattr(aux, '_custom_health_base_url', lambda *a: '')
    monkeypatch.setattr(aux, '_is_provider_unhealthy', lambda *a: False)
    monkeypatch.setattr(aux, 'resolve_provider_client', denied)
    with pytest.raises(RoutingPolicyError):
        aux._try_main_agent_model_fallback('different')


@pytest.mark.parametrize('async_mode', [False, True])
def test_anthropic_guard_after_sanitization(policy, monkeypatch, async_mode):
    from agent import auxiliary_client as aux, anthropic_adapter as ant
    sent = Mock(return_value=NS(content=[], usage=None, stop_reason='end_turn'))
    sdk = NS(base_url='https://allowed.invalid', messages=NS(create=sent))
    monkeypatch.setattr(ant, 'sanitize_anthropic_kwargs', lambda kw, **opts: kw.update(model='forbidden'))
    wrapper = aux.AnthropicAuxiliaryClient(sdk, 'allowed', 'test', sdk.base_url)
    if async_mode:
        wrapper = aux.AsyncAnthropicAuxiliaryClient(wrapper)
    with pytest.raises(RoutingPolicyError):
        result = wrapper.chat.completions.create(model='allowed', messages=[])
        if async_mode:
            asyncio.run(result)
    sent.assert_not_called()


def test_anthropic_stream_policy_error_is_terminal(policy):
    from agent.anthropic_adapter import create_anthropic_message
    send = Mock(return_value=NS())
    sdk = NS(messages=NS(create=send, stream=Mock(side_effect=RoutingPolicyError('stream not supported by routing policy'))))
    with pytest.raises(RoutingPolicyError):
        create_anthropic_message(sdk, {'model': 'allowed', 'messages': []})
    send.assert_not_called()


@pytest.mark.parametrize('malformed', ['enabled', ['enabled']])
def test_current_policy_rejects_malformed_config(monkeypatch, malformed):
    from hermes_cli import config, routing_policy

    monkeypatch.setattr(config, 'read_raw_config_readonly', lambda: {'routing_policy': malformed})
    with pytest.raises(RoutingPolicyError, match='policy must be a mapping'):
        routing_policy.current_routing_policy()


@pytest.mark.parametrize('malformed', [None, {'enabled': True, 'deny': None}])
def test_current_policy_rejects_null_values_before_default_merge(monkeypatch, malformed):
    from hermes_cli import config, routing_policy

    monkeypatch.setattr(config, 'read_raw_config_readonly', lambda: {'routing_policy': malformed})
    with pytest.raises(RoutingPolicyError):
        routing_policy.current_routing_policy()


def test_current_policy_wraps_malformed_host_url(monkeypatch):
    from hermes_cli import config, routing_policy

    monkeypatch.setattr(config, 'read_raw_config_readonly', lambda: {
        'routing_policy': {'enabled': True, 'deny': {'base_url_hosts': ['https://[broken']}}
    })
    with pytest.raises(RoutingPolicyError, match='base_url_hosts'):
        routing_policy.current_routing_policy()
