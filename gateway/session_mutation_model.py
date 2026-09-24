"""Resolve explicit model changes using the existing source model switch pipeline."""
import asyncio
import json
from dataclasses import asdict, replace

from hermes_state_runtime import RuntimeStoreError


async def prepare_model(authority, live, payload, prepared):
    from gateway.session_policy import restore_policy, launch_key, bind_launch_key
    from gateway.session_policy_credentials import recover_config_secrets
    from hermes_cli.model_switch import switch_model
    from hermes_cli.config import get_compatible_custom_providers
    old = restore_policy(prepared['snapshot']['receipt']['policy'])
    config = old.config(authority)
    _, runtime = authority.runner._resolve_session_agent_runtime(source=live.source, session_key=live.route)
    result = await asyncio.to_thread(switch_model, raw_input=payload['model'],
        current_provider=old.provider, current_model=old.model, current_base_url=old.base_url or '',
        current_api_key=runtime.get('api_key') or '', explicit_provider=payload.get('provider', ''),
        is_global=False, user_providers=config.get('providers'),
        custom_providers=get_compatible_custom_providers(config))
    if not result.success:
        raise RuntimeStoreError('model_resolution_failed')
    frozen = old.config()
    frozen.setdefault('model', {}).update(default=result.new_model, provider=result.target_provider,
                                         base_url=result.base_url)
    # Creation identity remains immutable; runtime selection lives in the policy.
    policy = replace(old, model=result.new_model, config_json=json.dumps(frozen))
    secrets = recover_config_secrets(authority, old) if old.config_secret_ref else {}
    # Re-fingerprint frozen config references because policy identity includes model.
    # An explicit launch key cannot silently cross into a different provider.
    key = launch_key(authority, old)
    if result.provider_changed:
        policy = replace(policy, credential_ref=None)
        key = None
    policy = bind_launch_key(authority, prepared['snapshot']['receipt']['session_id'], policy, key,
                             config_secrets=secrets)
    return dict(prepared, policy=asdict(policy))
