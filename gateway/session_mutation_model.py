"""Resolve explicit model changes using the existing source model switch pipeline."""
import asyncio
import json
from dataclasses import asdict, replace

from hermes_state_runtime import RuntimeStoreError


async def prepare_model(authority, live, payload, prepared):
    from gateway.session_policy import restore_policy, launch_key, bind_launch_key
    from gateway.session_policy_credentials import recover_config_secrets
    from hermes_cli.model_switch import apply_model_selection, selection_route_changed, switch_model
    from hermes_cli.config import get_compatible_custom_providers
    old = restore_policy(prepared['snapshot']['receipt']['policy'])
    config = old.config(authority)
    _, runtime = authority.runner._resolve_session_agent_runtime(source=live.source, session_key=live.route)

    def switch(**kwargs):
        # The alias table switch_model resolved against, read in the same call (process-global).
        result = switch_model(**kwargs)
        return result, _alias_credential(result)
    result, alias_credential = await asyncio.to_thread(switch, raw_input=payload['model'],
        current_provider=old.provider, current_model=old.model, current_base_url=old.base_url or '',
        current_api_key=runtime.get('api_key') or '', explicit_provider=payload.get('provider', ''),
        is_global=False, user_providers=config.get('providers'),
        custom_providers=get_compatible_custom_providers(config))
    if not result.success:
        raise RuntimeStoreError('model_resolution_failed')
    refusal = await _selection_guard_refusal(authority, live, payload, prepared['snapshot'], old, result, runtime)
    if refusal is not None:
        return refusal
    frozen = old.config()
    # The one selection shape config.yaml gets (#25106): api_mode follows the target, a context pin
    # and the endpoint-bound credential fields (inline key, key_env/api_key_env) drop on a route change.
    # Off-loop: the context-pin check can do cold-start disk I/O.
    frozen['model'] = await asyncio.to_thread(apply_model_selection, frozen.get('model'), result)
    # A direct alias's own credential is part of the selection (never ``result.api_key``, which the
    # switch resolved for its probe): its env pointer is committed and re-read every turn; a literal
    # key becomes a private config secret, a null in the durable receipt like build_policy's.
    key_env, alias_key = alias_credential
    if key_env:
        frozen['model']['key_env'] = key_env
    if alias_key:
        frozen['model']['api_key'] = None
    # Creation identity (request_json) remains immutable; runtime selection lives in the policy.
    policy = replace(old, model=result.new_model, config_json=json.dumps(frozen))
    secrets = recover_config_secrets(authority, old) if old.config_secret_ref else {}
    # A cleared field's private value must not be re-hydrated into the new route by config().
    secrets = {path: value for path, value in secrets.items()
               if not (len(path) == 2 and path[0] == 'model' and path[1] not in frozen['model'])}
    if alias_key:
        secrets[('model', 'api_key')] = alias_key
    # Re-fingerprint frozen config references because policy identity includes model.
    # An explicit launch key belongs to the endpoint it authenticated, not to a provider name:
    # custom -> custom on another base_url must not carry it (same route identity as config keys).
    key = launch_key(authority, old)
    current = {'provider': old.provider, 'base_url': old.base_url or runtime.get('base_url') or ''}
    if selection_route_changed(current, result):
        policy = replace(policy, credential_ref=None)
        key = None
    policy = bind_launch_key(authority, prepared['snapshot']['receipt']['session_id'], policy, key,
                             config_secrets=secrets)
    return dict(prepared, policy=asdict(policy))


def _alias_credential(result):
    """``(key_env, literal_key)`` the direct alias a switch resolved declares for its own endpoint
    (``api_key: "${VAR}"`` is the pointer ``VAR``); ``(None, None)`` when it declares none."""
    from hermes_cli.model_switch import DIRECT_ALIASES
    alias = DIRECT_ALIASES.get(result.resolved_via_alias or '') if result.success else None
    raw = (alias.api_key or '').strip() if alias is not None else ''
    if raw.startswith('${') and raw.endswith('}'):
        return raw[2:-1].strip() or None, None
    return ((alias.key_env or '').strip() or None) if alias is not None and not raw else None, raw or None


def _persisted_selection_context(authority, snapshot, old):
    from agent.usage_anchor import persisted_anchor_tokens
    from hermes_cli.model_selection_guards import SelectionContext
    target = snapshot.get('target') or snapshot['receipt']['session_id']
    tokens = persisted_anchor_tokens(authority.db, target, authority.db.get_messages_as_conversation(target))
    return SelectionContext(context_tokens=tokens, current_model=old.model or None) if tokens else None


async def _selection_guard_refusal(authority, live, payload, snapshot, old, result, runtime):
    """The shared selection-guard registry (cost, data-policy, large-context switch) runs before the
    commit. A flagged target is refused with ``status: confirmation_required`` and a ``confirm``
    token bound to the resolved target, the selection it replaces and the session revision; nothing
    is written. Only that exact token in ``payload['confirm']`` applies it: a stale or foreign
    token (target re-resolved elsewhere, another switch landed, a turn settled) gets a fresh
    refusal, never a silent apply."""
    import hashlib
    from hermes_cli.model_selection_guards import combined_message, selection_context_for_agent, selection_warnings
    from hermes_cli.route_identity import normalize_route_base_url
    lookup = (getattr(authority.runner, '_resident_agent_for', None)
              or getattr(authority.runner, '_cached_agent_for', None))
    agent = lookup(live.route) if live is not None and callable(lookup) else None
    context = selection_context_for_agent(agent)
    if context is None:
        # No resident agent (evicted by the previous switch, or an owner restart): the session's
        # persisted usage anchor still measures the context at stake, so the large-context guard
        # does not go quiet on exactly the switch that follows another one.
        context = await asyncio.to_thread(_persisted_selection_context, authority, snapshot, old)
    # Off-loop: pricing lookups may hit models.dev on a cache miss.
    warnings = await asyncio.to_thread(
        selection_warnings, result.new_model, provider=result.target_provider,
        base_url=result.base_url or old.base_url or '', api_key=result.api_key or runtime.get('api_key') or '',
        model_info=result.model_info, selection_context=context)
    if not warnings:
        return None
    binding = {'target': [result.new_model, result.target_provider, normalize_route_base_url(result.base_url),
                          result.api_mode or ''],
               'current': [old.model, old.provider or '', normalize_route_base_url(old.base_url)],
               'session': [snapshot['target'], snapshot['target_revision']],
               'guards': sorted(w.kind for w in warnings)}
    token = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    if payload.get('confirm') == token:
        return None
    # A fresh dict: the prepared snapshot carries the frozen policy and never leaves the owner. No
    # ``model`` key: a client that predates the exchange must not paint the target as applied.
    return {'status': 'confirmation_required', 'confirm_required': True, 'confirm': token,
            'target_model': result.new_model, 'target_provider': result.target_provider,
            'confirm_message': combined_message(warnings),
            'warnings': [{'kind': w.kind, 'title': w.title, 'message': w.message} for w in warnings]}
