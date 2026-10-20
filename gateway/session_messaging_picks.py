"""A viewer's model and reasoning picks on a messaging chat (Telegram, Discord, ...), #141152.

Desktop and the TUI send ``session.mutate`` ``model`` / ``reasoning`` for whatever session they show.
A LOCAL session commits the pick into its frozen policy; a messaging chat has no frozen policy, so
that path refused it with ``runtime_coordination_required`` and the reasoning pill waited forever for
a level. The chat's own ``/model`` and ``/reasoning`` set a per-chat override that its next turn
resolves (``_record_model_switch``, ``_apply_reasoning_selection``). A viewer's pick sets that same
override, so the chat and every viewer see one setting.

The mutation names the session a viewer resumed, which for a compressed chat is its lineage root
(``messaging_owner_ref``): the fences are checked on that row and the override is keyed by the lane's
route, so it applies to the compression tip the next turn writes.
"""
import asyncio

from hermes_state_runtime import RuntimeStoreError

CHAT_PICKS = frozenset({'model', 'reasoning'})


def is_chat_pick(live, operation):
    from gateway.config import Platform
    return operation in CHAT_PICKS and live is not None and live.source.platform != Platform.LOCAL


def _require_fences(authority, ref, params):
    row = authority.db.get_session(ref.session_id)
    if row is None:
        raise RuntimeStoreError('not_found')
    if row['runtime_revision'] != params['expected_revision']:
        raise RuntimeStoreError('revision_conflict')
    if row['runtime_generation'] != params['expected_generation']:
        raise RuntimeStoreError('stale_generation')
    return row


def _platform_key(live):
    from gateway.run import _platform_config_key
    return _platform_config_key(live.source.platform)


def _apply_reasoning(runner, live, effort):
    """``/reasoning <level>`` (``None``: ``/reasoning reset``) as the chat would run it."""
    return runner._apply_reasoning_selection(live.route, _platform_key(live), 'reset' if effort is None else effort)


async def _pick_model(authority, owner, live, payload, row):
    """The chat's ``/model <name> [--provider p]`` resolution, guarded like a LOCAL pick, then committed
    as its session override. Returns the receipt fields, or the guard's confirmation request."""
    from gateway.session_mutation_model import _Route, _selection_guard_refusal
    from gateway.slash_commands_model import _ModelSwitchContext
    from hermes_constants import get_hermes_home
    runner = authority.runner
    ctx = _ModelSwitchContext(session_key=live.route, source=live.source,
                              config_path=get_hermes_home() / 'config.yaml', persist_global=False)
    ctx.read_config()
    ctx.apply_override(runner._session_model_override(live.route) or {})
    result, error = await runner._perform_model_switch(ctx, payload['model'], payload.get('provider', ''), live.source)
    if error is not None:
        raise RuntimeStoreError('model_resolution_failed', detail=error)
    snapshot = {'target': owner.session_id, 'target_revision': row['runtime_revision']}
    current = _Route(ctx.current_provider, ctx.current_model, ctx.current_base_url, ctx.current_api_key)
    refusal = await _selection_guard_refusal(authority, live, payload, snapshot, current, result)
    if refusal is not None:
        return refusal
    # No in-place swap of the cached agent: a turn may be running on it. The override is what the
    # next turn builds from, and the record step evicts the cached agent.
    await runner._record_model_switch(result, ctx, source=live.source, one_turn=False, picker=True)
    fields = {'model': result.new_model, 'provider': result.target_provider}
    if 'reasoning' in payload:
        _apply_reasoning(runner, live, payload['reasoning'])
        fields['effort'] = payload['reasoning']
    return fields


async def apply_chat_pick(authority, ref, owner, live, params):
    """Apply a ``model`` / ``reasoning`` mutation to the messaging chat *live* mounted on *owner* (the
    caller authorized control and holds its mutation lock). The fences are the mount's, the ones
    ``session.resume`` handed the viewer; comparing them with a compression segment's row refused a
    pick on a compressed chat with ``stale_generation``."""
    from hermes_state_mutations import validate_action
    operation, payload = params['operation'], params['payload']
    validate_action(operation, payload)
    row = await asyncio.to_thread(_require_fences, authority, owner, params)
    runner = authority.runner
    if operation == 'reasoning':
        await asyncio.to_thread(_apply_reasoning, runner, live, payload['effort'])
        fields = {'effort': payload['effort']}
    else:
        async with runner._model_switch_lock():
            fields = await _pick_model(authority, owner, live, payload, row)
    result = {'session_id': ref.session_id, 'operation': operation, 'revision': row['runtime_revision'],
              'execution_generation': row['runtime_generation'], **fields}
    if fields.get('status') == 'confirmation_required':
        return {'session_id': ref.session_id, 'operation': operation, **fields}
    from gateway.session_local import local_session_info
    # Every viewer repaints its model and reasoning pill, as after a LOCAL pick.
    live.event_stream.publish(owner.session_id, local_session_info(authority, owner), event_type='session.info')
    return result


def messaging_route_info(runner, live):
    """``model`` and the effort fields a messaging chat's next turn resolves, without building an agent
    or resolving credentials: session ``/model`` > ``channel_overrides`` > config.yaml, and the session
    ``/reasoning`` > per-model > global level (``_resolve_session_reasoning_config``)."""
    from gateway.run import _load_gateway_config, _resolve_gateway_model
    from gateway.session_local import effort_fields
    config = _load_gateway_config()
    block = config.get('model')
    block = block if isinstance(block, dict) else {}
    model, provider, base_url = _resolve_gateway_model(config), block.get('provider'), block.get('base_url')
    channel = runner._channel_override_for(live.source)
    if channel is not None:
        model, provider = channel.model or model, channel.provider or provider
    override = runner._session_model_override(live.route) or runner.session_store.get_model_override(live.route) or {}
    model = override.get('model') or model
    provider, base_url = override.get('provider') or provider, override.get('base_url') or base_url
    reasoning = runner._resolve_session_reasoning_config(source=live.source, session_key=live.route, model=model)
    tier = runner._resolve_session_service_tier(source=live.source, session_key=live.route)
    return {'model': model, **effort_fields(reasoning, tier, model, provider, base_url)}
