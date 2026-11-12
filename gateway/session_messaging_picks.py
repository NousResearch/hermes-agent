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


def _require_chat_pick_owner(db, session_id):
    """Check both the viewer's mount and the physical chat before runtime overrides change."""
    from hermes_state_lineage import guarded_discovered_session_write
    guarded_discovered_session_write(
        db, (session_id,), lambda conn: (session_id, db._compression_tip_on(conn, session_id)),
        lambda conn: None)


def _ownership_refusal(exc):
    from hermes_state_lineage import SessionLineageInventoryError
    return RuntimeStoreError('storage_unavailable' if isinstance(exc, SessionLineageInventoryError)
                             else 'admission_conflict')


def is_chat_pick(live, operation):
    from gateway.config import Platform
    return operation in CHAT_PICKS and live is not None and live.source.platform != Platform.LOCAL


def _platform_key(live):
    from gateway.run import _platform_config_key
    return _platform_config_key(live.source.platform)


def _apply_reasoning(runner, live, effort):
    """``/reasoning <level>`` (``None``: ``/reasoning reset``) as the chat would run it."""
    return runner._apply_reasoning_selection(live.route, _platform_key(live), 'reset' if effort is None else effort)


async def _pick_model(authority, owner, live, payload, row):
    """The chat's ``/model <name> [--provider p]`` resolution, guarded like a LOCAL pick, then committed
    as its session override. Returns the receipt fields, or the guard's confirmation request."""
    from gateway.session_mutation_model import _Route, _selection_guard_refusal, is_model_reset
    from gateway.slash_commands_model import _ModelSwitchContext
    from hermes_constants import get_hermes_home
    runner = authority.runner
    ctx = _ModelSwitchContext(session_key=live.route, source=live.source,
                              config_path=get_hermes_home() / 'config.yaml', persist_global=False)
    ctx.read_config()
    if is_model_reset(payload):  # the chat's own `/model reset`: its override goes, its reply is the note
        default_model, default_provider = await asyncio.to_thread(
            runner._channel_default_route, live.source, ctx.config_path)
        note = await runner._handle_model_reset(None, ctx, picker=True, persist_db=False)
        return ({'model': (await asyncio.to_thread(messaging_route_info, runner, live))['model'], 'note': note},
                {'model': default_model, 'provider': default_provider or 'auto'})
    # After a restart the chat's saved provider/base_url is only on disk: resolve against it, not
    # config.yaml's provider, or the record step overwrites it (as run_turn_prepare rehydrates).
    runner._rehydrate_session_model_override(live.route)
    ctx.apply_override(runner._session_model_override(live.route) or {})
    result, error = await runner._perform_model_switch(ctx, payload['model'], payload.get('provider', ''), live.source)
    if error is not None:
        raise RuntimeStoreError('model_resolution_failed', detail=error)
    snapshot = {'target': owner.session_id, 'target_revision': row['runtime_revision']}
    current = _Route(ctx.current_provider, ctx.current_model, ctx.current_base_url, ctx.current_api_key)
    refusal = await _selection_guard_refusal(authority, live, payload, snapshot, current, result)
    if refusal is not None:
        return refusal, None
    # No in-place swap of the cached agent: a turn may be running on it. The override is what the
    # next turn builds from, and the record step evicts the cached agent.
    await runner._record_model_switch(
        result, ctx, source=live.source, one_turn=False, picker=True, persist_db=False)
    fields = {'model': result.new_model, 'provider': result.target_provider}
    if 'reasoning' in payload:
        _apply_reasoning(runner, live, payload['reasoning'])
        fields['effort'] = payload['reasoning']
    return fields, {'model': result.new_model, 'provider': result.target_provider,
                    'base_url': result.base_url, 'api_mode': result.api_mode}


async def apply_chat_pick(authority, ref, owner, live, params, *, principal_id):
    """Apply a ``model`` / ``reasoning`` mutation to the messaging chat *live* mounted on *owner* (the
    caller authorized control and holds its mutation lock). The fences are the mount's, the ones
    ``session.resume`` handed the viewer; comparing them with a compression segment's row refused a
    pick on a compressed chat with ``stale_generation``. The receipt is keyed on that mount too: an
    exact retry answers its stored receipt without re-applying the override, and the commit bumps the
    mount's revision, so a retry of an older pick refuses ``revision_conflict``."""
    from hermes_state_mutation_chat_pick import chat_pick_key, commit_chat_pick, prepare_chat_pick
    from hermes_state_mutations import validate_action
    operation, payload = params['operation'], params['payload']
    validate_action(operation, payload)
    key, digest = chat_pick_key(principal_id, owner.session_id, params)
    fenced = dict(epoch=authority.epoch, key=key, digest=digest, session_id=owner.session_id, params=params)
    prepared = await asyncio.to_thread(prepare_chat_pick, authority.db, **fenced)
    if 'receipt' in prepared:
        return prepared['receipt']
    from hermes_state_lineage import SessionLineageError
    try:
        await asyncio.to_thread(_require_chat_pick_owner, authority.db, owner.session_id)
    except SessionLineageError as exc:
        raise _ownership_refusal(exc) from exc
    row = prepared['row']
    runner = authority.runner
    route = live.route
    prior_reasoning = runner._session_state(route).conversation.reasoning_override
    prior_reasoning_config = runner._reasoning_config
    prior_memory = runner._session_model_overrides.get(route)
    prior_saved = runner.session_store.get_model_override(route) if operation == 'model' else None
    prior_note = getattr(runner, '_pending_model_notes', {}).get(route)
    prior_once = getattr(runner, '_pending_one_turn_model_restores', {}).get(route)
    model_update = None
    if operation == 'reasoning':
        await asyncio.to_thread(_apply_reasoning, runner, live, payload['effort'])
        fields = {'effort': payload['effort']}
    else:
        async with runner._model_switch_lock():
            fields, model_update = await _pick_model(authority, owner, live, payload, row)
    if fields.get('status') == 'confirmation_required':
        return {'session_id': ref.session_id, 'operation': operation, **fields}
    result = {'session_id': ref.session_id, 'operation': operation, 'revision': row['runtime_revision'],
              'execution_generation': row['runtime_generation'], **fields}
    try:
        result = await asyncio.to_thread(commit_chat_pick, authority.db, **fenced,
                                         result=result, model_update=model_update)
    except Exception as exc:
        # The cross-store owner may change after the precheck. Restore runtime and
        # routing overrides if the receipt did not commit; the DB model belongs to
        # the guarded receipt transaction, so there is no provisional DB write.
        runner._set_session_reasoning_override(route, prior_reasoning)
        runner._reasoning_config = prior_reasoning_config
        if operation == 'model':
            if prior_memory is None:
                runner._session_model_overrides.pop(route, None)
            else:
                runner._session_model_overrides[route] = prior_memory
            for name, old in (('_pending_model_notes', prior_note),
                              ('_pending_one_turn_model_restores', prior_once)):
                mapping = getattr(runner, name, None)
                if mapping is not None:
                    if old is None:
                        mapping.pop(route, None)
                    else:
                        mapping[route] = old
            try:
                await runner.async_session_store.set_model_override(route, prior_saved)
            except Exception as restore_error:
                raise RuntimeStoreError('storage_unavailable') from restore_error
        runner._evict_cached_agent(route)
        if isinstance(exc, SessionLineageError):
            raise _ownership_refusal(exc) from exc
        raise
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
    # The wire level clamps per api_mode (a Codex app-server route keeps ``ultra``); config.yaml's
    # api_mode belongs to its own provider only.
    api_mode = override.get('api_mode') or (block.get('api_mode') if provider == block.get('provider') else None)
    reasoning = runner._resolve_session_reasoning_config(source=live.source, session_key=live.route, model=model)
    tier = runner._resolve_session_service_tier(source=live.source, session_key=live.route)
    return {'model': model, **effort_fields(reasoning, tier, model, provider, base_url, api_mode)}
