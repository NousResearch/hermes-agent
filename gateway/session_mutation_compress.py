"""Summary preparation has no canonical store; publication owns the only write."""
import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)


def compress_request(payload):
    """``session.mutate(compress)`` payload → the shared ``CompressRequest``.

    A ``focus``-only payload is the raw ``/compress`` argument string (Ink/Desktop send it that
    way, like ``session.compress`` always has), so it goes through the parser every native surface
    uses: ``--preview`` stays read-only and ``here [N]`` stays a boundary instead of becoming a
    focus topic. ``--aggressive`` has no canonical implementation and is refused before any write.
    """
    from agent.conversation_compression_manual import CompressRequest, parse_compress_args
    if set(payload) <= {'focus'}:
        request = parse_compress_args(payload.get('focus', ''))
    else:
        request = CompressRequest(preview=payload.get('preview', False), partial=payload.get('partial', False),
                                  keep_last=payload.get('keep_last', 2), focus_topic=payload.get('focus'))
    if request.aggressive:
        raise RuntimeStoreError('unsupported_compress_options')
    return request


async def prepare_compress(authority, live, payload, prepared):
    from gateway.session_policy import restore_policy, policy_scope
    from agent.context_compressor import ContextCompressor
    from agent.agent_init import _parse_config_int
    from agent.model_metadata import estimate_request_tokens_rough
    from hermes_cli.partial_compress import (
        rejoin_compressed_head_and_tail, split_history_for_partial_compress, summarize_compress_preview)
    from utils import is_truthy_value
    request = compress_request(payload)
    snapshot = prepared['snapshot']
    policy = restore_policy(snapshot['receipt']['policy'])
    config = policy.config(authority)
    options = config.get('compression') or {}
    if options.get('checkpoint_required'):
        raise RuntimeStoreError('compression_checkpoint_required')
    model, runtime = authority.runner._resolve_session_agent_runtime(source=live.source, session_key=live.route)
    if runtime.get('api_mode') == 'codex_app_server':
        raise RuntimeStoreError('runtime_coordination_required')
    history = authority.db.get_messages_as_conversation(snapshot['target'])
    messages = [m for m in history if m.get('role') in {'user', 'assistant', 'tool'}]
    head, tail = messages, []
    if request.partial:
        head, tail = split_history_for_partial_compress(messages, request.keep_last)
        if not tail:  # degenerate split: nothing to keep verbatim → full compression
            head = messages
    if request.preview:
        # Read-only: the report is the whole result; the caller commits nothing.
        report = summarize_compress_preview(messages, request.partial, request.keep_last, request.focus_topic,
                                            estimate_request_tokens_rough(messages))
        return {'status': 'preview', 'lines': report['lines'], 'head_count': report['head_count'],
                'tail_count': report['tail_count'], 'message_count': report['total']}
    # The session's resident agent owns its memory providers; give them the same pre-compress
    # turn (and the same best-effort contract) the in-process compressor does. No resident agent
    # (cold session, or a managed worker that runs out of process) means no provider to call.
    cached_agent = getattr(authority.runner, '_cached_agent_for', None)
    resident = cached_agent(live.route) if callable(cached_agent) else None

    def summarize():
        with policy_scope(policy, authority=authority):
            from agent.conversation_compression import _pre_compress_memory_context
            memory_context = _pre_compress_memory_context(resident, head, False) if resident is not None else ''
            engine = _plugin_engine(authority, policy, config, model, runtime, resident, snapshot['target'])
            if engine is not None:
                try:
                    return _engine_compress(engine, head, tail, messages, request, memory_context), engine
                except BaseException:
                    if resident is None:
                        _end_engine(engine, snapshot['target'], messages)
                    raise
            compressor = ContextCompressor(model, base_url=runtime.get('base_url') or '',
                api_key=runtime.get('api_key') or '', provider=runtime.get('provider') or '',
                api_mode=runtime.get('api_mode') or '', quiet_mode=True, abort_on_summary_failure=True,
                protect_first_n=options.get('protect_first_n', 3), protect_last_n=options.get('protect_last_n', 20),
                min_tail_user_messages=max(1, _parse_config_int(options.get('min_tail_user_messages', 1), 1)),
                custom_providers=config.get('custom_providers'))
            compressed = compressor.compress(json.loads(json.dumps(head)), force=True,
                                               focus_topic=request.focus_topic, memory_context=memory_context)
            if not any(m.get('_compressed_summary') for m in compressed):
                return None, compressor
            return rejoin_compressed_head_and_tail(compressed, tail), compressor
    compressed, compressor = await asyncio.to_thread(summarize)
    # The same report every native /compress prints ("Compressed: N → M messages").
    from agent.manual_compression_feedback import summarize_manual_compression
    if compressed is None:
        if resident is None and not isinstance(compressor, ContextCompressor):
            await asyncio.to_thread(_end_engine, compressor, snapshot['target'], messages)
        # Nothing removable (a short session keeps every message): main's read-only
        # "No changes from compression" report, estimated like main's (system prompt + tools).
        row = None if resident is not None else authority.db.get_session(snapshot['target'])  # resident: row unused
        tokens = _request_tokens(resident, messages, row)
        return {'status': 'noop', 'summary': summarize_manual_compression(
            messages, messages, tokens, tokens, compression_state=compressor)}
    summary = summarize_manual_compression(messages, compressed, estimate_request_tokens_rough(messages),
                                           estimate_request_tokens_rough(compressed), compression_state=compressor)
    from agent.model_metadata import estimate_messages_tokens_rough
    hook = None if policy.safe_mode else {  # same rough basis as the agent path's compaction record
        'before_tokens': estimate_messages_tokens_rough(messages),
        'after_tokens': estimate_messages_tokens_rough(compressed), 'messages_removed': len(messages) - len(compressed)}
    # The committed boundary is announced to the engine that compressed (``on_session_start``
    # with boundary_reason="compression"), as the agent path does after its own commit.
    engine_owner = SimpleNamespace(context_compressor=compressor, platform=_platform(live.source),
                                   _gateway_session_key=live.route,
                                   owned=resident is None and not isinstance(compressor, ContextCompressor))
    return dict(prepared, messages=compressed, summary=summary, post_compaction=hook, engine_owner=engine_owner,
                in_place=is_truthy_value(options.get('in_place'), default=True))


def _engine_compress(engine, head, tail, messages, request, memory_context):
    """One forced ``compress()`` on a plugin engine; None when it changed nothing or would grow the
    conversation (the agent path refuses a growing result the same way)."""
    from agent.conversation_compression import _supported_compression_kwargs
    from agent.model_metadata import estimate_messages_tokens_rough, estimate_request_tokens_rough
    from hermes_cli.partial_compress import rejoin_compressed_head_and_tail
    engine._last_compress_refused_would_grow = False
    compressed = engine.compress(json.loads(json.dumps(head)), **_supported_compression_kwargs(
        engine.compress, current_tokens=estimate_request_tokens_rough(head),
        focus_topic=request.focus_topic, force=True, memory_context=memory_context))
    if compressed == head:
        return None
    compressed = rejoin_compressed_head_and_tail(compressed, tail)
    if estimate_messages_tokens_rough(compressed) > estimate_messages_tokens_rough(messages):
        engine._last_compress_refused_would_grow = True  # read by the /compress report
        return None
    return compressed


def _end_engine(engine, session_id, messages):
    try:
        engine.on_session_end(session_id, messages)
    except Exception:  # health: allow BLE001 -- releasing a one-off engine must not fail /compress
        logger.warning('context engine on_session_end failed', exc_info=True)


def _configured_context_length(config, model, provider, base_url, custom_providers):
    """The configured window agent init hands a plugin engine: ``model.context_length`` when it
    describes this runtime, else the matching custom provider's entry."""
    from agent.agent_init import _scope_context_length_to_default_runtime
    from hermes_cli.config import get_custom_provider_context_length
    model_cfg = config.get('model') if isinstance(config.get('model'), dict) else {}
    try:
        configured = int(model_cfg['context_length']) if model_cfg.get('context_length') is not None else None
    except (TypeError, ValueError):
        configured = None
    if configured is not None:
        route = SimpleNamespace(model=model, provider=provider, base_url=base_url)
        configured = _scope_context_length_to_default_runtime(
            route, config, model_cfg, custom_providers, configured, base_url)
    if configured is None and custom_providers:
        configured = get_custom_provider_context_length(
            model=model, base_url=base_url, custom_providers=custom_providers)
    return int(configured) if configured else None


def _plugin_engine(authority, policy, config, model, runtime, resident, target):
    """The session's configured plugin context engine, or None for the built-in compressor.

    A resident agent's engine is the one bound to this session. A cold or managed-worker session
    gets the engine its frozen policy names, selected and bound the way agent init does, so a
    plugin engine compresses its own history instead of having it rewritten underneath it."""
    from agent.context_compressor import ContextCompressor
    if resident is not None:
        engine = getattr(resident, 'context_compressor', None)
        return None if engine is None or isinstance(engine, ContextCompressor) else engine
    if policy.safe_mode:
        return None
    from agent.agent_init import _select_context_engine
    engine = _select_context_engine(config)
    if engine is None:
        return None
    from agent.model_metadata import get_model_context_length
    thresholds = (config.get('compression') or {}).get('model_thresholds')
    if isinstance(thresholds, dict) and thresholds:  # before update_model, like agent init
        engine.model_thresholds = {str(k): float(v) for k, v in thresholds.items()}
    from hermes_cli.config import get_compatible_custom_providers
    base_url, api_key = runtime.get('base_url') or '', runtime.get('api_key') or ''
    provider = runtime.get('provider') or ''
    providers = get_compatible_custom_providers(config)
    context_length = get_model_context_length(
        model, base_url=base_url, api_key=api_key, provider=provider, custom_providers=providers,
        config_context_length=_configured_context_length(config, model, provider, base_url, providers))
    engine.update_model(model=model, context_length=context_length, base_url=base_url, api_key=api_key,
                        provider=provider, api_mode=runtime.get('api_mode') or '')
    bind = getattr(engine, 'bind_session_state', None)
    if callable(bind):
        bind(session_db=authority.db, session_id=target)
    return engine


def _platform(source):
    platform = getattr(source, 'platform', None)
    return getattr(platform, 'value', platform) or 'cli'


async def finish_compress_engine(authority, prepared, result):
    """After the receipt transaction: a committed ``/compress`` (``result``) gets the agent path's
    post-commit notification, the engine's compression boundary plus the relay/metrics segment
    hand-off. A cold session's engine was built for this compression only, so it gets the session
    end a resident agent's boundary eviction gives, committed or not. An exact retry (no
    ``engine_owner``) notifies nothing."""
    from agent.conversation_compression import _notify_context_engine_compression_complete
    from gateway.run import _async_profile_runtime_scope
    owner = (prepared or {}).get('engine_owner')
    if owner is None or (result is None and not owner.owned):
        return

    def notify():
        if result is not None:
            _notify_context_engine_compression_complete(owner, new_session_id=result['target_session_id'],
                                                        old_session_id=result['previous_target_session_id'])
        if owner.owned:
            target = result['target_session_id'] if result is not None else prepared['snapshot']['target']
            _end_engine(owner.context_compressor, target, prepared['messages'])
    try:
        async with _async_profile_runtime_scope(Path(authority.profile_id)):
            await asyncio.to_thread(notify)
    except Exception:  # health: allow BLE001 -- the compaction is committed; an engine failure must not fail the reply
        logger.warning('context engine compression notification failed', exc_info=True)


async def fire_post_compaction_hook(authority, prepared, result):
    """``post_compaction`` for a committed ``/compress``: this path summarizes outside the agent's
    ``compress_context``, so it fires here, after the commit, in the session's profile."""
    from agent.compaction_events import fire_post_compaction
    from gateway.run import _async_profile_runtime_scope
    target, previous = result['target_session_id'], result['previous_target_session_id']
    try:
        async with _async_profile_runtime_scope(Path(authority.profile_id)):
            await asyncio.to_thread(
                fire_post_compaction, session_id=target, old_session_id='' if previous == target else previous,
                kind='summarize', reason='manual', **prepared['post_compaction'])
    except Exception:  # the compaction is committed; a plugin-side failure must not fail the reply
        logger.warning('post_compaction hook delivery failed', exc_info=True)


def _request_tokens(agent, messages, row=None):
    """Main's /compress request estimate: the agent's system prompt and tool schemas plus the messages.
    No resident agent (a cold session): the prompt and tools[] pin the session row stores, which the
    next agent restores and sends, so the figure matches what main's resident agent reported."""
    from agent.model_metadata import estimate_request_tokens_rough
    prompt = getattr(agent, '_cached_system_prompt', '') or ''
    tools = getattr(agent, 'tools', None) or None
    if agent is None and isinstance(row, dict):
        prompt = row.get('system_prompt') or ''
        try:
            pin = json.loads(row.get('tool_names') or 'null')
        except (TypeError, ValueError):
            pin = None  # a pin hash whose row was swept resolves to itself
        from tools.mcp_tool_agent import pinned_tool_defs
        tools = pinned_tool_defs(pin) or None
    return estimate_request_tokens_rough(messages, system_prompt=prompt, tools=tools)
