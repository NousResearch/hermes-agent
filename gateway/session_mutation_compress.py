"""Summary preparation has no canonical store; publication owns the only write."""
import asyncio
import json
import logging
from pathlib import Path

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
        # Nothing removable (a short session keeps every message): main's read-only
        # "No changes from compression" report, estimated like main's (system prompt + tools).
        tokens = _request_tokens(resident, messages)
        return {'status': 'noop', 'summary': summarize_manual_compression(
            messages, messages, tokens, tokens, compression_state=compressor)}
    summary = summarize_manual_compression(messages, compressed, estimate_request_tokens_rough(messages),
                                           estimate_request_tokens_rough(compressed), compression_state=compressor)
    from agent.model_metadata import estimate_messages_tokens_rough
    hook = None if policy.safe_mode else {  # same rough basis as the agent path's compaction record
        'before_tokens': estimate_messages_tokens_rough(messages),
        'after_tokens': estimate_messages_tokens_rough(compressed), 'messages_removed': len(messages) - len(compressed)}
    return dict(prepared, messages=compressed, summary=summary, post_compaction=hook,
                in_place=is_truthy_value(options.get('in_place'), default=True))


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


def _request_tokens(agent, messages):
    """Main's /compress request estimate: the agent's system prompt and tool schemas plus the messages."""
    from agent.model_metadata import estimate_request_tokens_rough
    return estimate_request_tokens_rough(messages, system_prompt=getattr(agent, '_cached_system_prompt', '') or '',
                                         tools=getattr(agent, 'tools', None) or None)
