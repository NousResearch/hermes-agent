"""Summary preparation has no canonical store; publication owns the only write."""
import asyncio
import json

from hermes_state_runtime import RuntimeStoreError


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
    def summarize():
        with policy_scope(policy, authority=authority):
            compressor = ContextCompressor(model, base_url=runtime.get('base_url') or '',
                api_key=runtime.get('api_key') or '', provider=runtime.get('provider') or '',
                api_mode=runtime.get('api_mode') or '', quiet_mode=True, abort_on_summary_failure=True,
                protect_first_n=options.get('protect_first_n', 3), protect_last_n=options.get('protect_last_n', 20),
                min_tail_user_messages=max(1, _parse_config_int(options.get('min_tail_user_messages', 1), 1)),
                custom_providers=config.get('custom_providers'))
            compressed = compressor.compress(json.loads(json.dumps(head)), force=True,
                                               focus_topic=request.focus_topic)
            if not any(m.get('_compressed_summary') for m in compressed):
                raise RuntimeStoreError('nothing_to_compress')
            return rejoin_compressed_head_and_tail(compressed, tail)
    compressed = await asyncio.to_thread(summarize)
    return dict(prepared, messages=compressed, in_place=is_truthy_value(options.get('in_place'), default=True))
