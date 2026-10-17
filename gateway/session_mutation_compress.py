"""Summary preparation has no canonical store; publication owns the only write."""
import asyncio
import json

from hermes_state_runtime import RuntimeStoreError

# A manual /compress runs outside a turn, so no execution stamp exists to hang a status on, and the
# canonical mutation published nothing: a TUI/dashboard viewer's status bar sat on "ready" for the
# whole summarization (main's in-process path pinned the pair itself —
# ``tui_gateway/methods_session.py::_compress_live`` — and that path is unreachable now that the Ink
# TUI attaches to the shared owner). Publish the pair the auto-compaction flow already uses, which
# both Ink (``createGatewayEventHandler``) and Desktop match, under the short-session guard and the
# wording main used. The terminal frame keeps kind ``compacted`` whatever the outcome — Ink clears
# its compacting state on that kind alone (``status``/``ready`` would leave the ticker frozen) — so
# its text carries the outcome instead of main's bare ``ready``.
_VIEWER_COMPACT_MIN_MESSAGES = 4
_COMPACTED_STATUS = "✓ Context compression complete"
_NO_CHANGES_STATUS = "Context compression: no changes"
_FAILED_STATUS = "Context compression did not complete"


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
    # ``head`` is the portion being summarized — the whole history for a full run, the prefix for a
    # partial one — so both the count and the guard follow the summarized portion: a long session
    # whose ``here N`` leaves a short head stays silent while the summarizer runs, where main's
    # guard read the whole history.
    announced = len(head) >= _VIEWER_COMPACT_MIN_MESSAGES
    if announced:
        _publish_compaction_status(live, snapshot, 'compacting',
                                   _compacting_status(len(head), _request_tokens(resident, head),
                                                      request.focus_topic))

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
    # Pessimistic until the summarizer answers, so anything that escapes it (a raise, an interrupt)
    # retires the status without claiming a completion that did not happen.
    # The frame still leaves prepare, one durable write ahead of the commit (``session_mutations``):
    # another viewer's metadata edit or a model switch during the summarization bumps the revision
    # that commit re-validates, so it can refuse (``revision_conflict``) after viewers have been told
    # the summary completed; the requester's error and the unchanged transcript are the authority
    # then. Publishing after the commit would move the terminal publication into
    # ``session_mutations``, so the lifecycle stays in one module and takes that window.
    terminal = _FAILED_STATUS
    try:
        compressed, compressor = await asyncio.to_thread(summarize)
        terminal = _NO_CHANGES_STATUS if compressed is None else _COMPACTED_STATUS
    finally:
        # Retired on success, no-op and raise alike: a viewer must never be left showing "compacting".
        if announced:
            _publish_compaction_status(live, snapshot, 'compacted', terminal)
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
    return dict(prepared, messages=compressed, summary=summary,
                in_place=is_truthy_value(options.get('in_place'), default=True))


def _request_tokens(agent, messages):
    """Main's /compress request estimate: the agent's system prompt and tool schemas plus the messages."""
    from agent.model_metadata import estimate_request_tokens_rough
    return estimate_request_tokens_rough(messages, system_prompt=getattr(agent, '_cached_system_prompt', '') or '',
                                         tools=getattr(agent, 'tools', None) or None)


def _compacting_status(count, tokens, focus_topic):
    focus_suffix = f', focus: "{focus_topic}"' if focus_topic else ''
    return f"⠋ compressing {count} messages (~{tokens:,} tok){focus_suffix}…"


def _publish_compaction_status(live, snapshot, kind, text):
    """A manual /compress's viewer lifecycle, on the session's own stream.

    The id is the receipt's logical session id (what a viewer is attached with and what the client
    filters frames by), not ``snapshot['target']``, which follows a non-in-place rotation to the new
    physical transcript.
    """
    live.event_stream.publish(snapshot['receipt']['session_id'], {'kind': kind, 'text': text},
                              event_type='status.update')
