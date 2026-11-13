"""What a teammate's finished turn tells the ``message_agent`` sender that asked it something.

Forgotten reply: the single-reply protocol makes the teammate's final response the reply. A model that ends its
turn with a silence marker or nothing, after it wrote its answer beside a tool call or tried to
send it back through ``message_agent`` (refused: the final response is the reply), would leave the
sender with an empty receipt. Its last visible words that turn are relayed instead, once, marked
so the sender knows they were not a final answer. A notify-only request (``[no reply needed]``)
never gets one: silence is its expected outcome.
"""
import json

FORGOTTEN_REPLY_PREFIX = '[no final reply; their last message this turn:] '


# A turn its own gateway's stop or restart interrupted settles ``interrupted`` and is never re-run: for
# the sender that is a dead request, not a cancel it asked for.
OWNER_STOPPED_ERROR = ('its gateway stopped or restarted mid-turn, so the turn was interrupted and is not '
                       'run again (some steps may already have run)')


def owner_stopped(result):
    """True for a turn result its own gateway's stop or restart interrupted (``hermes_agent/gateway/run.py`` reasons)."""
    from hermes_agent.gateway.run import _INTERRUPT_REASON_GATEWAY_RESTART, _INTERRUPT_REASON_GATEWAY_SHUTDOWN
    return result.get('interrupt_message') in (_INTERRUPT_REASON_GATEWAY_RESTART, _INTERRUPT_REASON_GATEWAY_SHUTDOWN)


def forgotten_reply(authority, admission_id, record):
    """The marked fallback reply for ``record``'s settled-but-empty admission, or None."""
    from hermes_agent.tools.bot_mode_probe import NO_REPLY_MARKER
    author = record.get('author') or {}
    if not author.get('is_bot') or NO_REPLY_MARKER in (record.get('message') or ''):
        return None
    from hermes_agent.gateway.session_results import admission_result
    saved = (admission_result(authority.db, admission_id) or {}).get('result') or {}
    text = _last_words(saved.get('messages') or [], str(author.get('name') or '').lower())
    return FORGOTTEN_REPLY_PREFIX + text if text else None


def _last_words(messages, sender):
    from hermes_agent.gateway.response_filters import is_intentional_silence_response
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        for call in reversed(message.get('tool_calls') or []):
            text = _reply_attempt(call, sender)
            if text:
                return text
        content = message.get('content')
        if isinstance(content, str) and content.strip() and not is_intentional_silence_response(content):
            return content.strip()
    return None


def _reply_attempt(call, sender):
    """The body of a ``message_agent`` call this turn made back to its own sender."""
    function = (call or {}).get('function') or {} if isinstance(call, dict) else {}
    if function.get('name') != 'message_agent':
        return None
    try:
        args = json.loads(function.get('arguments') or '{}')
    except ValueError:
        return None
    if not isinstance(args, dict) or str(args.get('target') or '').strip().lstrip('@').lower() != sender:
        return None
    return str(args.get('message') or '').strip() or None
