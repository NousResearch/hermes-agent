"""The turn-result rules the store applies when it stores a settled admission's result: the current-turn projection of an agent result and the anchor search it falls back on.

Standard library and the sibling ``session_rules`` modules only.
"""

from __future__ import annotations

from typing import Any

from session_rules.turns import user_originated_turn_view


def reanchor_current_turn_user_idx(messages: list[Any], user_message: Any) -> int:
    """Locate this turn's user message after compaction rebuilt ``messages``.

    Prefers the LAST user message whose content exactly matches this turn's text, else
    the last user-originated turn; compaction handoffs are never the fallback.
    Returns -1 when there is no user-originated message.

    Compression replaces list entries with fresh copies (and may append a todo-snapshot user message or a
    restored user turn AFTER the surviving copy of the current turn's message), so a pre-compression index
    is meaningless. Prefer the LAST user message whose content exactly matches this turn's text — the
    surviving copy in the common case — so the injection stamp and the #48677 persist override can't land on
    a todo-snapshot or historical row. Fall back to the last *user-originated* turn when no exact match
    survives (merge-summary-into-tail rewrites the content but the trackers still need a live anchor).
    Compaction handoffs must never become the fallback anchor (#80622) — they are reference-only
    scaffolding, not the active ask.
    """
    fallback = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not (isinstance(msg, dict) and msg.get("role") == "user"):
            continue
        # Typed synthetic current events keep their persistence anchor when raw
        # content is unchanged; not eligible for the human-only fallback below.
        if msg.get("content") == user_message:
            return i
        live_view = user_originated_turn_view(msg)
        if live_view is None:
            continue
        if live_view.get("content") == user_message:
            return i
        # Prefer a real human turn over a synthetic handoff / continuation marker
        # when the exact content was rewritten by merge-into-tail.
        if fallback < 0:
            fallback = i
    return fallback


def compact_result(result, *, user_message=None):
    """Retain replay/accounting and current-turn API output, never cumulative history."""
    result = dict(result)
    value = result.get('result')
    if not isinstance(value, dict):
        return result
    value = dict(value)
    # The tool schema list rides every turn result for in-process transcript logging; it is the
    # same session-wide copy each turn (tens of KB, more with MCP) and no stored-result reader uses it.
    value.pop('tools', None)
    messages = value.get('messages')
    if isinstance(messages, list):
        # Agent results include all previous turns. APIs only project this turn's
        # assistant/tool suffix; previous_response_id stores its own history. Compression
        # is the exception: its replacement snapshot is the only immutable record of the
        # rewritten prefix for this turn. A later live transcript may include a successor.
        if not value.get('_compressed') and not value.get('_messages_are_turn_suffix'):
            anchor = value.get('current_turn_user_idx')
            if not (type(anchor) is int and 0 <= anchor < len(messages)
                    and isinstance(messages[anchor], dict) and messages[anchor].get('role') == 'user'):
                anchor = reanchor_current_turn_user_idx(messages, user_message)
            value['messages'] = messages[anchor + 1:]
            if anchor >= 0 and isinstance(messages[anchor], dict) and messages[anchor].get('api_content'):
                # The user row as the model saw it (memory prefetch, plugin context): a chained
                # Responses turn replays it instead of a row rebuilt from the raw request text.
                value['_turn_user_row'] = {k: messages[anchor][k] for k in ('role', 'content', 'api_content')}
            # Recovery/verification nudges are later user-role rows within this same turn.
            # Consumers must not infer a new boundary from them after the real anchor is gone.
            value['_messages_are_turn_suffix'] = True
            value.pop('current_turn_user_idx', None)
    result['result'] = value
    return result
