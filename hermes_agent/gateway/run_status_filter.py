"""Gateway status-line filtering helpers (facade: ``hermes_agent/gateway/run.py``'s ``_prepare_gateway_status_message``).

DM-only status lines:
The blocked-overflow warning (context over the compression threshold, compression blocked) tells the reader to
run /new or /compress. In a DM that reader is the owner and can act; in any shared chat (group, supergroup,
forum, channel, thread, guild, ...) those commands are admin-gated and the token counts and block reason leak
internal state to every member (#101003). Only complete, unmodified instances of the emit-site template match,
so similar or user-quoted text is never swallowed.
"""

import re
from typing import Any

from hermes_agent.agent.conversation_compression import CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE

# "" = an unclassified source: keep the warning (the pre-#101003 behaviour) rather than guess it is shared.
_DM_CHAT_TYPES = frozenset({"dm", "private", "direct", ""})
_CANONICAL_COUNT = r"(?:0|[1-9][0-9]{0,2}(?:,[0-9]{3})*)"


def _status_template_to_regex(template: str) -> str:
    """Compile a compression status template constant into a regex source.

    Literal text is escaped verbatim (wording drift can't diverge from the matcher); ``{field}`` -> numeric."""
    parts = re.split(r"\{[^{}]*\}", template)
    return r"[\d,]+".join(re.escape(part) for part in parts)


def _template_regex(template: str) -> re.Pattern[str]:
    """Full-match regex for one emit-site template: literal text escaped, counts canonical, reason free-form."""
    fields = {"{tokens:,}": _CANONICAL_COUNT, "{threshold:,}": _CANONICAL_COUNT, "{reason}": r"[\s\S]+?"}
    parts = re.split(r"(\{tokens:,\}|\{threshold:,\}|\{reason\})", template)
    return re.compile("".join(fields.get(part) or re.escape(part) for part in parts))


_DM_ONLY_STATUS_RES = (_template_regex(CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE),)


def status_chat_type(source: Any) -> Any:
    """The chat type DM-only statuses are filtered on: a session every participant shares is never a DM,
    even one its adapter keys ``dm`` (a Slack group DM: ``shared_session`` on a ``dm`` source)."""
    return "group" if getattr(source, "shared_session", False) else getattr(source, "chat_type", None)


def is_terminal_failure_status(message: Any) -> bool:
    from hermes_agent.gateway.warning_notifications import TerminalFailureText
    return isinstance(message, TerminalFailureText)


def hidden_outside_dm(chat_type: Any, message: str) -> bool:
    """True when ``message`` is a DM-only status and ``chat_type`` is not a DM."""
    kind = str(getattr(chat_type, "value", chat_type) or "").strip().lower()
    if kind in _DM_CHAT_TYPES:
        return False
    return any(pattern.fullmatch(message) for pattern in _DM_ONLY_STATUS_RES)
