"""Gateway status-line filtering helpers (facade: ``gateway/run.py``'s ``_prepare_gateway_status_message``).

DM-only status lines:
The blocked-overflow warning (context over the compression threshold, compression blocked) tells the reader to
run /new or /compress. In a DM that reader is the owner and can act; in any shared chat (group, supergroup,
forum, channel, thread, guild, ...) those commands are admin-gated and the token counts and block reason leak
internal state to every member (#101003). Only complete, unmodified instances of the emit-site template match,
so similar or user-quoted text is never swallowed.
"""

import re
from collections.abc import Callable
from typing import Any

from agent.conversation_compression import CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE
from agent.memory_provider import RecallIndicator
from gateway.display_config import resolve_display_setting
from gateway.warning_notifications import TerminalFailureText

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


# The empty-response ladder's buffered lines (agent/turn_empty_response.py), flushed when the turn ends
# empty: its final reply explains that, and "switching to fallback..." would otherwise be the chat's last
# status even though the switch failed. Chat surfaces drop them like the retry counter.
EMPTY_LADDER_STATUS_PATTERN = (
    r"model\s+returning\s+empty\s+responses\s+[—-]+\s+switching\s+to\s+fallback"
    r"|repeatedly\s+returning\s+empty\s+content\s+[—-]+\s+skipping\s+further\s+retries"
)

_DM_ONLY_STATUS_RES = (_template_regex(CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE),)


def status_chat_type(source: Any) -> Any:
    """The chat type DM-only statuses are filtered on: a session every participant shares is never a DM,
    even one its adapter keys ``dm`` (a Slack group DM: ``shared_session`` on a ``dm`` source)."""
    return "group" if getattr(source, "shared_session", False) else getattr(source, "chat_type", None)


def hidden_chat_status(message: str, chat_type: Any, platform_key: str, load_config: Callable[[], dict]) -> bool:
    """Filter typed status notices and DM-only warnings before chat delivery."""
    if isinstance(message, TerminalFailureText):
        return True
    if isinstance(message, RecallIndicator):
        return resolve_display_setting(load_config(), platform_key, "recall_notifications") == "off"
    kind = str(getattr(chat_type, "value", chat_type) or "").strip().lower()
    if kind in _DM_CHAT_TYPES:
        return False
    return any(pattern.fullmatch(str(message)) for pattern in _DM_ONLY_STATUS_RES)
