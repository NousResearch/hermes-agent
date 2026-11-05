"""Break the non-retryable API-error → chat-echo → new-turn burn loop (#143940).

When the model API returns a terminal client error (billing 402, permanent auth
401/403), the agent emits a diagnostic status and a ``final_response``. On
messaging surfaces those outbound texts can be re-admitted as inbound (self-echo,
peer/profile relay, or interrupt follow-up). The turn manager then starts another
turn, hits the same 402, and loops without backoff.

Three cooperating guards:

1. Messaging status delivery suppresses the terminal diagnostic (the final reply
   still informs the user).
2. After such a failure, a session hold drops inbound/follow-ups that look like
   the error echo until a dissimilar human message (or a slash command) arrives.
3. Pending follow-ups drained after a terminal failure are discarded when they
   match the echo shape, so an in-turn interrupt cannot re-arm the loop.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

# SessionEntry.metadata key. Value is a small dict; absence means no hold.
HOLD_META_KEY = "api_error_turn_hold"

# Failure reasons that must end the run and arm the hold (issue #143940).
_TERMINAL_HOLD_REASONS = frozenset({
    "billing",
    "auth",
    "auth_permanent",
})

# Observed diagnostic / final_response shapes from agent/turn_recovery.py and
# agent/conversation_loop.py, plus the sanitized gateway.errors.provider_kept_failing
# rewrite that would otherwise keep the loop alive after sanitization.
_RAW_DIAGNOSTIC_RE = re.compile(
    r"("
    r"retrying\s+won'?t\s+help"
    r"|billing\s+or\s+credits\s+exhausted"
    r"|provider\s+reported\s+usage/credit\s+exhaustion"
    r"|credits\s+exhausted\s*[—\-:]"
    r")",
    re.IGNORECASE,
)

_SANITIZED_GATEWAY_ECHO_RE = re.compile(
    r"("
    r"the\s+ai\s+model\s+service\s+kept\s+failing"
    r"|authentication\s+failed\.?\s+check\s+your\s+api\s+key"
    r")",
    re.IGNORECASE,
)

# Short envelopes that are just the HTTP client error (status/final after redact).
_HTTP_CLIENT_ERROR_LINE_RE = re.compile(
    r"^[\W_]*(?:error[:\s]+)?http\s*40[123]\b",
    re.IGNORECASE,
)


def looks_like_terminal_nonretryable_diagnostic(text: str) -> bool:
    """True for the agent diagnostic lines that must not be posted to messaging chats.

    Broader prose that merely *mentions* a status code stays False so ordinary
    assistant content is not silenced.
    """
    body = str(text or "").strip()
    if not body or len(body) > 2000 or body.count("\n") > 6:
        return False
    if _RAW_DIAGNOSTIC_RE.search(body):
        return True
    return bool(_HTTP_CLIENT_ERROR_LINE_RE.match(body))


def looks_like_api_error_echo(text: str) -> bool:
    """True when inbound/follow-up text looks like a re-admitted API-error notice."""
    body = str(text or "").strip()
    if not body or len(body) > 2000 or body.count("\n") > 6:
        return False
    if looks_like_terminal_nonretryable_diagnostic(body):
        return True
    return bool(_SANITIZED_GATEWAY_ECHO_RE.search(body))


def is_terminal_nonretryable_api_failure(result: Optional[Mapping[str, Any]]) -> bool:
    """True when an agent/gateway result is a terminal billing/auth failure."""
    if not isinstance(result, dict):
        return False
    if result.get("failure_retryable") is True:
        return False
    reason = str(result.get("failure_reason") or "").strip()
    if reason in _TERMINAL_HOLD_REASONS:
        return True
    # Some paths only stamp ``failed`` + billing_block without failure_reason.
    if result.get("failed") and isinstance(result.get("billing_block"), dict):
        return True
    return False


def arm_api_error_hold(entry: Any, *, reason: str) -> bool:
    """Arm the per-session hold on *entry*. Returns True when metadata changed."""
    if entry is None:
        return False
    meta = getattr(entry, "metadata", None)
    if not isinstance(meta, dict):
        try:
            entry.metadata = {}
            meta = entry.metadata
        except (AttributeError, TypeError) as exc:
            logger.debug("api_error_hold: cannot attach metadata on %r: %s", type(entry).__name__, exc)
            return False
    prev = meta.get(HOLD_META_KEY)
    payload = {"reason": str(reason or "api_error"), "armed_at": time.time()}
    if prev == payload:
        return False
    meta[HOLD_META_KEY] = payload
    return True


def clear_api_error_hold(entry: Any) -> bool:
    """Clear the hold. Returns True when a hold was present."""
    if entry is None:
        return False
    meta = getattr(entry, "metadata", None)
    if not isinstance(meta, dict) or HOLD_META_KEY not in meta:
        return False
    meta.pop(HOLD_META_KEY, None)
    return True


def api_error_hold_active(entry: Any) -> bool:
    meta = getattr(entry, "metadata", None) if entry is not None else None
    return isinstance(meta, dict) and HOLD_META_KEY in meta


def maybe_arm_hold_from_result(entry: Any, result: Optional[Mapping[str, Any]]) -> bool:
    """Arm the hold when *result* is a terminal non-retryable API failure."""
    if not is_terminal_nonretryable_api_failure(result):
        return False
    reason = str((result or {}).get("failure_reason") or "api_error")
    return arm_api_error_hold(entry, reason=reason)


def should_drop_text_under_api_error_hold(
    entry: Any,
    text: str,
    *,
    is_command: bool = False,
) -> tuple[bool, bool]:
    """Decide whether to drop *text* while a hold is active.

    Returns ``(drop, cleared)``. Slash commands never drop (caller still runs them).
    A dissimilar human message clears the hold and is admitted.
    """
    if not api_error_hold_active(entry):
        return False, False
    if is_command:
        return False, False
    if looks_like_api_error_echo(text):
        return True, False
    cleared = clear_api_error_hold(entry)
    return False, cleared


def discard_pending_api_error_echo(
    result: Optional[Mapping[str, Any]],
    pending_text: Optional[str],
) -> bool:
    """True when a drained follow-up after a terminal failure is an error echo and must die."""
    if not pending_text or not is_terminal_nonretryable_api_failure(result):
        return False
    return looks_like_api_error_echo(pending_text)

def arm_hold_on_session_store(store: Any, entry: Any, result: Optional[Mapping[str, Any]], session_key: Optional[str]) -> bool:
    """Arm + persist the hold after a terminal failure. Returns True when armed."""
    if not maybe_arm_hold_from_result(entry, result if isinstance(result, dict) else None):
        return False
    if store is not None and hasattr(store, "_save"):
        store._save()
    logger.info(
        "Armed api-error turn hold for session %s after terminal failure (%s)",
        session_key or "?",
        (result or {}).get("failure_reason") if isinstance(result, dict) else "?",
    )
    return True


def take_pending_if_api_error_echo(
    result: Optional[Mapping[str, Any]],
    pending_event: Any,
    pending: Optional[str],
    session_key: Optional[str],
) -> tuple[Any, Optional[str]]:
    """Clear ``(pending_event, pending)`` when the drained follow-up is an error echo."""
    if not discard_pending_api_error_echo(result, pending):
        return pending_event, pending
    logger.info(
        "Discarding non-retryable API error echo follow-up for session %s",
        session_key or "?",
    )
    return None, None

