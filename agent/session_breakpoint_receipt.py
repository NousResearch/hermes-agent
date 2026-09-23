"""F-T31 A6 — typed session breakpoint receipts for network-class CLI turn failures.

Observed 2026-09-02 §4: five CLI sessions hit the local DNS outage
(``[Errno 8] nodename nor servname ...`` mid-turn) and their transcripts
simply STOP — no marker, no receipt, nothing that tells the returning human
where the conversation suspended. Recovery in every observed case was the
USER re-prompting 2.8 hours later; the session layer had zero automatic
continuation and zero receipt.

The implementation is deliberately minimal (a receipt, NOT auto-continuation): after a CLI turn fails with a network-class error, append
a typed assistant-side marker to the in-memory conversation history so it
persists with the session on terminal close. The marker names the suspension
class, the error summary, and the last successful user-turn index, so
``/history`` shows exactly where the break happened.

Classification reuses ``agent.error_classifier``: ``timeout`` reason covers
DNS/connect/transport per ``_CONNECTION_MESSAGE_PATTERNS``; ``server_error``
and ``overloaded`` are 5xx (upstream) and stay OUT of the network-receipt
class (a 5xx is not evidence the LOCAL session is suspended).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Typed receipt identifier (stable for dashboards / greps).
SESSION_SUSPENDED_NETWORK = "SESSION_SUSPENDED_NETWORK"

_BREAKPOINT_DISPLAY_KIND = "session_breakpoint_receipt"


def _last_successful_user_turn(history: list) -> int:
    """Index of the last user message that HAS a following assistant reply.

    A user message with no assistant reply after it is the suspended turn
    itself (or a staged-but-never-run turn); the receipt should point at the
    last GOOD exchange, which is the pair the human can anchor on.
    """
    last_good = -1
    for idx in range(len(history) - 1, -1, -1):
        message = history[idx]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        for later in history[idx + 1:]:
            if isinstance(later, dict) and later.get("role") == "assistant":
                last_good = idx
                break
        if last_good >= 0:
            break
    return last_good


def build_network_breakpoint_message(
    error_summary: str, *, last_successful_turn: int, reason: str
) -> Dict[str, Any]:
    """Construct the typed assistant-side receipt message (NOT appended here)."""
    return {
        "role": "assistant",
        "content": (
            f"{SESSION_SUSPENDED_NETWORK}: the turn failed with a network-class "
            f"error ({reason}); the session suspended here — the last successful "
            f"exchange is user turn #{last_successful_turn}. Re-send your last "
            f"message to continue from this point. Error: {error_summary}"
        ),
        "display_kind": _BREAKPOINT_DISPLAY_KIND,
        "display_metadata": {
            "receipt_type": SESSION_SUSPENDED_NETWORK,
            "reason": reason,
            "last_successful_turn": last_successful_turn,
        },
    }


def maybe_append_network_breakpoint(
    history: Optional[list],
    error: BaseException,
    *,
    provider: str = "",
    model: str = "",
    error_summary: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Append the typed receipt when *error* classifies as a local network failure.

    Returns the appended message (for tests / logging), or None when the error
    is not network-class or the inputs are malformed. Never raises — a receipt
    failure must not break the error path it decorates (A6 zero-network spec).
    """
    try:
        if not isinstance(history, list):
            return None
        from agent.error_classifier import classify_api_error

        exc = error if isinstance(error, BaseException) else Exception(str(error))
        reason = classify_api_error(exc, provider=provider or "", model=model or "").reason.value
        if reason != "timeout":
            # Only DNS/connect/transport (classified ``timeout`` via
            # _CONNECTION_MESSAGE_PATTERNS) suspends the LOCAL session; 5xx
            # upstream errors are provider-side and do not get a receipt.
            return None
        summary = (error_summary or str(error) or type(error).__name__)[:300]
        message = build_network_breakpoint_message(
            summary,
            last_successful_turn=_last_successful_user_turn(history),
            reason=reason,
        )
        history.append(message)
        return message
    except Exception:
        # Receipt is best-effort decoration of an error path: never raise.
        return None
