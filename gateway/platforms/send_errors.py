"""Send-error classification helpers for gateway platforms.

Extracted verbatim from ``gateway/platforms/base.py`` (shard s2, cluster c8:
_error_blob, classify_send_error, is_chat_level_not_found, plus the
chat/sub-chat not-found substring constants).
``gateway/platforms/base.py`` re-exports the public names so existing
import sites keep working unchanged.
"""

from typing import Optional


# ``not_found`` substrings split by blast radius.  A *chat-level* not_found means
# the chat/user/group itself is gone, so the whole target is dead.  A
# *thread/topic/message-level* not_found (a deleted forum topic, an edited-away
# message) leaves the parent chat reachable — it must NOT mark the whole chat
# dead.  ``classify_send_error`` collapses both into ``"not_found"``;
# ``is_chat_level_not_found`` recovers the distinction for the dead-target path.
# See gateway.dead_targets.
_CHAT_LEVEL_NOT_FOUND_SUBSTRINGS = ("chat not found",)
_SUBCHAT_NOT_FOUND_SUBSTRINGS = (
    "message to edit not found",
    "message to reply not found",
    "thread not found",
    "topic_deleted",
    "message_id_invalid",
)


def _error_blob(exc: Optional[BaseException] = None, error_text: str = "") -> str:
    """Build the lowercased text blob both send-error classifiers match against.

    Single source of truth so ``classify_send_error`` and
    ``is_chat_level_not_found`` can never drift (e.g. one including the
    exception class name and the other not) and silently disagree on the same
    failure.  Includes ``str(exc)`` (when non-empty) and the exception's class
    name, plus any explicit ``error_text``.
    """
    parts = []
    if error_text:
        parts.append(error_text)
    if exc is not None:
        exc_str = str(exc)
        if exc_str:
            parts.append(exc_str)
        parts.append(exc.__class__.__name__)
    return " ".join(parts).lower()


def classify_send_error(exc: Optional[BaseException], error_text: str = "") -> str:
    """Map a send exception / error string to a :data:`SEND_ERROR_KINDS` value.

    Platform-neutral: matches on the lowercased text of ``exc`` (and/or the
    explicit ``error_text``) against the substrings the major messaging APIs
    use.  Conservative — anything unrecognized returns ``"unknown"`` so callers
    never mistake an unclassified failure for a benign one.
    """
    # Local import: _RETRYABLE_ERROR_PATTERNS stays in base.py (also used by
    # BasePlatformAdapter._is_retryable_error); module-level import would
    # cycle with base.py's re-export of this module.
    from gateway.platforms.base import _RETRYABLE_ERROR_PATTERNS
    blob = _error_blob(exc, error_text)
    if not blob.strip():
        return "unknown"
    if "message_too_long" in blob or "too long" in blob or "message is too long" in blob:
        return "too_long"
    if (
        "can't parse entities" in blob
        or "cant parse entities" in blob
        or "can't find end" in blob
        or "unsupported start tag" in blob
        or ("entity" in blob and "parse" in blob)
        or ("bad request" in blob and "entit" in blob)
    ):
        return "bad_format"
    if (
        "forbidden" in blob
        or "bot was blocked" in blob
        or "blocked by the user" in blob
        or "user is deactivated" in blob
        or "not enough rights" in blob
        or "have no rights" in blob
        or "not a member" in blob
    ):
        return "forbidden"
    if any(s in blob for s in _CHAT_LEVEL_NOT_FOUND_SUBSTRINGS) or any(
        s in blob for s in _SUBCHAT_NOT_FOUND_SUBSTRINGS
    ):
        return "not_found"
    if (
        "flood" in blob
        or "too many requests" in blob
        or "retry after" in blob
        or "rate limit" in blob
    ):
        return "rate_limited"
    for pat in _RETRYABLE_ERROR_PATTERNS:
        if pat in blob:
            return "transient"
    if "connecttimeout" in blob:
        return "transient"
    return "unknown"


def is_chat_level_not_found(exc: Optional[BaseException] = None, error_text: str = "") -> bool:
    """Whether a ``not_found`` failure means the *whole chat* is gone.

    :func:`classify_send_error` collapses chat-level and thread/topic/message-level
    not_found into the single ``"not_found"`` kind.  Only the chat-level case (the
    chat/user/group no longer exists) should mark a delivery target dead; a deleted
    forum topic or an edited-away message leaves the parent chat reachable.  When
    both a chat-level and a sub-chat marker are present, the sub-chat reading wins
    (conservative: never kill a chat that may still be reachable).

    Argument order mirrors :func:`classify_send_error` (``exc`` first) and both
    share :func:`_error_blob`, so the two classifiers cannot disagree on the same
    failure.
    """
    blob = _error_blob(exc, error_text)
    if any(s in blob for s in _SUBCHAT_NOT_FOUND_SUBSTRINGS):
        return False
    return any(s in blob for s in _CHAT_LEVEL_NOT_FOUND_SUBSTRINGS)
