"""Plain-language copy for chat-turn failures shown in the CLI response panel.

The chat panel used to echo ``Error: HTTP 401: Invalid API key`` as the assistant's answer. These
helpers map the classifier verdict (``agent/error_classifier.py``) to WHAT happened + WHAT TO DO,
and demote the raw provider text to a ``Details:`` line. Copy lives in the i18n catalog under
``cli.error.*`` and is resolved at render time for the active language.
"""

from __future__ import annotations

from agent.i18n import t

_SUMMARY_LIMIT = 120

# FailoverReason.value -> catalog key suffix. Reasons sharing copy point at one key.
_REASON_KEYS: dict[str, str] = {
    "auth": "auth",
    "auth_permanent": "auth",
    "billing": "billing",
    "model_not_found": "model_not_found",
    "rate_limit": "rate_limit",
    "upstream_rate_limit": "rate_limit",
    "upstream_blocked": "upstream_blocked",
    "overloaded": "overloaded",
    "server_error": "server_error",
    "timeout": "timeout",
}
_UNKNOWN_KEY = "unknown"


def _short(text: str, limit: int = _SUMMARY_LIMIT) -> str:
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


def chat_error_response(
    error: Exception | str, *, provider: str = "", model: str = "", failure_reason: str | None = None,
) -> str:
    """Two-line panel text: plain sentence with the fix command, then ``Details: <raw>``.

    ``failure_reason`` is the verdict the turn loop already stamped on its result
    (``agent/turn_failure_copy.stamp_failure``). When present it is used as-is instead of
    re-classifying a summarised string (which has no status code and almost always lands on
    'unknown'); for the loop's own site codes the ``error`` text is already the user-facing copy,
    so it is returned verbatim rather than wrapped and demoted to a Details line."""
    reason = str(failure_reason or "").strip()
    if reason:
        from agent.turn_failure_copy import SITE_FAILURE_CODES

        text = str(error or "").strip()
        if reason in SITE_FAILURE_CODES and text:
            return text
    else:
        from agent.error_classifier import classify_api_error

        exc = error if isinstance(error, Exception) else Exception(str(error))
        reason = classify_api_error(exc, provider=provider or "", model=model or "").reason.value
    lead = t(
        f"cli.error.{_REASON_KEYS.get(reason, _UNKNOWN_KEY)}",
        provider=provider or t("cli.error.the_provider"),
        model=model or t("cli.error.the_current_model"),
    )
    return t("cli.error.with_details", lead=lead, details=_short(str(error), 300))


def agent_init_failure_message(error: BaseException) -> str:
    """Copy for a failed AIAgent build on first message: the user's turn was dropped."""
    return t("cli.error.agent_init_failed", error=_short(str(error)) or type(error).__name__)
