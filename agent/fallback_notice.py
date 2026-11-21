"""The user-visible fallback-switch notice: why the picked model could not respond, and when it is shown.

The fallback walk (``chat_completion_helpers.try_activate_fallback``) builds one line per switch and
shows it at the switch, before the fallback model is called, so the user sees which model is answering
while the turn runs. The error handler that classified the failure records the provider's own words on
``agent._fallback_trigger``; the switch for that same reason quotes them, redacted and truncated.
"""

from __future__ import annotations

import logging

from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)

_FALLBACK_REASON_LABELS = {
    FailoverReason.auth: "authentication failed",
    FailoverReason.auth_permanent: "authentication permanently failed",
    FailoverReason.billing: "billing or quota exhausted",
    FailoverReason.rate_limit: "rate limit",
    FailoverReason.upstream_rate_limit: "upstream model rate limit",
    FailoverReason.overloaded: "provider overloaded",
    FailoverReason.server_error: "provider server error",
    FailoverReason.timeout: "request timeout",
    FailoverReason.ssl_cert_verification: "TLS certificate verification failed",
    FailoverReason.context_overflow: "context window exceeded",
    FailoverReason.payload_too_large: "request payload too large",
    FailoverReason.image_too_large: "image payload too large",
    FailoverReason.model_not_found: "model not found",
    FailoverReason.provider_policy_blocked: "provider policy blocked the request",
    FailoverReason.content_policy_blocked: "content policy blocked the request",
    FailoverReason.format_error: "request format rejected",
    FailoverReason.role_alternation: "adjacent same-role messages rejected",
    FailoverReason.invalid_encrypted_content: "encrypted reasoning state rejected",
    FailoverReason.multimodal_tool_content_unsupported: "multimodal tool content unsupported",
    FailoverReason.thinking_signature: "thinking signature rejected",
    FailoverReason.long_context_tier: "long-context tier unavailable",
    FailoverReason.oauth_long_context_beta_forbidden: "OAuth long-context beta unavailable",
    FailoverReason.llama_cpp_grammar_pattern: "grammar pattern rejected",
    FailoverReason.unknown: "provider failure",
}

_FALLBACK_DETAIL_MAX_CHARS = 200


def fallback_reason_text(reason: FailoverReason | None) -> str:
    """Return a concise operator-facing explanation for a fallback switch."""
    label = _FALLBACK_REASON_LABELS.get(reason)
    return label or str(getattr(reason, "value", None) or reason or "provider failure").replace("_", " ")


def record_fallback_trigger(agent, reason: FailoverReason | None, error_text: str) -> None:
    """Remember the provider's words for the switch this failure may cause. Masked here, while the
    failing route's credentials are still on the agent: a provider echoing an opaque key back
    (``Credential <key> rejected``) is caught by value, which pattern masking alone can miss."""
    from agent.credential_pool_errors import redact_error_text
    known = (getattr(agent, "api_key", None), getattr(agent, "_anthropic_api_key", None))
    agent._fallback_trigger = (reason, redact_error_text(str(error_text or ""), known))


def fallback_trigger_text(agent, reason: FailoverReason | None) -> str:
    """The reason label plus the provider's own error. The error handler that classified the failure
    stashes ``(reason, summary)`` on ``_fallback_trigger``; it is used once, and only by a switch for
    that same reason, so an empty-response or truncation switch never borrows an older error."""
    label = fallback_reason_text(reason)
    stashed = getattr(agent, "_fallback_trigger", None)
    agent._fallback_trigger = None
    if not (isinstance(stashed, tuple) and len(stashed) == 2 and reason is not None and stashed[0] == reason):
        return label
    from agent.redact import redact_sensitive_text
    detail = " ".join(redact_sensitive_text(str(stashed[1] or "")).split())
    if not detail:
        return label
    if len(detail) > _FALLBACK_DETAIL_MAX_CHARS:
        detail = detail[:_FALLBACK_DETAIL_MAX_CHARS - 1].rstrip() + "…"
    return f"{label}: {detail}"


def fallback_notice_hidden(agent) -> bool:
    """A background wake turn (a completed process, a plugin's scheduled check: no user message) shows
    no switch notice. Most of those turns end silent, and the notice is posted before anyone knows,
    so during an outage every wake would post one into the chat. The kind is known at turn start,
    so a turn the user sent still gets its notice before the fallback answers."""
    from gateway.response_filters import is_machinery_display_kind
    return is_machinery_display_kind(getattr(agent, "_turn_display_kind", None))


def surface_hidden_fallback(agent) -> None:
    """Turn-start reconciliation of a switch a wake turn did not show (call after the primary-restore
    attempt). The fallback still serving (a quota cooldown keeps it across turns): the next turn the
    user watches announces the held notice before it runs, so a hidden switch is never an
    undisclosed one. The primary back: the user never saw it leave, so the held notice is dropped."""
    held = getattr(agent, "_hidden_fallback_notice", None)
    if not held or fallback_notice_hidden(agent):
        return
    agent._hidden_fallback_notice = None
    if getattr(agent, "_provider_fallback_active", False):
        announce_fallback_switch(agent, held)


def announce_primary_restored(agent, previous_model: str, previous_provider: str) -> None:
    """The recovery line after a turn-start restore. None for a switch the user never saw (hidden
    on a wake turn), and none on a wake turn: during an outage each wake would restore, re-fail and
    post it again."""
    unseen = getattr(agent, "_hidden_fallback_notice", None)
    agent._hidden_fallback_notice = None
    if unseen or fallback_notice_hidden(agent):
        return
    try:
        agent._emit_diagnostic_status(
            f"✅ Primary model restored: {agent.model} via {agent.provider}; "
            f"fallback {previous_model} via {previous_provider} is no longer active.")
    except Exception:  # health: allow BLE001 -- a surface failure must never undo a successful restore
        logger.debug("primary restored notice failed to emit", exc_info=True)


def announce_fallback_switch(agent, notice: str) -> None:
    """Show the switch now, before the fallback model starts answering: the user must see which
    model this turn runs on while it runs, not after it ends. Display-only (status rail), so the
    cached prompt is untouched; it obeys the muted-warnings setting like every diagnostic. Not
    buffered with the retry chatter, so a terminal-failure flush cannot repeat it.

    Its own ``fallback`` kind: messaging adapters edit one status message per kind in a turn, so on
    the shared ``lifecycle`` kind the next retry or re-prompt line replaced it (Telegram)."""
    if fallback_notice_hidden(agent):
        logger.info("fallback switch notice not shown on a background wake turn: %s", notice)
        agent._hidden_fallback_notice = notice
        return
    agent._hidden_fallback_notice = None
    from gateway.warning_notifications import DiagnosticText
    try:
        agent._emit_status_kind(FALLBACK_STATUS_KIND, DiagnosticText(notice), origin="fallback_switch")
    except Exception:  # health: allow BLE001 -- a surface failure must never fail the switch itself
        logger.debug("fallback switch notice failed to emit", exc_info=True)


FALLBACK_STATUS_KIND = "fallback"
