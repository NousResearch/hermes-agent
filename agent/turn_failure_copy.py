"""User-facing copy and ``failure_reason`` stamping for terminal failed-turn results.

Every terminal result dict the turn loop returns must carry ``failure_reason`` (a
``FailoverReason`` value or one of :data:`SITE_FAILURE_CODES`) and ``failure_retryable`` so
``agent/error_surface.py`` yields a specific descriptor instead of ``unknown``. The copy
tables here say WHAT happened and WHAT TO DO in plain words; raw provider detail rides a
trailing "Provider said:" / "Details:" line.

The copy lives in the catalog (``explainer.failure.*`` / ``explainer.cause.*``). Every builder
returns English by default: that is the text a result's ``error`` stores, transcripts persist and
the gateway matches. Pass ``lang=None`` for the active language when the text is only shown to a
person (a ``final_response`` that is never written back as an assistant row).
"""

from __future__ import annotations

import time
from string import Formatter
from typing import Any, Dict, NamedTuple, Optional, Tuple

from agent.error_classifier import FailoverReason
from agent.i18n import DEFAULT_LANGUAGE, t, tl
from hermes_constants import display_hermes_home

# Failure codes minted by loop sites that are not provider verdicts (see module docstring).
SITE_FAILURE_CODES = frozenset({
    "context_overflow", "truncated", "invalid_response", "empty_response", "loop_error",
    "interpreter_shutdown", "session_busy",
})


def stamp_failure(result: Dict[str, Any], reason: str, retryable: bool) -> Dict[str, Any]:
    """Stamp the UI verdict fields on a terminal result (in place; returns it)."""
    result["failure_reason"] = reason
    result["failure_retryable"] = bool(retryable)
    return result


# ---- failed-turn transcript boundary ----------------------------------------------------------
# The Hermes-authored assistant row that closes a durable turn which ended without one. A
# transcript boundary, NOT the model's answer: no provider/model error or refusal detail is
# ever interpolated (that rides ``final_response``). Owned here so the core closer
# (``agent/conversation_loop.py::run_conversation``) and the gateway's own writer
# (``gateway/run_turn.py::_hmwa_close_failed_turn``) say the same thing.

FAILED_TURN_NOTICE = (
    "Your request was not processed. Send it again if you still want me to carry it out."
)
PARTIAL_FAILED_TURN_NOTICE = (
    "This turn did not complete. Some actions may already have run; verify their effects "
    "before resending."
)
# ``messages.display_kind`` of that row: display-only (stripped before every provider request),
# so renderers show a Hermes notice and room pollers never read it as the model's reply.
FAILED_TURN_DISPLAY_KIND = "failed_turn"


def untyped_failed_turn_display_kind(role: Any, content: Any) -> Optional[str]:
    """``FAILED_TURN_DISPLAY_KIND`` for a boundary row persisted before the closers typed it
    (exact notice text, so a real reply quoting it stays a reply); read-side only."""
    if role == "assistant" and isinstance(content, str) and content.strip() in (
        FAILED_TURN_NOTICE, PARTIAL_FAILED_TURN_NOTICE,
    ):
        return FAILED_TURN_DISPLAY_KIND
    return None


def failed_turn_notice(turn_messages: Any) -> str:
    """Boundary copy for a failed turn: never claim "not processed" when a tool may have run."""
    for row in turn_messages or ():
        if isinstance(row, dict) and (
            row.get("role") == "tool" or (row.get("role") == "assistant" and row.get("tool_calls"))
        ):
            return PARTIAL_FAILED_TURN_NOTICE
    return FAILED_TURN_NOTICE


def provider_label_for(provider: Any) -> str:
    """Human-friendly provider name for chat copy (``"OpenRouter"``, ``"Nous Portal"``…)."""
    from hermes_cli.models import provider_label

    return provider_label(str(provider or ""))


# ---- turn_exit_reason → failure verdict (finalize_turn stamps these) --------------------------

class ExitFailure(NamedTuple):
    """Verdict for a loop exit. ``fails_turn`` False = advisory: the descriptor fields are
    stamped so Desktop/TUI show a specific code, but ``failed``/``completed`` keep the values
    the loop chose — cron silence, the kanban dispatcher breaker and gateway transcript
    persistence all key on ``failed`` and must not change because a code was added."""

    reason: str
    retryable: bool
    fails_turn: bool = True


# (exit-reason prefix, failure_reason, retryable, fails_turn). Prefix match: several reasons
# carry a parenthesised detail (``local_processing_error(...)``).
_EXIT_REASON_FAILURES: Tuple[Tuple[str, str, bool, bool], ...] = (
    # Advisory: the reasoning-only text may literally be the answer, and cron stays silent.
    ("empty_response_exhausted", "empty_response", True, False),
    ("all_retries_exhausted_no_response", FailoverReason.server_error.value, True, True),
    # #55316/#54756: the loop stopped on a tool tail with no follow-up text; the
    # finalizer synthesizes the visible close and fails the turn.
    ("pending_tool_result", "loop_error", True, True),
    ("interpreter_shutdown", "interpreter_shutdown", False, True),
    # Advisory: a deterministic local bug is not a task failure for the kanban breaker.
    ("local_processing_error", "loop_error", False, False),
    ("repeated_outer_errors", "loop_error", True, True),
    ("error_near_max_iterations", "loop_error", True, True),
    ("context_compression_timeout", "context_overflow", False, True),
    ("context_compression_exhausted", "context_overflow", False, True),
    ("ollama_runtime_context_too_small", "context_overflow", False, True),
    # Advisory: the loop ends these as an incomplete (not failed) turn with an explainer.
    ("redirect_restart_limit_exceeded", "loop_error", True, False),
    ("rebuilt_restart_limit_exceeded", "loop_error", True, False),
)


# Provider error code carried inside an HTTP-200 body → classifier reason.
_INVALID_RESPONSE_CODES: Dict[int, str] = {
    429: FailoverReason.rate_limit.value,
    500: FailoverReason.server_error.value, 502: FailoverReason.server_error.value,
    503: FailoverReason.overloaded.value, 529: FailoverReason.overloaded.value,
    504: FailoverReason.timeout.value, 524: FailoverReason.timeout.value,
}


def invalid_response_failure_reason(response: Any) -> str:
    """``failure_reason`` for an empty/malformed HTTP-200 body: the embedded provider error
    code when there is one (so the desktop shows Retry + Switch provider consistently), else
    the ``invalid_response`` site code."""
    err = getattr(response, "error", None) if response is not None else None
    code = getattr(err, "code", None) if err is not None else None
    if code is None and isinstance(err, dict):
        code = err.get("code")
    try:
        return _INVALID_RESPONSE_CODES.get(int(code), "invalid_response") if code is not None else "invalid_response"
    except (TypeError, ValueError):
        return "invalid_response"


def exit_reason_failure(turn_exit_reason: Any) -> Optional[ExitFailure]:
    """:class:`ExitFailure` for a loop exit that carries a failure verdict, else None."""
    reason = str(turn_exit_reason or "")
    for prefix, code, retryable, fails_turn in _EXIT_REASON_FAILURES:
        if reason.startswith(prefix):
            return ExitFailure(code, retryable, fails_turn)
    return None


def is_max_iteration_handoff(result: Any) -> bool:
    """A non-failed, non-interrupted ``max_iterations_reached(N/N)`` result that still carries a
    summary. ``completed`` is False because the work did not finish in that turn, but the turn
    itself is a resumable boundary — not a failure — so cron delivers the summary and an active
    ``/goal`` may judge it (#102213). Provider/API failures never match (cf. #63180)."""
    if not isinstance(result, dict):
        return False
    if result.get("failed") is True or result.get("interrupted") is True:
        return False
    if result.get("completed") is not False:
        return False
    reason = result.get("turn_exit_reason")
    if not (isinstance(reason, str) and reason.startswith("max_iterations_reached(")):
        return False
    return bool(str(result.get("final_response") or "").strip())


# ---- chat copy tables -----------------------------------------------------------------------
# Values are catalog keys; the English source of every sentence is ``locales/en.yaml``.

# Lead clause per classifier reason once retries and fallback are exhausted.
_EXHAUSTED_LEADS: Dict[str, str] = {
    FailoverReason.rate_limit.value: "explainer.failure.exhausted_lead.rate_limit",
    FailoverReason.upstream_rate_limit.value: "explainer.failure.exhausted_lead.rate_limit",
    FailoverReason.overloaded.value: "explainer.failure.exhausted_lead.overloaded",
    FailoverReason.server_error.value: "explainer.failure.exhausted_lead.server_error",
    FailoverReason.timeout.value: "explainer.failure.exhausted_lead.timeout",
}
_EXHAUSTED_DEFAULT_LEAD = "explainer.failure.exhausted_lead.default"

# Terminal copy for a non-retryable provider rejection, keyed by classifier reason.
_NONRETRYABLE_COPY: Dict[str, str] = {
    reason.value: f"explainer.failure.nonretryable.{reason.value}"
    for reason in (
        FailoverReason.model_not_found, FailoverReason.format_error, FailoverReason.role_alternation,
        FailoverReason.ssl_cert_verification, FailoverReason.provider_policy_blocked,
        FailoverReason.upstream_blocked,
    )
}
_NONRETRYABLE_DEFAULT_COPY = "explainer.failure.nonretryable.default"
_AUTH_COPY: Dict[str, str] = {"oauth": "explainer.failure.auth_oauth", "api_key": "explainer.failure.auth_api_key"}

# English for importers that print it on their own line (CLI); chat copy is ``content_policy_copy``.
CONTENT_POLICY_NEXT_STEPS = (
    "Try rewording your message or removing sensitive attachments, or switch to another "
    "model with /model."
)

# ---- one reason → "what happened" gloss, shared by cron, subagent and chat notices ------------

# FailoverReason / site code → catalog key of one clause (no HTTP codes, no "provider" jargon).
# Reasons absent here are NOT provider-shaped; callers fall back to the raw error text.
FAILURE_CAUSE_GLOSS: Dict[str, str] = {
    FailoverReason.timeout.value: "explainer.cause.timeout",
    FailoverReason.rate_limit.value: "explainer.cause.rate_limit",
    FailoverReason.upstream_rate_limit.value: "explainer.cause.rate_limit",
    FailoverReason.overloaded.value: "explainer.cause.overloaded",
    FailoverReason.server_error.value: "explainer.cause.server_error",
    FailoverReason.billing.value: "explainer.cause.billing",
    # Wire-level billing code (not a FailoverReason) that error_surface routes to the billing layer.
    "billing_unverified": "explainer.cause.billing",
    FailoverReason.auth.value: "explainer.cause.auth",
    FailoverReason.auth_permanent.value: "explainer.cause.auth",
    FailoverReason.upstream_blocked.value: "explainer.cause.upstream_blocked",
    FailoverReason.model_not_found.value: "explainer.cause.model_not_found",
    FailoverReason.content_policy_blocked.value: "explainer.cause.content_policy_blocked",
    FailoverReason.provider_policy_blocked.value: "explainer.cause.provider_policy_blocked",
    "context_overflow": "explainer.cause.context_overflow",
    "payload_too_large": "explainer.cause.context_overflow",
}
# Clauses that name who was asking: a whole sentence per asker (``_job`` suffix for a cron job),
# so a translation never embeds an English "this job" / "the job's".
_ASKER_GLOSS_KEYS = frozenset({"explainer.cause.model_not_found", "explainer.cause.context_overflow"})
_JOB_ASKER = ("this job", "the job's")


def failure_cause_gloss(
    reason: Any, *, subject: str = "it", possessive: str = "its", lang: Optional[str] = None,
) -> Optional[str]:
    """Plain clause for a classified ``failure_reason`` in the active language (``lang`` to pin
    one); None when the reason has no gloss. ``subject``/``possessive`` name who was asking:
    ``("this job", "the job's")`` selects the cron phrasing, anything else the subagent one."""
    key = FAILURE_CAUSE_GLOSS.get(str(reason or ""))
    if key is None:
        return None
    if key in _ASKER_GLOSS_KEYS and (subject, possessive) == _JOB_ASKER:
        key += "_job"
    return t(key, lang=lang)


# ---- site-code copy -------------------------------------------------------------------------

# Chat copy for the codes in SITE_FAILURE_CODES that a loop site renders itself
# (``empty_response`` is worded by agent/turn_explainers.py, ``session_busy`` by the lease).
_FAILURE_CODE_COPY: Dict[str, str] = {
    code: f"explainer.failure.{code}"
    for code in ("context_overflow", "truncated", "invalid_response", "loop_error", "interpreter_shutdown")
}

# One-off outcome strings: deterministic loop exits that are NOT failure codes (the result
# they ride carries a code from the table above, or none at all).
# ``server_context_rejection`` deliberately avoids the overflow phrases gateway/run_turn.py
# matches on (``_CONTEXT_OVERFLOW_ERROR_PHRASES``): this failure is transient, so the user's
# message must stay in the transcript and the session must not be auto-reset.
# ``truncated_unreported`` rides failure_reason="truncated": args were cut mid-JSON but the model
# never reported an output-length stop, so it doesn't claim one (#91717).
# ``stream_closed_tool_call`` rides failure_reason="truncated": clean EOF (no transport error, no
# finish_reason) mid tool-call, retries exhausted — not a network problem on the user's side (#102766).
# ``local_processing_error`` rides failure_reason="loop_error" (advisory; the turn is incomplete).
_ONE_OFF_COPY: Dict[str, str] = {
    code: f"explainer.failure.{code}"
    for code in (
        "payload_too_large", "compression_disabled", "server_context_rejection", "truncated_unreported",
        "stream_dropped_tool_call", "stream_closed_tool_call", "local_processing_error", "reasoning_only",
        "max_iterations_no_summary", "nous_rate_limit",
    )
}
_SITE_COPY: Dict[str, str] = {**_FAILURE_CODE_COPY, **_ONE_OFF_COPY}


def _blank_missing_fields(key: str, fields: Dict[str, Any]) -> None:
    """Give every placeholder of the English template that ``fields`` lacks an empty string."""
    for _, name, _, _ in Formatter().parse(t(key, lang=DEFAULT_LANGUAGE)):
        if name:
            fields.setdefault(name, "")


def site_copy(code: str, *, lang: Optional[str] = DEFAULT_LANGUAGE, **fields: Any) -> str:
    """Chat copy for a failure code or one-off loop outcome; unknown fields default to empty strings.
    The English default carries its catalog key (``render_localized`` shows the active language)."""
    key = _SITE_COPY[code]
    fields.setdefault("home", display_hermes_home())
    _blank_missing_fields(key, fields)
    return tl(key, **fields) if lang == DEFAULT_LANGUAGE else t(key, lang=lang, **fields)


def exhausted_copy(
    reason: str, *, label: str, attempts: int, summary: str, reset_seconds: Optional[float] = None,
    lang: Optional[str] = DEFAULT_LANGUAGE,
) -> str:
    """Chat copy once retries + fallback are exhausted (``max_retries_exhausted_result``). A rate
    limit whose reset window is known names it: an 8.6h plan quota is not "wait a minute" (#89401)."""
    lead = t(_EXHAUSTED_LEADS.get(reason, _EXHAUSTED_DEFAULT_LEAD), lang=lang, label=label, attempts=attempts)
    if reset_seconds is not None and reset_seconds >= 120:
        from agent.retry_utils import format_reset_window
        situation = t("explainer.failure.exhausted_situation_reset", lang=lang,
                      window=format_reset_window(reset_seconds))
    else:
        situation = t("explainer.failure.exhausted_situation_unavailable", lang=lang)
    return t("explainer.failure.exhausted", lang=lang, lead=lead, situation=situation, summary=summary)


def limit_reset_copy(resets_at: float, now: Optional[float] = None, lang: Optional[str] = DEFAULT_LANGUAGE) -> str:
    """One chat/CLI line naming when the provider says the limit lifts (#98852): the Retry-After
    / ``resets_at`` the loop already honours for backoff, shown to the user instead of a bare
    "wait a minute". Local wall-clock time plus the remaining wait; empty once it has passed."""
    now = time.time() if now is None else now
    remaining = int(resets_at - now)
    if remaining <= 0:
        return ""
    hours, minutes = divmod((remaining + 59) // 60, 60)
    wait = (t("explainer.failure.limit_wait_hours", lang=lang, hours=hours, minutes=f"{minutes:02d}") if hours
            else t("explainer.failure.limit_wait_minutes", lang=lang, minutes=minutes))
    return t("explainer.failure.limit_resets", lang=lang,
             time=time.strftime('%H:%M', time.localtime(resets_at)), wait=wait)


def oauth_relogin_command(provider: Any) -> str:
    """The exact re-login command for a rejected OAuth grant, naming the provider slug and the active
    named profile: a profile's credentials are its own (93889b770da), so a bare ``hermes auth`` from
    the root profile re-signs the wrong store and the goal judge, reading a bare 401, guesses which
    service revoked the token (#114012)."""
    from hermes_constants import profile_cli_selector

    slug = str(provider or "").strip().lower()
    if slug == "nous":
        return f"hermes {profile_cli_selector()}portal"
    return f"hermes {profile_cli_selector()}auth add {slug} --type oauth"


def relogin_command_hint(provider: Any) -> str:
    """Re-sign-in command for a rejected credential on surfaces that may not know the provider:
    the exact OAuth command for a known OAuth slug, ``hermes auth add <slug>`` for a known API-key
    slug, and the ``<provider>`` placeholder when the slug is unknown — always carrying the
    ``-p <profile>`` selector so a profile user never re-signs the ROOT store (#114012)."""
    from hermes_constants import profile_cli_selector

    slug = str(provider or "").strip().lower()
    if not slug:
        return f"hermes {profile_cli_selector()}auth add <provider>"
    from agent.error_surface import auth_kind

    if auth_kind(slug) == "oauth":
        return oauth_relogin_command(slug)
    return f"hermes {profile_cli_selector()}auth add {slug}"


def nonretryable_copy(
    classified: Any, *, provider: Any, model: Any, summary: str, prefix_suggestion: Optional[str] = None,
    lang: Optional[str] = DEFAULT_LANGUAGE,
) -> str:
    """Chat copy for a terminal non-retryable rejection (auth, model missing, TLS, generic 4xx)."""
    label = provider_label_for(provider)
    if getattr(classified, "is_auth", False):
        from agent.error_surface import auth_kind

        key = _AUTH_COPY[auth_kind(str(provider or ""))]
    else:
        key = _NONRETRYABLE_COPY.get(classified.reason.value, _NONRETRYABLE_DEFAULT_COPY)
    prefix_hint = (
        t("explainer.failure.model_prefix_hint", lang=lang, suggestion=prefix_suggestion)
        if prefix_suggestion else ""
    )
    body = t(key, lang=lang, label=label, model=model, home=display_hermes_home(), prefix_hint=prefix_hint,
             relogin=oauth_relogin_command(provider))
    return t("explainer.failure.with_provider_detail", lang=lang, body=body, summary=summary)


def content_policy_copy(*, label: str, summary: str, lang: Optional[str] = DEFAULT_LANGUAGE) -> str:
    return t("explainer.failure.content_policy", lang=lang, label=label, summary=summary)


def short_detail(exc: Any, limit: int = 200) -> str:
    """First line of an exception's text, capped, for a trailing ``Details:`` line."""
    text = (str(exc) or type(exc).__name__).strip().splitlines()
    first = text[0] if text else type(exc).__name__
    return first if len(first) <= limit else first[: limit - 1] + "…"
