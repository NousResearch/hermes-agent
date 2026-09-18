"""User-facing copy and ``failure_reason`` stamping for terminal failed-turn results.

Every terminal result dict the turn loop returns must carry ``failure_reason`` (a
``FailoverReason`` value or one of :data:`SITE_FAILURE_CODES`) and ``failure_retryable`` so
``agent/error_surface.py`` yields a specific descriptor instead of ``unknown``. The copy
tables here say WHAT happened and WHAT TO DO in plain words; raw provider detail rides a
trailing "Provider said:" / "Details:" line.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Tuple

from agent.error_classifier import FailoverReason
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


# ---- chat copy tables -----------------------------------------------------------------------

_NEXT_STEPS_RETRY = "Wait a minute and send /retry, or switch models with /model."
_NEXT_STEPS_LOOP = (
    "Your message is saved. Send `continue` to try again, or start a new session with /new. "
    "If it happens again, run `hermes doctor` and share the error details."
)

# Lead sentence per classifier reason once retries and fallback are exhausted.
_EXHAUSTED_LEADS: Dict[str, str] = {
    FailoverReason.rate_limit.value: "{label} rate-limited every one of {attempts} attempts",
    FailoverReason.upstream_rate_limit.value: "{label} rate-limited every one of {attempts} attempts",
    FailoverReason.overloaded.value: "{label} reported it was overloaded on all {attempts} attempts",
    FailoverReason.server_error.value: "{label} returned a server error on all {attempts} attempts",
    FailoverReason.timeout.value: "{label} didn't respond in time on any of {attempts} attempts",
}
_EXHAUSTED_DEFAULT_LEAD = "{label} didn't answer after {attempts} attempts"

# Terminal copy for a non-retryable provider rejection, keyed by classifier reason.
_NONRETRYABLE_COPY: Dict[str, str] = {
    FailoverReason.model_not_found.value: (
        "Model '{model}' isn't available on {label}. Pick a different model with /model "
        "(or `hermes model` in a terminal).{prefix_hint}"
    ),
    FailoverReason.format_error.value: (
        "{label} rejected this request as malformed, so the model didn't answer. Start a clean "
        "session with /new or switch models with /model; if it keeps happening, run `hermes doctor`."
    ),
    FailoverReason.ssl_cert_verification.value: (
        "Hermes couldn't verify {label}'s security certificate, so the connection was refused. "
        "This is usually a corporate proxy or an outdated certificate store on this computer — "
        "see the terminal or `{home}/logs/agent.log` for the exact fix, or try another provider "
        "with /model."
    ),
    FailoverReason.provider_policy_blocked.value: (
        "{label}'s account settings don't allow this model for your request, so it didn't "
        "answer. Check the provider's data/privacy settings, or switch models with /model."
    ),
}
_NONRETRYABLE_DEFAULT_COPY = (
    "{label} rejected the request and retrying won't help. Pick another model with /model, "
    "or check the details in `{home}/logs/agent.log`."
)
_AUTH_COPY: Dict[str, str] = {
    "oauth": "{label} rejected your sign-in, so the model can't be reached. Sign in again: `{relogin}`.",
    "api_key": (
        "{label} rejected your API key, so the model can't be reached. Update it in "
        "Settings → Providers, or run `hermes setup` in a terminal."
    ),
}

CONTENT_POLICY_NEXT_STEPS = (
    "Try rewording your message or removing sensitive attachments, or switch to another "
    "model with /model."
)

# ---- one reason → "what happened" gloss, shared by cron, subagent and chat notices ------------

# FailoverReason / site code → one clause (no HTTP codes, no "provider" jargon). ``{subject}``
# is who was asking ("the job", "it"), ``{possessive}`` its possessive ("the job's", "its").
# Reasons absent here are NOT provider-shaped; callers fall back to the raw error text.
FAILURE_CAUSE_GLOSS: Dict[str, str] = {
    FailoverReason.timeout.value: "the AI model service did not respond in time",
    FailoverReason.rate_limit.value: "the AI model service was rate-limited (too many requests)",
    FailoverReason.upstream_rate_limit.value: "the AI model service was rate-limited (too many requests)",
    FailoverReason.overloaded.value: "the AI model service is overloaded right now",
    FailoverReason.server_error.value: "the AI model service returned an internal error",
    FailoverReason.billing.value: "the AI model service says the account's usage or credit limit is reached",
    # Wire-level billing code (not a FailoverReason) that error_surface routes to the billing layer.
    "billing_unverified": "the AI model service says the account's usage or credit limit is reached",
    FailoverReason.auth.value: "the AI model service rejected the sign-in",
    FailoverReason.auth_permanent.value: "the AI model service rejected the sign-in",
    FailoverReason.model_not_found.value: "the model {subject} uses was not found at the AI model service",
    FailoverReason.content_policy_blocked.value: "the AI model service's safety filter rejected the request",
    "context_overflow": "{possessive} request grew too large for the model",
    "payload_too_large": "{possessive} request grew too large for the model",
}


def failure_cause_gloss(reason: Any, *, subject: str = "it", possessive: str = "its") -> Optional[str]:
    """Plain clause for a classified ``failure_reason``; None when the reason has no gloss."""
    template = FAILURE_CAUSE_GLOSS.get(str(reason or ""))
    return template.format(subject=subject, possessive=possessive) if template else None


# ---- site-code copy -------------------------------------------------------------------------

# Chat copy for the codes in SITE_FAILURE_CODES that a loop site renders itself
# (``empty_response`` is worded by agent/turn_explainers.py, ``session_busy`` by the lease).
_FAILURE_CODE_COPY: Dict[str, str] = {
    "context_overflow": (
        "This conversation has grown too long for {model} to read, and Hermes couldn't shrink "
        "it enough automatically. Start a new session with /new (your history is kept), or try "
        "/compress once more. Switching to a model with a bigger context window also works."
    ),
    "truncated": (
        "The model's reply was cut off before it finished (it hit its output length limit), so "
        "Hermes didn't run the incomplete action. Nothing was changed. Send `continue`, ask for "
        "the work in smaller steps, or raise max_tokens for this model."
    ),
    "invalid_response": (
        "{label} sent back an empty or broken reply {attempts} times — it is probably overloaded "
        "or rate-limiting you. " + _NEXT_STEPS_RETRY + "\n\nDetails: {detail}"
    ),
    "loop_error": (
        "Hermes hit repeated errors and stopped this turn so it wouldn't keep retrying. "
        + _NEXT_STEPS_LOOP + "\n\nDetails: {detail}"
    ),
    "interpreter_shutdown": (
        "Hermes was shutting down and stopped this turn. Your conversation is saved — reopen "
        "it{resume} and send your message again."
    ),
}

# One-off outcome strings: deterministic loop exits that are NOT failure codes (the result
# they ride carries a code from the table above, or none at all).
_ONE_OFF_COPY: Dict[str, str] = {
    "payload_too_large": (
        "This conversation (including attachments) has grown too large to send to {model}, and "
        "Hermes couldn't shrink it enough automatically. Start a new session with /new (your "
        "history is kept), or try /compress once more."
    ),
    "compression_disabled": (
        "This conversation is too long for {model} and automatic shrinking is turned off in "
        "your settings (compression.enabled). Run /compress to shrink it now, /new to start "
        "fresh, or pick a model with a bigger context window."
    ),
    "stream_dropped_tool_call": (
        "The connection to {label} kept dropping while the model was writing a large action, "
        "so nothing was run. Check your network and send /retry; asking for the file in smaller "
        "pieces also helps."
    ),
    # Rides failure_reason="loop_error" (advisory; the turn is incomplete, not failed).
    "local_processing_error": (
        "Hermes hit an internal error while handling the model's reply and stopped this turn. "
        + _NEXT_STEPS_LOOP + "\n\nDetails: {detail}"
    ),
    "reasoning_only": (
        "⚠️ {model} spent all of its output budget thinking and never wrote an answer. Lower "
        "its reasoning effort with `/reasoning low`, or switch to a different model with /model. "
        "Its last thoughts, which may contain the answer:\n\n{preview}"
    ),
    "max_iterations_no_summary": (
        "I ran out of steps for this turn ({limit} tool calls) before finishing, and couldn't "
        "produce a summary. Send `continue` to keep going, or raise `max_iterations` in your config."
    ),
    "nous_rate_limit": (
        "Wait for the reset and send /retry, or switch models with /model. To avoid waits, add "
        "a backup provider with `hermes fallback add`."
    ),
}
_SITE_COPY: Dict[str, str] = {**_FAILURE_CODE_COPY, **_ONE_OFF_COPY}


def site_copy(code: str, **fields: Any) -> str:
    """Chat copy for a failure code or one-off loop outcome; unknown fields default to empty strings."""
    fields.setdefault("home", display_hermes_home())
    return _SITE_COPY[code].format_map(_Defaults(fields))


def exhausted_copy(reason: str, *, label: str, attempts: int, summary: str) -> str:
    """Chat copy once retries + fallback are exhausted (``max_retries_exhausted_result``)."""
    lead = _EXHAUSTED_LEADS.get(reason, _EXHAUSTED_DEFAULT_LEAD).format(label=label, attempts=attempts)
    return (
        f"{lead} — it looks temporarily unavailable. {_NEXT_STEPS_RETRY} To avoid this in future, "
        f"add a backup provider with `hermes fallback add`.\n\nProvider said: {summary}"
    )


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
) -> str:
    """Chat copy for a terminal non-retryable rejection (auth, model missing, TLS, generic 4xx)."""
    label = provider_label_for(provider)
    if getattr(classified, "is_auth", False):
        from agent.error_surface import auth_kind

        template = _AUTH_COPY[auth_kind(str(provider or ""))]
    else:
        template = _NONRETRYABLE_COPY.get(classified.reason.value, _NONRETRYABLE_DEFAULT_COPY)
    prefix_hint = (
        f" If you typed the name yourself it may be missing its vendor prefix — did you mean "
        f"'{prefix_suggestion}'?"
        if prefix_suggestion else ""
    )
    body = template.format(label=label, model=model, home=display_hermes_home(), prefix_hint=prefix_hint,
                           relogin=oauth_relogin_command(provider))
    return f"{body}\n\nProvider said: {summary}"


def content_policy_copy(*, label: str, summary: str) -> str:
    return (
        f"{label}'s safety filter refused this request, so the model didn't answer. "
        f"{CONTENT_POLICY_NEXT_STEPS}\n\nProvider said: {summary}"
    )


def short_detail(exc: Any, limit: int = 200) -> str:
    """First line of an exception's text, capped, for a trailing ``Details:`` line."""
    text = (str(exc) or type(exc).__name__).strip().splitlines()
    first = text[0] if text else type(exc).__name__
    return first if len(first) <= limit else first[: limit - 1] + "…"


class _Defaults(dict):
    def __missing__(self, key: str) -> str:
        return ""
