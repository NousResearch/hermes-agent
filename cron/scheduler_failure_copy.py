"""Plain-language copy for the one-line cron failure notice delivered to a job's chat.

Executor source-stage verdicts bypass provider text classification. Legacy/provider failures
still use ``agent.error_classifier.classify_api_error`` (no cron-local regex ladder).
Every notice says WHAT happened and WHAT TO DO, and names the exact ``hermes cron`` command plus
the real output directory — "cron output" alone sent operators hunting.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from hermes_constants import display_hermes_home


def cron_output_dir_display(job_id: str) -> str:
    """User-facing path of a job's saved run output (profile-aware)."""
    return f"{display_hermes_home()}/cron/output/{job_id}/"


_HTTP_STATUS_IN_TEXT = re.compile(r"(?:\bHTTP\b|\bError code\b|\bstatus(?: code)?\b)\W{0,3}(\b[45]\d\d\b)", re.IGNORECASE)
_LEADING_EXC_TYPE = re.compile(r"^(?:[\w.]+\.)?([A-Z]\w*(?:Error|Timeout|Exception))\s*:")


def classify_cron_failure_reason(text: str) -> str:
    """``FailoverReason`` value for a cron failure string (``"unknown"`` when unclassifiable).

    The scheduler only has ``str(exc)``, so the status code and exception type the classifier
    keys on are rebuilt from the text: a whole-token HTTP code after ``HTTP`` / ``Error code`` /
    ``status`` (a bare ``429`` inside a job id or hash never counts, #83188) and a leading
    ``ReadTimeout:``-style type prefix."""
    from agent.error_classifier import classify_api_error

    status = _HTTP_STATUS_IN_TEXT.search(text)
    type_name = _LEADING_EXC_TYPE.match(text)
    exc_cls = type(type_name.group(1), (Exception,), {}) if type_name else Exception
    exc = exc_cls(text)
    if status:
        exc.status_code = int(status.group(1))
    return classify_api_error(exc).reason.value


# What happened, per reason: the one gloss table shared with subagent notices lives in
# agent/turn_failure_copy.py so the two never drift; the job is the subject here.
def _provider_failure_cause(reason: str) -> Optional[str]:
    from agent.turn_failure_copy import failure_cause_gloss

    return failure_cause_gloss(reason, subject="this job", possessive="the job's")


_TRANSIENT_REASONS = frozenset({"timeout", "rate_limit", "upstream_rate_limit", "overloaded", "server_error"})

# Reason -> what to do. Transient reasons get the backup-provider clause from the scheduler
# (it knows whether a fallback chain is configured) instead of a fixed sentence.
_PROVIDER_FAILURE_ACTION: dict[str, str] = {
    "billing": (
        "Top up or wait for the limit to reset, or pin another provider with "
        "`hermes cron edit {job_id} --provider <name>`."
    ),
    "auth": (
        "Sign in again with /login (or `{relogin}` in a terminal), or pin a "
        "working provider with `hermes cron edit {job_id} --provider <name>`, then "
        "`hermes cron run {job_id}` to retry."
    ),
    "model_not_found": "Pick another model with `hermes cron edit {job_id} --model <name>`.",
    "upstream_blocked": (
        "A firewall in front of the provider blocked the request (not your key): set a User-Agent "
        "via `extra_headers` on the provider's custom_providers entry, or pin another provider with "
        "`hermes cron edit {job_id} --provider <name>`."
    ),
    "context_overflow": "Shorten the job's prompt with `hermes cron edit {job_id} --prompt <text>`.",
}
_PROVIDER_FAILURE_ACTION["auth_permanent"] = _PROVIDER_FAILURE_ACTION["auth"]
_PROVIDER_FAILURE_ACTION["billing_unverified"] = _PROVIDER_FAILURE_ACTION["billing"]
_PROVIDER_FAILURE_ACTION["payload_too_large"] = _PROVIDER_FAILURE_ACTION["context_overflow"]
_PROVIDER_FAILURE_ACTION["content_policy_blocked"] = (
    "Reword the job's prompt with `hermes cron edit {job_id} --prompt <text>`, or pick another "
    "model with `hermes cron edit {job_id} --model <name>`."
)
_PROVIDER_FAILURE_ACTION["provider_policy_blocked"] = (
    "Retrying won't help: check the account's status and data/privacy settings with the provider, "
    "or pin another model with `hermes cron edit {job_id} --model <name>`."
)
_PROVIDER_FAILURE_ACTION["guardrail_blocked"] = (
    "The provider's prompt-injection guardrail blocked the request (not your key; tool output is "
    "scanned too): set it to Redact or Flag in the provider's settings (OpenRouter: Workspaces → "
    "Guardrails), or pin another provider with `hermes cron edit {job_id} --provider <name>`."
)
_DEFAULT_FAILURE_ACTION = "Run it again with `hermes cron run {job_id}`, or edit it with `hermes cron edit {job_id}`."


def provider_failure_notice(
    job_name: str, job_id: str, reason: str, *, backup_provider_phrase: str, provider: Any = None,
) -> Optional[str]:
    """The notice for a provider-shaped ``reason``, or None when the reason is not one.
    ``provider`` is the job's pinned slug (if any) so the auth action names its exact sign-in."""
    cause = _provider_failure_cause(reason)
    if cause is None:
        return None
    if reason in _TRANSIENT_REASONS:
        action = (
            f"{backup_provider_phrase} It will run again at its next scheduled time; "
            f"`hermes cron run {job_id}` tries now."
        )
    else:
        template = _PROVIDER_FAILURE_ACTION.get(reason, _DEFAULT_FAILURE_ACTION)
        relogin = ""
        if "{relogin}" in template:  # only the auth copy names a command; skip the lookup otherwise
            from agent.turn_failure_copy import relogin_command_hint

            relogin = relogin_command_hint(provider)
        action = template.format(job_id=job_id, relogin=relogin)
    return (
        f"⚠️ Cron '{job_name}' failed: {cause}. {action} "
        f"Run log: `hermes cron runs {job_id}`."
    )


def generic_failure_notice(job_name: str, job_id: str, cleaned_error: str) -> str:
    """Unclassified failure: the cleaned error text plus where to look and what to do."""
    return (
        f"⚠️ Cron '{job_name}' failed: {cleaned_error}. "
        f"See the full run with `hermes cron runs {job_id}` (output saved under "
        f"{cron_output_dir_display(job_id)}); run it again with `hermes cron run {job_id}`, "
        f"edit it with `hermes cron edit {job_id}`, or pause it with `hermes cron pause {job_id}`."
    )


def script_timeout_notice(job_name: str, job_id: str) -> str:
    return (
        f"⚠️ Cron '{job_name}' failed: its script timed out. No model was invoked. "
        f"Check the script's output under {cron_output_dir_display(job_id)} or `hermes cron runs {job_id}`, "
        f"then run it again with `hermes cron run {job_id}`."
    )


def inactivity_notice(job_name: str, job_id: str) -> str:
    return (
        f"⚠️ Cron '{job_name}' failed: the job stalled — it stopped doing anything for too long "
        f"and was cut off. Check what it was doing in the saved output under "
        f"{cron_output_dir_display(job_id)} (`hermes cron runs {job_id}`), then run it again with "
        f"`hermes cron run {job_id}`."
    )


def blocked_config_notice(job_name: str, reason: str) -> str:
    """One-time notice when the pre-run configuration check refused to start the job."""
    reason = reason.rstrip()
    if reason and reason[-1] not in ".!?":
        reason += "."
    return (
        f"⛔ Cron '{job_name}' did not run: {reason} Nothing was charged. Hermes will try again at "
        "the next scheduled time and will not repeat this alert; check with "
        "`hermes cron doctor`."
    )


# --- Per-run failure origin ---------------------------------------------------------------------
# The executor stamps WHICH stage failed on the job's WORKING copy; delivery and the incident
# ledger classify from this origin instead of guessing at error text. ``agent_provider`` means the
# agent/provider setup or execution failed — NOT that a model necessarily received a request.
FAILURE_ORIGIN_MONITOR = "monitor_source"
FAILURE_ORIGIN_SCRIPT = "script_execution"
FAILURE_ORIGIN_AGENT = "agent_provider"
# Source stages stop before Hermes's agent; an independent script may call its own model.
_SOURCE_FAILURE_ORIGINS = frozenset({FAILURE_ORIGIN_MONITOR, FAILURE_ORIGIN_SCRIPT})


def source_failure_origin(job: dict) -> Optional[str]:
    """The bounded source-stage verdict, or None for absent/malformed legacy metadata."""
    origin = job.get("_failure_origin")
    return origin if isinstance(origin, str) and origin in _SOURCE_FAILURE_ORIGINS else None


def source_failure_notice(job_name: str, job_id: str, *, kind: str) -> str:
    """Content-free notice for a monitor-source or script failure.

    Names the failing stage and the command to inspect the LOCAL run log; never echoes the source's
    stdout/stderr, an authenticated URL or arbitrary error text (those stay in the saved run output).
    """
    what = "monitored source" if kind == FAILURE_ORIGIN_MONITOR else "script"
    return (
        f"⚠️ Cron '{job_name}' failed: its {what} did not run successfully. "
        f"Hermes did not enter its agent stage. Inspect the full run with "
        f"`hermes cron runs {job_id}` (output saved under {cron_output_dir_display(job_id)}), "
        f"then run it again with `hermes cron run {job_id}`."
    )


def summarize_cron_failure(
    job: dict, error: Optional[str], *, fallback_phrase, code_skew_probe,
) -> str:
    """One-line failure notice for chat delivery (full details stay in the run output).

    Deterministic scheduler/script shapes are matched first (their text can contain "timed out"
    and would otherwise be blamed on the model service); a known SOURCE origin then wins over any
    error-text reading; everything else goes through the shared ``classify_api_error`` verdict and
    the copy table above. ``fallback_phrase`` / ``code_skew_probe`` are the scheduler's lazily
    evaluated backup-provider and diagnostic callbacks, passed in so this renderer stays free
    of scheduler-side imports.
    """
    job_name = job.get("name") or job.get("id") or "cron job"
    job_id = job.get("id") or job_name
    text = (error or "unknown error").strip()
    lower = text.lower()

    # Script runner contract ("Script timed out after {n}s: {path}") — also for agent jobs with a
    # context script. Must precede origin / provider classification so it never claims a model
    # failure. See #78503, #82460.
    if lower.startswith("script timed out"):
        return script_timeout_notice(job_name, job_id)

    # An executor source verdict outranks source text that resembles a watchdog/provider error.
    origin = source_failure_origin(job)
    if origin is not None:
        return source_failure_notice(job_name, job_id, kind=origin)

    # Scheduler inactivity watchdog ("idle for {n}s (limit {m}s)"): the job's OWN tool call went
    # quiet, no model service involved. Its text may still contain "timed out", so it must be
    # recognised before the classifier (field-reported: a stuck `terminal` call was blamed on the
    # provider and the operator debugged the wrong system).
    if re.search(r"idle for \d+s\s*\(limit \d+s\)", lower):
        return inactivity_notice(job_name, job_id)

    # no_agent jobs never reach a model, so provider errors are structurally impossible for them:
    # gate on job MODE before classifying, or a script's own wording ("429", "timed out") would
    # blame the wrong subsystem.
    if not job.get("no_agent"):
        notice = provider_failure_notice(
            job_name, job_id, classify_cron_failure_reason(text),
            backup_provider_phrase=fallback_phrase(), provider=job.get("provider"))
        if notice is not None:
            return notice

    # Strip exception wrappers; bound input first so a multi-KB blob can't slow the regexes.
    cleaned = re.sub(r"^(RuntimeError|Exception|ValueError|HTTPStatusError):\s*", "", text[:2000])
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(".")
    if len(cleaned) > 180:
        cleaned = cleaned[:177].rstrip() + "..."
    message = generic_failure_notice(job_name, job_id, cleaned)

    # Import-class failures (#95294 part 3): a long-lived gateway whose checkout was updated
    # underneath it (interrupted `hermes update`, manual git pull) serves MIXED modules and every
    # agent cron job dies with `cannot import name X`. The error reads like a code bug, so APPEND
    # cause + fix — never replace the raw error, which carries the failing symbol. Fail-safe: skew
    # is None on non-git/no-fingerprint; no_agent jobs excluded (a fresh subprocess resolves
    # imports against disk, so its ImportError is the script's own problem).
    if not job.get("no_agent") and re.search(
        r"cannot import name|modulenotfounderror|importerror", lower
    ):
        try:
            skew = code_skew_probe()
        except Exception:
            logging.getLogger(__name__).debug("Cron code-skew diagnostics failed", exc_info=True)
            skew = None  # delivery must never die on a diagnostics probe
        if skew is not None:
            boot_rev, disk_rev = skew
            message += (
                f" Likely cause: the gateway is running stale code (booted "
                f"on {boot_rev}, disk is at {disk_rev}) — run "
                "`hermes gateway restart` to fix it."
            )

    return message
