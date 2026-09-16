"""Plain-language copy for the one-line cron failure notice delivered to a job's chat.

The scheduler classifies the failure text through ``agent.error_classifier.classify_api_error``
(one classifier for the whole app, no cron-local regex ladder) and looks the verdict up here.
Every notice says WHAT happened and WHAT TO DO, and names the exact ``hermes cron`` command plus
the real output directory — "cron output" alone sent operators hunting.
"""

from __future__ import annotations

import re
from typing import Optional

from hermes_constants import display_hermes_home


def cron_output_dir_display(job_id: str) -> str:
    """User-facing path of a job's saved run output (profile-aware)."""
    return f"{display_hermes_home()}/cron/output/{job_id}/"


_HTTP_STATUS_IN_TEXT = re.compile(r"(?:\bHTTP\b|\bError code\b|\bstatus(?: code)?\b)\W{0,3}(\b[45]\d\d\b)", re.I)
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
        "Sign in again with /login (or `hermes auth add <provider>` in a terminal), or pin a "
        "working provider with `hermes cron edit {job_id} --provider <name>`, then "
        "`hermes cron run {job_id}` to retry."
    ),
    "model_not_found": "Pick another model with `hermes cron edit {job_id} --model <name>`.",
    "context_overflow": "Shorten the job's prompt with `hermes cron edit {job_id} --prompt <text>`.",
}
_PROVIDER_FAILURE_ACTION["auth_permanent"] = _PROVIDER_FAILURE_ACTION["auth"]
_PROVIDER_FAILURE_ACTION["billing_unverified"] = _PROVIDER_FAILURE_ACTION["billing"]
_PROVIDER_FAILURE_ACTION["payload_too_large"] = _PROVIDER_FAILURE_ACTION["context_overflow"]
_PROVIDER_FAILURE_ACTION["content_policy_blocked"] = (
    "Reword the job's prompt with `hermes cron edit {job_id} --prompt <text>`, or pick another "
    "model with `hermes cron edit {job_id} --model <name>`."
)
_DEFAULT_FAILURE_ACTION = "Run it again with `hermes cron run {job_id}`, or edit it with `hermes cron edit {job_id}`."


def provider_failure_notice(
    job_name: str, job_id: str, reason: str, *, backup_provider_phrase: str,
) -> Optional[str]:
    """The notice for a provider-shaped ``reason``, or None when the reason is not one."""
    cause = _provider_failure_cause(reason)
    if cause is None:
        return None
    if reason in _TRANSIENT_REASONS:
        action = (
            f"{backup_provider_phrase} It will run again at its next scheduled time; "
            f"`hermes cron run {job_id}` tries now."
        )
    else:
        action = _PROVIDER_FAILURE_ACTION.get(reason, _DEFAULT_FAILURE_ACTION).format(job_id=job_id)
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
        "the next scheduled time once this is fixed and will not repeat this alert; check with "
        "`hermes cron doctor`."
    )
