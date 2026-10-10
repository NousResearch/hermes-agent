"""Plain-language copy for the one-line cron failure notice delivered to a job's chat.

The scheduler classifies the failure text through ``agent.error_classifier.classify_api_error``
(one classifier for the whole app, no cron-local regex ladder) and looks the verdict up here.
Every notice says what happened and where to inspect it. Generic failures name only the
profile-scoped history command; classified failures retain their specific recovery action.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from hermes_constants import display_hermes_home, profile_cli_selector


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
        "`{cron} edit {job_id} --provider <name>`."
    ),
    "auth": (
        "Sign in again with /login (or `{relogin}` in a terminal), or pin a "
        "working provider with `{cron} edit {job_id} --provider <name>`, then "
        "`{cron} run {job_id}` to retry."
    ),
    "model_not_found": "Pick another model with `{cron} edit {job_id} --model <name>`.",
    "upstream_blocked": (
        "A firewall in front of the provider blocked the request (not your key): set a User-Agent "
        "via `extra_headers` on the provider's custom_providers entry, or pin another provider with "
        "`{cron} edit {job_id} --provider <name>`."
    ),
    "context_overflow": "Shorten the job's prompt with `{cron} edit {job_id} --prompt <text>`.",
}
_PROVIDER_FAILURE_ACTION["auth_permanent"] = _PROVIDER_FAILURE_ACTION["auth"]
_PROVIDER_FAILURE_ACTION["billing_unverified"] = _PROVIDER_FAILURE_ACTION["billing"]
_PROVIDER_FAILURE_ACTION["payload_too_large"] = _PROVIDER_FAILURE_ACTION["context_overflow"]
_PROVIDER_FAILURE_ACTION["content_policy_blocked"] = (
    "Reword the job's prompt with `{cron} edit {job_id} --prompt <text>`, or pick another "
    "model with `{cron} edit {job_id} --model <name>`."
)
_PROVIDER_FAILURE_ACTION["provider_policy_blocked"] = (
    "Retrying won't help: check the account's status and data/privacy settings with the provider, "
    "or pin another model with `{cron} edit {job_id} --model <name>`."
)
_DEFAULT_FAILURE_ACTION = "Run it again with `{cron} run {job_id}`, or edit it with `{cron} edit {job_id}`."


def provider_failure_notice(
    job_name: str, job_id: str, reason: str, *, backup_provider_phrase: str, provider: Any = None,
) -> Optional[str]:
    """The notice for a provider-shaped ``reason``, or None when the reason is not one.
    ``provider`` is the job's pinned slug (if any) so the auth action names its exact sign-in."""
    cron = f"hermes {profile_cli_selector()}cron"
    cause = _provider_failure_cause(reason)
    if cause is None:
        return None
    if reason in _TRANSIENT_REASONS:
        action = (
            f"{backup_provider_phrase} It will run again at its next scheduled time; "
            f"`{cron} run {job_id}` tries now."
        )
    else:
        from agent.turn_failure_copy import relogin_command_hint

        action = _PROVIDER_FAILURE_ACTION.get(reason, _DEFAULT_FAILURE_ACTION).format(
            job_id=job_id, cron=cron, relogin=relogin_command_hint(provider))
    return (
        f"⚠️ Cron '{job_name}' failed: {cause}. {action} "
        f"Run log: `{cron} runs {job_id}`."
    )


def generic_failure_notice(job_name: str, job_id: str, error: str) -> str:
    """Short diagnostic and one read-only history pointer; full errors stay in run output."""
    cleaned = re.sub(r"^(RuntimeError|Exception|ValueError|HTTPStatusError):\s*", "", error[:2000])
    cleaned = re.sub(
        r"^Script (?:exited with code -?\d+\s+(?:stderr|stdout):\s*|execution failed:\s*)(?=\S)",
        "", cleaned,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(".")
    if len(cleaned) > 180:
        cleaned = cleaned[:177].rstrip() + "..."
    cron = f"hermes {profile_cli_selector()}cron"
    return (
        f"⚠️ Cron '{job_name}' failed\n"
        f"{cleaned}.\n"
        f"Details: `{cron} runs {job_id}`."
    )


def script_timeout_notice(job_name: str, job_id: str) -> str:
    cron = f"hermes {profile_cli_selector()}cron"
    return (
        f"⚠️ Cron '{job_name}' failed: its script timed out. No model was invoked. "
        f"Check the script's output under {cron_output_dir_display(job_id)} or `{cron} runs {job_id}`, "
        f"then run it again with `{cron} run {job_id}`."
    )


def inactivity_notice(job_name: str, job_id: str) -> str:
    cron = f"hermes {profile_cli_selector()}cron"
    return (
        f"⚠️ Cron '{job_name}' failed: the job stalled — it stopped doing anything for too long "
        f"and was cut off. Check what it was doing in the saved output under "
        f"{cron_output_dir_display(job_id)} (`{cron} runs {job_id}`), then run it again with "
        f"`{cron} run {job_id}`."
    )


def blocked_config_notice(job_name: str, reason: str) -> str:
    """One-time notice when the pre-run configuration check refused to start the job."""
    cron = f"hermes {profile_cli_selector()}cron"
    reason = reason.rstrip()
    if reason and reason[-1] not in ".!?":
        reason += "."
    return (
        f"⛔ Cron '{job_name}' did not run: {reason} Nothing was charged. Hermes will try again at "
        "the next scheduled time and will not repeat this alert; check with "
        f"`{cron} doctor`."
    )


def _failure_streak_nudge(job: dict) -> str:
    """Review nudge when a recurring job keeps failing, else "". The failure message is delivered
    BEFORE mark_job_run records this run, hence stored ``failure_streak`` + 1. Threshold:
    ``cron.failure_nudge_threshold`` (default 3, 0 disables)."""
    from cron.scheduler import load_config

    schedule_kind = (job.get("schedule") or {}).get("kind")
    if schedule_kind not in {"cron", "interval"}:
        return ""
    try:
        cfg = load_config() or {}
        threshold = int(
            ((cfg.get("cron") or {}) if isinstance(cfg, dict) else {}).get(
                "failure_nudge_threshold", 3
            )
        )
    except Exception:
        threshold = 3
    if threshold <= 0:
        return ""
    streak = int(job.get("failure_streak") or 0) + 1  # +1 = this run
    if streak < threshold:
        return ""
    return (
        f"\nConsecutive failures: {streak}. Scheduled retries continue; "
        f"pause with `hermes {profile_cli_selector()}cron pause {job['id']}` while repairing the cause."
    )
