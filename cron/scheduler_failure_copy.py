"""Plain-language copy for the one-line cron failure notice delivered to a job's chat.

The scheduler classifies the failure text through ``agent.error_classifier.classify_api_error``
(one classifier for the whole app, no cron-local regex ladder) and looks the verdict up here.
Every notice says WHAT happened and WHAT TO DO, and names the exact ``hermes cron`` command plus
the real output directory — "cron output" alone sent operators hunting.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agent.i18n import t
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

# Reason -> catalog key of what to do, resolved per call so the notice follows the active language.
# Transient reasons get the backup-provider clause from the scheduler (it knows whether a fallback
# chain is configured) instead of a fixed sentence.
_PROVIDER_FAILURE_ACTION_KEY: dict[str, str] = {
    "billing": "gateway.cron.failure.action_billing",
    "billing_unverified": "gateway.cron.failure.action_billing",
    "auth": "gateway.cron.failure.action_auth",
    "auth_permanent": "gateway.cron.failure.action_auth",
    "model_not_found": "gateway.cron.failure.action_model_not_found",
    "upstream_blocked": "gateway.cron.failure.action_upstream_blocked",
    "context_overflow": "gateway.cron.failure.action_context_overflow",
    "payload_too_large": "gateway.cron.failure.action_context_overflow",
    "content_policy_blocked": "gateway.cron.failure.action_content_policy",
    "provider_policy_blocked": "gateway.cron.failure.action_provider_policy",
}
_DEFAULT_FAILURE_ACTION_KEY = "gateway.cron.failure.action_default"


def provider_failure_notice(
    job_name: str, job_id: str, reason: str, *, backup_provider_phrase: str, provider: Any = None,
) -> Optional[str]:
    """The notice for a provider-shaped ``reason``, or None when the reason is not one.
    ``provider`` is the job's pinned slug (if any) so the auth action names its exact sign-in."""
    cause = _provider_failure_cause(reason)
    if cause is None:
        return None
    if reason in _TRANSIENT_REASONS:
        action = t("gateway.cron.failure.action_transient", backup_phrase=backup_provider_phrase, job_id=job_id)
    else:
        from agent.turn_failure_copy import relogin_command_hint

        action = t(_PROVIDER_FAILURE_ACTION_KEY.get(reason, _DEFAULT_FAILURE_ACTION_KEY),
                   job_id=job_id, relogin=relogin_command_hint(provider))
    return t("gateway.cron.failure.provider", job_name=job_name, cause=cause, action=action, job_id=job_id)


def generic_failure_notice(job_name: str, job_id: str, cleaned_error: str) -> str:
    """Unclassified failure: the cleaned error text plus where to look and what to do."""
    return t("gateway.cron.failure.generic", job_name=job_name, error=cleaned_error, job_id=job_id,
             output_dir=cron_output_dir_display(job_id))


def script_timeout_notice(job_name: str, job_id: str) -> str:
    return t("gateway.cron.failure.script_timeout", job_name=job_name, job_id=job_id,
             output_dir=cron_output_dir_display(job_id))


def inactivity_notice(job_name: str, job_id: str) -> str:
    return t("gateway.cron.failure.inactivity", job_name=job_name, job_id=job_id,
             output_dir=cron_output_dir_display(job_id))


def blocked_config_notice(job_name: str, reason: str) -> str:
    """One-time notice when the pre-run configuration check refused to start the job."""
    reason = reason.rstrip()
    if reason and reason[-1] not in ".!?":
        reason += "."
    return t("gateway.cron.failure.blocked_config", job_name=job_name, reason=reason)
