"""Plain-language copy for the cron failure notice delivered to a job's chat.

The scheduler classifies the failure text through ``agent.error_classifier.classify_api_error``
(one classifier for the whole app, no cron-local regex ladder) and looks the verdict up here.
Every notice says WHAT happened and names the one command that shows the run,
``hermes cron runs <job_id>`` -- which is also where the saved output path lives, so the
notice itself no longer spells the directory out.

Copy in this module is delivered to EVERY platform a job can target, including
plain-text-only sinks (ntfy, email) that render markdown literally. Keep it free of
``**bold**``, backticks around prose and other markup: the emoji prefix and a line break
are the only emphasis primitives that survive the whole fan-out.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# One prefix for the whole failure family so a chat-side filter can match them all.
_FAILURE_PREFIX = "⚠️ Cron"


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
        from agent.turn_failure_copy import relogin_command_hint

        action = _PROVIDER_FAILURE_ACTION.get(reason, _DEFAULT_FAILURE_ACTION).format(
            job_id=job_id, relogin=relogin_command_hint(provider))
    return (
        f"⚠️ Cron '{job_name}' failed: {cause}. {action} "
        f"Run log: `hermes cron runs {job_id}`."
    )


def _strip_job_name_echo(job_name: str, text: str) -> str:
    """Drop a ``<job name>:`` prefix that a script echoed into its own stderr.

    The notice header already names the job, so the echo would print it twice.

    Two traps this has to survive. The echo is NOT at index 0: the scheduler collapses
    ``Script exited with code 1\\nstderr:\\n<name>: ...`` into one line before the notice is
    built, so the prefix sits mid-string behind ``stderr:``. And the match must be caseless
    while the cut comes from the match on the ORIGINAL text -- slicing by ``len(job_name)``
    corrupts names whose ``lower()`` changes length (U+0130 folds to two code points),
    which silently ate a character of the error.
    """
    name = (job_name or "").strip()
    if not name or not text:
        return text
    # Anchored at the start or just after the "stderr:"/"stdout:" label the runner adds.
    pattern = rf"(?i)(?:(?<=^)|(?<=stderr:\s)|(?<=stdout:\s)){re.escape(name)}\s*:\s*"
    stripped = re.sub(pattern, "", text, count=1)
    # An error consisting of nothing but the echo would leave the notice bodyless;
    # keep the original text in that case so the alert still says something.
    return stripped if stripped.strip() else text


def generic_failure_notice(job_name: str, job_id: str, cleaned_error: str) -> str:
    """Unclassified failure: the cleaned error text plus the command to inspect it."""
    err = _strip_job_name_echo(job_name, cleaned_error.strip()).strip()
    return (
        f"{_FAILURE_PREFIX} {job_name} failed: {err or 'no error text captured'}\n"
        f"Run log: `hermes cron runs {job_id}`"
    )


def script_timeout_notice(job_name: str, job_id: str) -> str:
    return (
        f"{_FAILURE_PREFIX} {job_name} failed: script timed out; no model was invoked\n"
        f"Run log: `hermes cron runs {job_id}`"
    )


def inactivity_notice(job_name: str, job_id: str) -> str:
    return (
        f"{_FAILURE_PREFIX} {job_name} failed: stalled -- stopped responding and was cut off\n"
        f"Run log: `hermes cron runs {job_id}`"
    )


def blocked_config_notice(job_name: str, reason: str) -> str:
    """One-time notice when the pre-run configuration check refused to start the job."""
    reason = (reason or "").rstrip()
    # Strip ONE trailing period, never an ellipsis: "..." marks truncated text and
    # rstrip(".") would delete the marker along with it.
    if reason.endswith(".") and not reason.endswith("..."):
        reason = reason[:-1]
    return (
        f"⛔ Cron '{job_name}' did not run: {reason or 'configuration check failed'}. "
        f"Nothing was charged and this alert is sent once; check with `hermes cron doctor`."
    )
