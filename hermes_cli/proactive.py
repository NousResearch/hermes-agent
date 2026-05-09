"""Safe proactive reflection helpers for Hermes.

This module intentionally does not make the agent proactive by itself. It builds
and installs a cron prompt with strong safety boundaries so users can opt in to
periodic synthesis without granting the job permission to act externally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Literal

from cron.jobs import create_job, list_jobs, update_job

DEFAULT_JOB_NAME = "Proactive synthesis / safe nudges"
DEFAULT_SCHEDULE = "0 9 * * *"
DEFAULT_DELIVER = "local"
DEFAULT_ENABLED_TOOLSETS = ["session_search", "memory", "todo"]
_ALLOWED_CONFIDENCE = {"medium", "high"}


@dataclass(frozen=True)
class ProactivePromptOptions:
    lookback_days: int = 7
    max_sessions: int = 30
    min_confidence: Literal["medium", "high"] = "high"

    def normalized(self) -> "ProactivePromptOptions":
        lookback_days = max(1, min(int(self.lookback_days), 90))
        max_sessions = max(1, min(int(self.max_sessions), 200))
        min_confidence = str(self.min_confidence or "high").lower()
        if min_confidence not in _ALLOWED_CONFIDENCE:
            min_confidence = "high"
        return ProactivePromptOptions(
            lookback_days=lookback_days,
            max_sessions=max_sessions,
            min_confidence=min_confidence,  # type: ignore[arg-type]
        )


def build_reflection_prompt(
    *,
    lookback_days: int = 7,
    max_sessions: int = 30,
    min_confidence: str = "high",
) -> str:
    """Build the self-contained prompt used by the proactive cron job.

    The prompt is deliberately conservative: it asks Hermes to synthesize recent
    interactions, but to return ``[SILENT]`` unless there is one concrete,
    safe, timely thing worth sending to the user.
    """

    opts = ProactivePromptOptions(
        lookback_days=lookback_days,
        max_sessions=max_sessions,
        min_confidence=min_confidence,  # type: ignore[arg-type]
    ).normalized()

    return f"""You are Hermes running a safe proactive reflection pass.

Purpose:
- Synthesize Charles/user interactions from the last {opts.lookback_days} days across recent sessions.
- Look for one useful, timely, low-risk proactive nudge Hermes should send without waiting to be asked.
- Use session_search first: browse recent sessions, then search targeted terms if needed. Review up to {opts.max_sessions} relevant sessions.

Safety policy:
- Send at most one proactive message.
- Only message when you have {opts.min_confidence} confidence that it is useful, timely, and wanted.
- Prefer tiny, concrete nudges: a dropped ball, a blocker, a follow-up the user asked for, a safe reminder, or a concise synthesis that reduces cognitive load.
- Do not send emails, posts, DMs, calendar invites, payments, public messages, file deletes, or irreversible/external actions.
- Ask before any action involving money, reputation, external recipients, calendars with other people, destructive changes, or private data sharing.
- Do not expose private transcript details, secrets, tokens, credentials, customer data, or internal paths.
- Do not nag about paused/on-hold work, stale ambitions, or low-confidence guesses.
- If the user has recently said not to be nudged about something, treat that as binding.
- If a proactive message would be longer than a short text, compress it to one action and offer "More Info".

Output rules:
- If there is nothing worth sending, start your final response with exactly: [SILENT]
- If there is something worth sending, send only the user-facing message. No audit log, no analysis, no wrapper.
- Keep it under 80 words unless the situation is urgent.
- Include one obvious next action when possible.
""".strip()


def _find_existing_job() -> Dict[str, Any] | None:
    for job in list_jobs(include_disabled=True):
        if job.get("name") == DEFAULT_JOB_NAME:
            return job
    return None


def install_proactive_job(
    *,
    schedule: str = DEFAULT_SCHEDULE,
    deliver: str = DEFAULT_DELIVER,
    lookback_days: int = 7,
    max_sessions: int = 30,
    min_confidence: str = "high",
    paused: bool = False,
) -> Dict[str, Any]:
    """Create or update the built-in proactive synthesis cron job.

    Returns a small report with ``action`` (created/updated/created_paused) and
    the resulting job dict. The job is idempotent by name so repeated installs
    tune the same schedule/prompt instead of creating duplicates.
    """

    prompt = build_reflection_prompt(
        lookback_days=lookback_days,
        max_sessions=max_sessions,
        min_confidence=min_confidence,
    )
    existing = _find_existing_job()
    updates = {
        "schedule": schedule,
        "prompt": prompt,
        "name": DEFAULT_JOB_NAME,
        "deliver": deliver,
        "enabled_toolsets": list(DEFAULT_ENABLED_TOOLSETS),
    }

    if existing:
        job = update_job(existing["id"], updates)
        action = "updated"
    else:
        job = create_job(
            prompt=prompt,
            schedule=schedule,
            name=DEFAULT_JOB_NAME,
            deliver=deliver,
            enabled_toolsets=list(DEFAULT_ENABLED_TOOLSETS),
        )
        action = "created"

    if paused and job:
        job = update_job(
            job["id"],
            {
                "enabled": False,
                "state": "paused",
                "paused_reason": "created paused for review",
            },
        )
        action = "created_paused" if action == "created" else "updated_paused"

    return {"action": action, "job": job}


def _print_report(report: Dict[str, Any], *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    action = report.get("action", "ok")
    job = report.get("job") or {}
    print(f"Proactive synthesis job {action}.")
    if job:
        print(f"  ID: {job.get('id') or job.get('job_id')}")
        print(f"  Name: {job.get('name')}")
        print(f"  Schedule: {job.get('schedule_display') or job.get('schedule')}")
        print(f"  Deliver: {job.get('deliver')}")
        print(f"  State: {job.get('state')}")


def cmd_proactive(args) -> int:
    """CLI entrypoint for ``hermes proactive``."""

    subcmd = getattr(args, "proactive_command", None) or "prompt"
    if subcmd == "prompt":
        prompt = build_reflection_prompt(
            lookback_days=getattr(args, "lookback_days", 7),
            max_sessions=getattr(args, "max_sessions", 30),
            min_confidence=getattr(args, "min_confidence", "high"),
        )
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "lookback_days": max(1, min(int(getattr(args, "lookback_days", 7)), 90)),
                        "max_sessions": max(1, min(int(getattr(args, "max_sessions", 30)), 200)),
                        "min_confidence": str(getattr(args, "min_confidence", "high") or "high").lower(),
                        "prompt": prompt,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(prompt)
        return 0

    if subcmd == "install":
        report = install_proactive_job(
            schedule=getattr(args, "schedule", DEFAULT_SCHEDULE),
            deliver=getattr(args, "deliver", DEFAULT_DELIVER),
            lookback_days=getattr(args, "lookback_days", 7),
            max_sessions=getattr(args, "max_sessions", 30),
            min_confidence=getattr(args, "min_confidence", "high"),
            paused=getattr(args, "paused", False),
        )
        _print_report(report, as_json=getattr(args, "json", False))
        return 0

    print("Usage: hermes proactive [prompt|install]")
    return 2
