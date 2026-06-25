"""Calendar alignment audit rendering for Torben."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from .action_ledger import parse_time

LOCAL_TZ = ZoneInfo("America/New_York")


def _format_window(start_at: str | None, end_at: str | None) -> str:
    start = parse_time(start_at)
    end = parse_time(end_at)
    if not start or not end:
        return "time unknown"
    start_local = start.astimezone(LOCAL_TZ)
    end_local = end.astimezone(LOCAL_TZ)
    if start_local.date() == end_local.date():
        return f"{start_local:%a %-m/%-d %-I:%M %p}-{end_local:%-I:%M %p}"
    return f"{start_local:%a %-m/%-d %-I:%M %p} to {end_local:%a %-m/%-d %-I:%M %p}"


def render_calendar_alignment_audit(payload: dict[str, Any], *, max_items: int = 15) -> str:
    ea = payload.get("ea") or {}
    diagnostics = ((payload.get("source_diagnostics") or {}).get("google") or {})
    audit = diagnostics.get("audit") or {}
    candidates = list(ea.get("calendar_block_candidates") or [])
    accounts = list(diagnostics.get("accounts") or [])
    policy = ea.get("calendar_alignment_policy") or {}
    generated_at = audit.get("generated_at") or datetime.now(LOCAL_TZ).isoformat()

    lines = [
        f"Torben / Calendar Alignment Audit / {generated_at}",
        "",
        (
            f"Checked {len(accounts)} account(s), {diagnostics.get('calendar_events_collected', 0)} event(s), "
            f"and found {len(candidates)} calendar drift candidate(s)."
        ),
        (
            f"Google reads: {audit.get('google_read_api_calls', 0)}. "
            f"Google writes: {audit.get('google_write_api_calls', 0)}. "
            f"External mutations: {audit.get('external_mutations', 0)}."
        ),
    ]

    warnings = list(audit.get("warnings") or [])
    if warnings:
        lines.extend(["", "Source gaps:"])
        for warning in warnings[:5]:
            lines.append(f"- {warning}")

    if candidates:
        lines.extend(["", "Priority drift:"])
        for index, candidate in enumerate(candidates[:max_items], start=1):
            targets = ", ".join(candidate.get("target_accounts") or [])
            already = ", ".join(candidate.get("already_blocked_accounts") or []) or "none"
            lines.append(
                (
                    f"{index}. {_format_window(candidate.get('start_at'), candidate.get('end_at'))}: "
                    f"{candidate.get('summary') or 'Busy'} on {candidate.get('source_account') or 'source'} "
                    f"is missing on {targets or 'target calendars'}; already blocked on {already}."
                )
            )
        if len(candidates) > max_items:
            lines.append(f"...and {len(candidates) - max_items} more in the JSON artifact.")
    else:
        lines.extend(["", "No drift candidates found in this window."])

    lines.extend(
        [
            "",
            "Automation rule to establish:",
            policy.get("rule")
            or "Stage busy blocks on missing calendars when a non-transparent event appears on one enabled account.",
            policy.get("mutation_boundary")
            or "No Google Calendar events are created until an approval action is resolved.",
        ]
    )
    return "\n".join(line.rstrip() for line in lines).strip() + "\n"
