"""Status dashboards, Markdown reports, human-task digest, and Google Sheets row export."""
from __future__ import annotations

import brokers as brokers_mod
import ledger as ledger_mod

STATE_LABELS = {
    "new": "Not started",
    "searching": "Searching",
    "not_found": "Not found",
    "found": "Found (action needed)",
    "indirect_exposure": "Indirect exposure (PII on a relative's record)",
    "action_selected": "Action selected",
    "submitted": "Submitted",
    "verification_pending": "Awaiting verification",
    "awaiting_processing": "Processing",
    "confirmed_removed": "Removed",
    "reappeared": "Reappeared",
    "human_task_queued": "Human task",
    "blocked": "Blocked",
}


def status_counts(subject_id: str) -> dict:
    counts: dict[str, int] = {}
    for case in ledger_mod.load(subject_id).values():
        state = case.get("state", "new")
        counts[state] = counts.get(state, 0) + 1
    return counts


def metrics(subject_id: str) -> dict:
    """Outcome metrics: what's actually confirmed vs merely claimed, and what's overdue.

    removal_rate is confirmed_removed over cases we actually acted on (found/submitted/... ),
    NOT over the whole broker DB, so it reflects real progress on real exposure. `in_flight`
    is 'claimed' (submitted/verifying/processing) but not yet re-scan-confirmed. `overdue`
    counts cases whose recheck window has already passed (the cron backlog).
    """
    c = status_counts(subject_id)
    removed = c.get("confirmed_removed", 0)
    in_flight = c.get("submitted", 0) + c.get("verification_pending", 0) + c.get("awaiting_processing", 0)
    open_found = c.get("found", 0) + c.get("reappeared", 0) + c.get("action_selected", 0) \
        + c.get("indirect_exposure", 0)
    acted = removed + in_flight + open_found + c.get("human_task_queued", 0) + c.get("blocked", 0)
    return {
        "confirmed_removed": removed,
        "in_flight_claimed": in_flight,      # submitted but NOT yet verified gone
        "open_needs_action": open_found,
        "blocked": c.get("blocked", 0),
        "human_tasks": c.get("human_task_queued", 0),
        "acted_total": acted,
        "removal_rate": round(removed / acted, 3) if acted else 0.0,
        "overdue_rechecks": len(ledger_mod.due(subject_id)),
    }


def render_markdown(subject_id: str) -> str:
    ledger = ledger_mod.load(subject_id)
    counts = status_counts(subject_id)
    total = sum(counts.values())
    removed = counts.get("confirmed_removed", 0)

    m = metrics(subject_id)
    lines = [
        f"# unbroker - status for `{subject_id}`",
        "",
        f"**{removed} / {total} confirmed removed**  ·  removal rate (of acted-on cases): "
        f"{int(m['removal_rate'] * 100)}%",
        "",
        f"- Confirmed removed: {m['confirmed_removed']}",
        f"- In flight (submitted, not yet re-scan-confirmed): {m['in_flight_claimed']}",
        f"- Open / needs action: {m['open_needs_action']}",
        f"- Blocked (anti-bot): {m['blocked']}  ·  Human tasks: {m['human_tasks']}",
        f"- Overdue rechecks (cron backlog): {m['overdue_rechecks']}",
        "",
        "| State | Count |",
        "|---|---|",
    ]
    for state in ledger_mod.STATES:
        if counts.get(state):
            lines.append(f"| {STATE_LABELS.get(state, state)} | {counts[state]} |")

    tasks = [c for c in ledger.values() if c.get("state") == "human_task_queued"]
    if tasks:
        lines += ["", "## Outstanding human tasks"]
        for c in tasks:
            reason = c.get("human_task_reason", "manual step required")
            lines.append(f"- **{c.get('broker_id')}** - {reason}")

    indirect = [c for c in ledger.values() if c.get("state") == "indirect_exposure"]
    if indirect:
        lines += ["", "## Indirect exposure (your PII on third-party records)",
                  "Not removable via the broker's self-service opt-out (the record is about someone "
                  "else). Lever: a targeted CCPA/GDPR delete-my-PII request naming only your own "
                  "identifiers."]
        for c in indirect:
            ev = c.get("evidence") or {}
            note = ev.get("summary") or "subject's identifiers appear on another person's listing"
            lines.append(f"- **{c.get('broker_id')}** - {note}")
    return "\n".join(lines) + "\n"


def human_tasks_markdown(subject_id: str) -> str:
    """ONE consolidated digest of everything that genuinely needs a human.

    The autonomous run accumulates human-only work silently (never interrupting);
    this digest is presented once, at the end, so the operator clears it in a
    single sitting. Includes queued tasks and blocked-site operator-browser checks.
    """
    ledger = ledger_mod.load(subject_id)
    tasks = [(bid, c) for bid, c in sorted(ledger.items()) if c.get("state") == "human_task_queued"]
    blocked = [(bid, c) for bid, c in sorted(ledger.items()) if c.get("state") == "blocked"]

    lines = [f"# Human tasks for `{subject_id}`", ""]
    if not tasks and not blocked:
        lines.append("Nothing needs a human right now.")
        return "\n".join(lines) + "\n"

    lines.append(f"{len(tasks)} manual step(s) + {len(blocked)} blocked site(s). "
                 "Everything else ran (or will run) autonomously.")
    if tasks:
        lines += ["", "## Manual steps"]
        for bid, c in tasks:
            b = brokers_mod.get(bid) or {}
            opt = b.get("optout") or {}
            lines.append(f"### {b.get('name', bid)}")
            lines.append(f"- Why: {c.get('human_task_reason', 'manual step required')}")
            where = opt.get("url") or opt.get("email") or "(see broker record)"
            lines.append(f"- Where: {where}")
            for q in (opt.get("quirks") or [])[:2]:
                lines.append(f"- Note: {q}")
            lines.append("- Withhold: SSN and full ID numbers - always.")
            lines.append(f"- When done, tell the agent so it records the outcome for `{bid}`.")
    if blocked:
        lines += ["", "## Blocked sites (open in YOUR browser - it gets through where bots don't)"]
        for bid, c in blocked:
            b = brokers_mod.get(bid) or {}
            url = ((b.get("search") or {}).get("url")) or "(see broker record)"
            lines.append(f"- **{b.get('name', bid)}** - open {url}, search the subject, and report "
                         "the verdict (or a screenshot) back to the agent.")
    return "\n".join(lines) + "\n"


def sheets_rows(subject_id: str) -> list[list[str]]:
    """Header + one row per case for the optional Google Sheets tracker.

    The agent appends these via the `google-workspace` skill, e.g.:
      google_api.py sheets append <SHEET_ID> "Sheet1!A:F" --values <json-rows>
    """
    rows = [["broker_id", "state", "found", "tier", "removed_at", "next_recheck"]]
    for bid, c in sorted(ledger_mod.load(subject_id).items()):
        rows.append([
            bid,
            c.get("state", ""),
            str(c.get("found", "")),
            (c.get("automation") or {}).get("tier_used", ""),
            c.get("removal_confirmed_at") or "",
            c.get("next_recheck_at") or "",
        ])
    return rows
