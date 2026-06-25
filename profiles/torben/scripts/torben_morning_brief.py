from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from hermes_constants import get_hermes_home
from hermes_cli.signal_coo.action_ledger import ActionLedger
from hermes_cli.signal_coo.email_audit import collect_gmail_inbox_audit, write_json_artifact as write_email_json
from hermes_cli.signal_coo.google_evidence import collect_google_ea_evidence, write_json_artifact as write_google_json
from hermes_cli.signal_coo.morning_findings import filter_new_findings
from hermes_cli.signal_coo.relationship_learning import stage_relationship_learning_actions

LOCAL_TZ = ZoneInfo("America/New_York")


EMAIL_DRAFT_GUARDRAILS = [
    "Email drafts must use the same anti-slop bar as Eric's X/LinkedIn article drafts.",
    "Write tight and direct: short sentences, concrete facts, no padded empathy, no generic polish.",
    "No AI-sounding slop: no em dashes, no predictable 'not X but Y' contrast templates, no beige corporate phrasing, no filler openers.",
    "Sound like Eric: candid, human, specific, and practical; do not over-explain or apologize unless the situation requires it.",
    "If drafting a reply, produce only usable copy plus any one-line rationale; never wrap it in a generic assistant preface.",
]


def _candidate_local_date(item: dict) -> object | None:
    raw = str(item.get("internal_date_ms") or "")
    if raw.isdigit():
        return datetime.fromtimestamp(int(raw) / 1000, tz=LOCAL_TZ).date()
    return None


def _previous_day_items(items: list[dict], *, limit: int) -> list[dict]:
    target = (datetime.now(LOCAL_TZ).date() - timedelta(days=1))
    previous = [item for item in items if _candidate_local_date(item) == target]
    return (previous or items)[:limit]


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    home = get_hermes_home()
    config_path = home / "config" / "google_accounts.yaml"
    relationship_context_path = home / "config" / "relationship_context.yaml"
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    preview = _truthy(os.getenv("TORBEN_MORNING_BRIEF_PREVIEW"))

    ea = collect_google_ea_evidence(
        config_path=config_path,
        days=int(os.getenv("TORBEN_MORNING_BRIEF_CALENDAR_DAYS", "2")),
        max_calendar_events=int(os.getenv("TORBEN_MORNING_BRIEF_MAX_CALENDAR_EVENTS", "20")),
        max_email_messages=int(os.getenv("TORBEN_MORNING_BRIEF_MAX_GOOGLE_EMAILS", "12")),
        max_calendar_block_candidates=int(os.getenv("TORBEN_MORNING_BRIEF_MAX_BLOCK_CANDIDATES", "8")),
        include_secondary_calendars=False,
    )
    inbox = collect_gmail_inbox_audit(
        config_path=config_path,
        relationship_context_path=relationship_context_path,
        days=int(os.getenv("TORBEN_MORNING_BRIEF_EMAIL_DAYS", "2")),
        max_messages_per_account=int(os.getenv("TORBEN_MORNING_BRIEF_MAX_MESSAGES", "800")),
        max_body_fetches_per_account=int(os.getenv("TORBEN_MORNING_BRIEF_MAX_BODY_FETCHES", "400")),
        fetch_workers=int(os.getenv("TORBEN_MORNING_BRIEF_WORKERS", "8")),
    )
    write_google_json(ea, state_dir / "torben-google-ea-evidence-latest.json")
    write_email_json(inbox, state_dir / "torben-morning-brief-inbox-context-latest.json")

    email_audit = inbox.get("email_audit") or {}
    morning_candidates = email_audit.get("morning_briefing_candidates") or {}
    security_stories = list(morning_candidates.get("security_stories") or [])
    tools = list(morning_candidates.get("tools") or [])
    previous_day_stories = _previous_day_items(security_stories, limit=8)
    previous_day_tools = _previous_day_items(tools, limit=12)
    finding_dedupe = filter_new_findings(
        ledger_path=state_dir / "torben-morning-brief-findings-ledger.json",
        stories=previous_day_stories,
        tools=previous_day_tools,
        ttl_days=int(os.getenv("TORBEN_MORNING_BRIEF_FINDING_TTL_DAYS", "14")),
        dry_run=preview,
    )
    if preview:
        learn_contact_candidates = [
            {**candidate, "handle": f"PREVIEW-LEARN-{index:03d}"}
            for index, candidate in enumerate(list(morning_candidates.get("learn_contact_candidates") or [])[:5], start=1)
        ]
    else:
        learn_contact_candidates = stage_relationship_learning_actions(
            ledger=ActionLedger(state_dir / "torben-action-ledger.json"),
            candidates=list(morning_candidates.get("learn_contact_candidates") or [])[:5],
        )
    google_diag = ((ea.get("source_diagnostics") or {}).get("google") or {})
    gmail_diag = ((inbox.get("source_diagnostics") or {}).get("gmail") or {})
    payload = {
        "task": "torben_morning_brief_llm_context",
        "contracts": {
            "mutation_boundary": "read/summarize/stage only; do not send email, mutate calendars, archive, label, trade, or post",
            "brief_style": "conversational COO, specific, one screen, link-heavy where useful, no generic advice",
            "newsletter_signal": [
                "Prefer previous-day security, AI, and tool signal.",
                "Each story/tool item should be one sentence plus the best link.",
                "Use deduped_previous_day_* as the canonical story/tool input.",
                "Do not reintroduce suppressed_duplicate_findings unless there is a materially new angle.",
                "Do not turn newsletters into a report; surface only items Eric can use for thought leadership, security awareness, or tool exploration.",
                "Realtime email triage is separate. Do not repeat non-urgent realtime scan details in the morning brief.",
            ],
            "learn_contact": "If a learn-contact candidate matters, ask exactly one short question with its handle.",
            "email_draft_guardrails": EMAIL_DRAFT_GUARDRAILS,
        },
        "relationship_context": email_audit.get("relationship_context") or {},
        "llm_decision_contract": morning_candidates.get("llm_decision_contract") or {},
        "calendar": {
            "events": (ea.get("ea") or {}).get("calendar_events") or [],
            "block_candidates": (ea.get("ea") or {}).get("calendar_block_candidates") or [],
            "morning_brief_scope": (ea.get("ea") or {}).get("morning_brief") or {},
        },
        "inbox": {
            "category_counts": email_audit.get("category_counts") or {},
            "deduped_previous_day_security_stories": finding_dedupe["new_stories"],
            "deduped_previous_day_tools": finding_dedupe["new_tools"],
            "suppressed_duplicate_findings": finding_dedupe["duplicates"][:20],
            "ai_newsletter_sources": morning_candidates.get("ai_newsletter_sources") or [],
            "boardy_digest": morning_candidates.get("boardy_digest") or [],
            "critical_emails": morning_candidates.get("critical_emails") or [],
            "learn_contact_candidates": learn_contact_candidates,
        },
        "diagnostics": {
            "google_reads": (google_diag.get("audit") or {}).get("google_read_api_calls", 0),
            "google_writes": (google_diag.get("audit") or {}).get("google_write_api_calls", 0),
            "gmail_reads": (gmail_diag.get("audit") or {}).get("gmail_read_api_calls", 0),
            "gmail_writes": (gmail_diag.get("audit") or {}).get("gmail_write_api_calls", 0),
            "external_mutations": max(
                int((google_diag.get("audit") or {}).get("external_mutations", 0) or 0),
                int((gmail_diag.get("audit") or {}).get("external_mutations", 0) or 0),
            ),
            "warnings": list((google_diag.get("audit") or {}).get("warnings") or [])
            + list((gmail_diag.get("audit") or {}).get("warnings") or [])[:8],
            "finding_dedupe": {
                "ledger_path": finding_dedupe["ledger_path"],
                "ttl_days": finding_dedupe["ttl_days"],
                "dry_run": finding_dedupe["dry_run"],
                "new_story_count": len(finding_dedupe["new_stories"]),
                "new_tool_count": len(finding_dedupe["new_tools"]),
                "duplicate_count": len(finding_dedupe["duplicates"]),
            },
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
