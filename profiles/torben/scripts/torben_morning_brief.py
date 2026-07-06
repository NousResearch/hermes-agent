from __future__ import annotations

import json
import os
import hashlib
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from hermes_constants import get_hermes_home
from hermes_cli.signal_coo.action_ledger import ActionLedger
from hermes_cli.signal_coo.email_audit import (
    collect_gmail_inbox_audit,
    load_relationship_context,
    write_json_artifact as write_email_json,
)
from hermes_cli.signal_coo.google_evidence import collect_google_ea_evidence, write_json_artifact as write_google_json
from hermes_cli.signal_coo.morning_brief import build_meeting_signal_packets
from hermes_cli.signal_coo.morning_findings import filter_new_findings, record_llm_signal_candidates
from hermes_cli.signal_coo.relationship_learning import stage_relationship_learning_actions
from torben_attention_contract import (
    build_brief_attention_section,
    load_pattern_proposals,
    load_pending_decisions,
    summarize_open_loops,
)

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


WEB_RESEARCH_UNAVAILABLE_MESSAGE = "web research unavailable"


def build_web_research_status(env: Mapping[str, str] | None = None) -> dict[str, object]:
    source = os.environ if env is None else env
    configured = [name for name in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL") if str(source.get(name) or "").strip()]
    if configured:
        return {
            "status": "configured",
            "configured": True,
            "configured_vars": configured,
            "message": "web research configured",
            "agent_instruction": "Web tools may be used only for bounded research tied to supplied candidates.",
        }
    return {
        "status": "unavailable",
        "configured": False,
        "configured_vars": [],
        "message": WEB_RESEARCH_UNAVAILABLE_MESSAGE,
        "reason": "FIRECRAWL_API_KEY and FIRECRAWL_API_URL are not set for this Torben profile",
        "agent_instruction": (
            "Do not call web_search or web_extract. If an item would require outside web research, "
            "state the exact phrase 'web research unavailable' instead of silently omitting the limitation."
        ),
    }


def load_signal_rubric(path: str | Path) -> dict:
    rubric_path = Path(path)
    if not rubric_path.exists():
        return {
            "path": str(rubric_path),
            "sha256": "",
            "missing": True,
            "content": "",
        }
    content = rubric_path.read_text(encoding="utf-8").strip()
    return {
        "path": str(rubric_path),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "missing": False,
        "content": content,
    }


def build_llm_signal_judge_contract(rubric: dict) -> dict:
    return {
        "purpose": "The LLM chooses what is actually worth surfacing from bounded evidence; scripts do not assign final relevance scores.",
        "rubric_sha256": rubric.get("sha256") or "",
        "input_surfaces": [
            "calendar.meeting_signal_packets",
            "inbox.deduped_previous_day_security_stories",
            "inbox.deduped_previous_day_tools",
            "inbox.critical_emails",
            "inbox.learn_contact_candidates",
            "web_research.status",
        ],
        "hard_rules": [
            "Use the markdown rubric as the forcing function for final signal judgment.",
            "Prefer concrete article_synopsis, source_excerpt, named_tools, key_concepts, attendee/domain evidence, relationship context, and explicit unknowns.",
            "Do not summarize every candidate. Suppress weak candidates even when they matched a deterministic source rule.",
            "Do not invent a meeting agenda, person role, company description, or article claim that is not in evidence or verified by available research.",
            "If web_research.status is unavailable, do not call web_search or web_extract; surface the exact phrase 'web research unavailable' where outside context would have been used.",
            "If a meeting or article needs outside context and tools are available, do bounded research; otherwise state the unknown and ask the decision-forcing question.",
            "Respect hard_suppressed_items and suppressed_duplicate_findings unless there is a materially new angle.",
        ],
        "selection_guidance": {
            "default_shape": "Aim for a concise one-screen brief, usually 3-7 total surfaced items.",
            "not_a_hard_cap": "Surface more only when each item has a distinct reason Eric can act on today.",
            "required_fields_for_surfaced_items": [
                "what",
                "why_eric_cares",
                "evidence_used",
                "next_question_or_action",
                "best_link_when_available",
            ],
        },
        "output_schema": {
            "surfaced_items": [
                {
                    "candidate_fingerprint": "optional fingerprint from input",
                    "surface": True,
                    "reason": "rubric-grounded reason",
                    "brief_sentence": "line suitable for the morning brief",
                    "why_eric_cares": "security, AI, GTM, ops, finance, or relationship context",
                    "evidence_used": ["field names or evidence ids"],
                }
            ],
            "suppressed_items": [
                {
                    "candidate_fingerprint": "optional fingerprint from input",
                    "surface": False,
                    "suppression_reason": "duplicate, weak evidence, not timely, not actionable, or unsafe",
                }
            ],
        },
    }


def main() -> int:
    home = get_hermes_home()
    config_path = home / "config" / "google_accounts.yaml"
    relationship_context_path = home / "config" / "relationship_context.yaml"
    signal_rubric_path = home / "config" / "morning_brief_signal_rubric.md"
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    preview = _truthy(os.getenv("TORBEN_MORNING_BRIEF_PREVIEW"))
    web_research = build_web_research_status()
    signal_rubric = load_signal_rubric(signal_rubric_path)
    full_relationship_context = load_relationship_context(relationship_context_path)

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
    meeting_signal_packets = build_meeting_signal_packets(
        (ea.get("ea") or {}).get("calendar_events") or [],
        relationship_context=full_relationship_context,
    )
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
            ledger=ActionLedger(state_dir / "torben-action-ledger.jsonl"),
            candidates=list(morning_candidates.get("learn_contact_candidates") or [])[:5],
        )
    signal_candidate_ledger = record_llm_signal_candidates(
        ledger_path=state_dir / "torben-morning-brief-llm-signal-ledger.json",
        rubric_hash=signal_rubric.get("sha256") or "",
        stories=finding_dedupe["new_stories"],
        tools=finding_dedupe["new_tools"],
        meeting_packets=meeting_signal_packets,
        ttl_days=int(os.getenv("TORBEN_MORNING_BRIEF_SIGNAL_LEDGER_TTL_DAYS", "30")),
        dry_run=preview,
    )
    pending_decisions = load_pending_decisions(state_dir / "torben-pending-decisions.json")
    pattern_proposals = load_pattern_proposals(state_dir / "torben-pattern-proposals.json")
    open_loops = summarize_open_loops(state_dir / "torben-open-loops.csv")
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
                "Use llm_signal_rubric and llm_signal_judge_contract as the forcing function for choosing what to surface.",
            ],
            "meeting_signal": [
                "Use calendar.meeting_signal_packets for weak-context meetings.",
                "Do not invent agenda or relationship context; use known_context, company_context_hint, unknowns, and suggested_question.",
                "For intro calls, the useful outcome is the next decision, blocker, or reason to continue.",
            ],
            "learn_contact": "If a learn-contact candidate matters, ask exactly one short question with its handle.",
            "web_research": [
                "Read web_research.status before using web tools.",
                "If status is unavailable, do not call web_search or web_extract.",
                "If outside web research would otherwise be cited, include the exact phrase 'web research unavailable'.",
            ],
            "email_draft_guardrails": EMAIL_DRAFT_GUARDRAILS,
        },
        "web_research": web_research,
        "brief_attention": build_brief_attention_section(
            pending_decisions=pending_decisions,
            open_loops=open_loops,
            pattern_proposals=pattern_proposals,
        ),
        "llm_signal_rubric": signal_rubric,
        "llm_signal_judge_contract": build_llm_signal_judge_contract(signal_rubric),
        "relationship_context": email_audit.get("relationship_context") or {},
        "llm_decision_contract": morning_candidates.get("llm_decision_contract") or {},
        "calendar": {
            "events": (ea.get("ea") or {}).get("calendar_events") or [],
            "block_candidates": (ea.get("ea") or {}).get("calendar_block_candidates") or [],
            "morning_brief_scope": (ea.get("ea") or {}).get("morning_brief") or {},
            "meeting_signal_packets": meeting_signal_packets,
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
        "pending_decisions": pending_decisions + pattern_proposals,
        "open_loops": open_loops,
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
            "web_research": web_research,
            "finding_dedupe": {
                "ledger_path": finding_dedupe["ledger_path"],
                "ttl_days": finding_dedupe["ttl_days"],
                "dry_run": finding_dedupe["dry_run"],
                "new_story_count": len(finding_dedupe["new_stories"]),
                "new_tool_count": len(finding_dedupe["new_tools"]),
                "duplicate_count": len(finding_dedupe["duplicates"]),
            },
            "llm_signal_candidate_ledger": {
                "ledger_path": signal_candidate_ledger["ledger_path"],
                "ttl_days": signal_candidate_ledger["ttl_days"],
                "rubric_hash": signal_candidate_ledger["rubric_hash"],
                "dry_run": signal_candidate_ledger["dry_run"],
                "candidate_count": signal_candidate_ledger["candidate_count"],
            },
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    from torben_job_contract import run_job

    raise SystemExit(run_job("torben-morning-brief", main))
