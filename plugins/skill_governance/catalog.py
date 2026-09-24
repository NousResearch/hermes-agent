"""Approved skill-to-plugin candidate catalog.

This data is intentionally static and side-effect free: it records Vladimir's
approved roadmap so the first governance plugin can produce deterministic plans
before any credentialed Bitrix, Telegram, Google, Ozon, or GitHub plugin exists.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "bitrix_ops",
        "title": "Bitrix Ops Plugin",
        "priority": "very_high",
        "wave": 1,
        "areas": ["operations", "saturn-management", "external-api"],
        "sources": ["bitrix24-agent", "bitrix-tasks-fastlane", "OpenClaw Bitrix rules"],
        "tools": [
            "bitrix_get_task",
            "bitrix_create_task",
            "bitrix_add_comment",
            "bitrix_send_message",
            "bitrix_check_delivery",
            "bitrix_readback_dialog",
            "bitrix_resolve_user_safe",
        ],
        "business_value": "Controlled Bitrix operations with delivery proof instead of ad-hoc shell/API calls.",
        "guardrails": [
            "No live Bitrix send without an explicit live-go policy gate.",
            "Separate API sent, message_id, agent readback, and human visual ack.",
            "Never print raw user IDs, webhook URLs, tokens, or production DSNs.",
        ],
        "requires_live_go": True,
    },
    {
        "id": "telegram_thread_router",
        "title": "Telegram Thread Router Plugin",
        "priority": "very_high",
        "wave": 1,
        "areas": ["operations", "telegram", "routing"],
        "sources": ["telegram-business-hermes", "chip-bot-coop", "OpenClaw handoff rules"],
        "tools": [
            "telegram_get_thread_context",
            "telegram_classify_message",
            "telegram_send_handoff",
            "telegram_ack_message",
            "telegram_find_active_topic",
            "telegram_check_unanswered_mentions",
        ],
        "business_value": "Reduces Telegram task-routing chaos across Hermes, RaskovaloBot, and Saturn topics.",
        "guardrails": [
            "Respect current chat/thread/source-owner lock.",
            "Treat generic resume/recovery text from other threads as historical only.",
            "Use visible acknowledgements without leaking tool traces or private context.",
        ],
        "requires_live_go": True,
    },
    {
        "id": "management_digest",
        "title": "Management Digest Plugin",
        "priority": "very_high",
        "wave": 1,
        "areas": ["operations", "management", "reporting"],
        "sources": ["morning-command-center", "workshop-digest", "workshop-guardian-angel"],
        "tools": [
            "digest_collect_sources",
            "digest_rank_risks",
            "digest_build_manager_card",
            "digest_get_blockers",
            "digest_prepare_daily_report",
        ],
        "business_value": "Turns raw agent/CRM/chat activity into director-ready decisions, blockers, and actions.",
        "guardrails": [
            "Prefer actionable summaries over raw cron/watchdog noise.",
            "Keep manager cards plain: action, recommendation, blocker, next step.",
            "Do not include raw private messages, IDs, secrets, or tool traces.",
        ],
        "requires_live_go": False,
    },
    {
        "id": "watchdog_guardian",
        "title": "Watchdog / Guardian Plugin",
        "priority": "high",
        "wave": 1,
        "areas": ["operations", "monitoring", "reliability"],
        "sources": ["watchdog", "web-monitor", "cron-governor", "project-loop"],
        "tools": [
            "guardian_check_services",
            "guardian_check_cron",
            "guardian_check_gateway",
            "guardian_detect_silence",
            "guardian_summarize_incident",
            "guardian_create_recovery_task",
        ],
        "business_value": "Provides operational health control that distinguishes real incidents from noise.",
        "guardrails": [
            "Summarize only actionable failures by default.",
            "Escalate silence and missed checkpoints with proof, not speculation.",
            "Avoid live restarts/deploys unless the active policy permits them.",
        ],
        "requires_live_go": False,
    },
    {
        "id": "google_workspace_ops",
        "title": "Google Workspace Ops Plugin",
        "priority": "high",
        "wave": 2,
        "areas": ["operations", "documents", "external-api"],
        "sources": ["google-workspace", "google-api-operator", "google-workspace-ops"],
        "tools": [
            "gdocs_read_doc",
            "gdocs_extract_tasks",
            "gsheets_update_range",
            "gdrive_find_file",
            "gdrive_permission_check",
            "gdocs_write_report",
        ],
        "business_value": "Makes Google Docs and Sheets a structured input/output surface for Hermes tasks.",
        "guardrails": [
            "Read/classify task packets before acting.",
            "Preserve private links and OAuth material; report only sanitized proof.",
            "Separate document drafting from permission-changing operations.",
        ],
        "requires_live_go": True,
    },
    {
        "id": "procurement_tender_pipeline",
        "title": "Procurement / Tender Pipeline Plugin",
        "priority": "high",
        "wave": 2,
        "areas": ["saturn-business", "procurement", "documents"],
        "sources": [
            "zakupochny-kontur",
            "tz-parser-for-procurement",
            "ocr-procurement-pipeline",
            "commercial-proposal-builder",
        ],
        "tools": [
            "procurement_parse_tz",
            "procurement_extract_items",
            "procurement_normalize_catalog",
            "procurement_build_kp",
            "procurement_check_missing_docs",
            "procurement_estimate_margin",
        ],
        "business_value": "Accelerates Saturn tender intake, normalization, and commercial proposal preparation.",
        "guardrails": [
            "Keep OCR/extraction evidence attached to the local artifact, not raw chat.",
            "Mark assumptions and missing documents explicitly.",
            "Do not submit or send commercial offers without a separate business gate.",
        ],
        "requires_live_go": False,
    },
    {
        "id": "document_factory",
        "title": "Document Factory Plugin",
        "priority": "high",
        "wave": 2,
        "areas": ["saturn-business", "documents", "office"],
        "sources": ["xlsx-quality-gate", "officecli-safe-operator", "ocr-and-documents", "powerpoint"],
        "tools": [
            "docx_fill_template",
            "xlsx_validate_formulas",
            "xlsx_recalculate",
            "pdf_extract_text",
            "pdf_fix_text",
            "office_export_pdf",
            "document_quality_gate",
        ],
        "business_value": "Turns office documents into a verified production pipeline for KP, reports, and tables.",
        "guardrails": [
            "Always validate generated files open and formulas recalculate.",
            "Send document files as attachments, not images.",
            "Keep signatures/stamps under separate asset-governance approval.",
        ],
        "requires_live_go": False,
    },
    {
        "id": "ozon_marketplace_import",
        "title": "Ozon / Marketplace Import Plugin",
        "priority": "medium_high",
        "wave": 2,
        "areas": ["saturn-business", "marketplace", "external-api"],
        "sources": ["marketplace-product-import", "Ozon Seller API guardrails"],
        "tools": [
            "ozon_validate_products",
            "ozon_prepare_import",
            "ozon_check_attributes",
            "ozon_dry_run_import",
            "ozon_submit_import_guarded",
            "ozon_import_status",
        ],
        "business_value": "Makes product import controlled: validate locally, dry-run, then gated live submit.",
        "guardrails": [
            "Keep Ozon credentials only in the active profile environment.",
            "Live product import requires explicit GO and import-understanding flag.",
            "Default to validation and draft artifacts, not live API writes.",
        ],
        "requires_live_go": True,
    },
    {
        "id": "broker_perplexity_relay",
        "title": "Broker / Perplexity Relay Plugin",
        "priority": "high",
        "wave": 3,
        "areas": ["agent-infra", "broker", "orchestration"],
        "sources": ["perplex", "deep", "OpenClaw 5КАПМОСТИК broker handoff rules"],
        "tools": [
            "broker_send_report",
            "broker_request_next_tz",
            "broker_readback_marker",
            "broker_parse_verdict",
            "broker_save_artifact",
            "broker_check_manual_control",
        ],
        "business_value": "Hardens external broker handoff, marker readback, verdict parsing, and artifact proof.",
        "guardrails": [
            "Use canonical broker thread and marker-scoped readback.",
            "Stop on HOLD or MANUAL_CONTROL unless the active policy explicitly permits continuation.",
            "Never print cookies, browser session values, or raw private broker artifacts.",
        ],
        "requires_live_go": True,
    },
    {
        "id": "github_pr_governance",
        "title": "GitHub / PR Governance Plugin",
        "priority": "high",
        "wave": 3,
        "areas": ["agent-infra", "engineering", "github"],
        "sources": ["github-pr-workflow", "github", "requesting-code-review", "task-proof-loop"],
        "tools": [
            "repo_status_guard",
            "github_open_pr",
            "github_check_ci",
            "github_exact_head_merge",
            "github_postmerge_verify",
            "github_detect_duplicate_pr",
            "github_blocker_report",
        ],
        "business_value": "Turns PR work into a safe pipeline instead of scattered git and gh shell commands.",
        "guardrails": [
            "No direct push to main; no force-push; no amend after first push.",
            "Use exact-head merge and post-merge verification when merge is authorized.",
            "Check dirty worktree and duplicate/stale PR state before editing.",
        ],
        "requires_live_go": True,
    },
    {
        "id": "employee_access_adoption",
        "title": "Employee Access / Adoption Plugin",
        "priority": "medium_high",
        "wave": 3,
        "areas": ["operations", "employee-access", "analytics"],
        "sources": ["Employee Access contour", "employee-access-adoption-vs-product-spec"],
        "tools": [
            "employee_access_usage_summary",
            "employee_access_classify_activity",
            "employee_access_find_inactive",
            "employee_access_privacy_safe_report",
            "employee_access_route_binding_check",
        ],
        "business_value": "Shows adoption depth and routing health without exposing employee-private data.",
        "guardrails": [
            "Report activity by type and depth, not raw messages or identifiers.",
            "Keep adoption analytics separate from product roadmap completion claims.",
            "Route access provisioning through authorization and sandbox boundaries.",
        ],
        "requires_live_go": False,
    },
    {
        "id": "skill_governance",
        "title": "Skill Governance / Skill-to-Plugin Advisor Plugin",
        "priority": "medium",
        "wave": 3,
        "areas": ["agent-infra", "skills", "governance"],
        "sources": ["hermes-skill-library-operations", "hermes-agent-skill-authoring", "curator"],
        "tools": [
            "skills_installed_audit",
            "skills_find_plugin_candidates",
            "skills_detect_duplicates",
            "skills_usage_stats",
            "skills_to_plugin_plan",
        ],
        "business_value": "Keeps Hermes skills from becoming a fragile pile of procedures by promoting mature capabilities into plugins.",
        "guardrails": [
            "Prefer read-only audit before installing, editing, or deleting skills.",
            "Preserve provenance and avoid copying secrets from skills into plugin code.",
            "Keep plugin plans incremental with tests and safe defaults.",
        ],
        "requires_live_go": False,
    },
]


def all_candidates() -> list[dict[str, Any]]:
    """Return a deep copy so handlers cannot mutate the catalog."""

    return deepcopy(CANDIDATES)


def find_candidate(candidate_id: str) -> dict[str, Any] | None:
    """Return a deep-copied candidate by id."""

    normalized = (candidate_id or "").strip().lower().replace("-", "_")
    for candidate in CANDIDATES:
        if candidate["id"] == normalized:
            return deepcopy(candidate)
    return None
