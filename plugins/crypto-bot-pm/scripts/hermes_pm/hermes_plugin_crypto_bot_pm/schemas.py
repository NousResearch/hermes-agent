from __future__ import annotations

from typing import Any


def _schema(name: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "output_format": {
                    "type": "string",
                    "enum": ["json", "text"],
                    "description": (
                        "Output format requested from the underlying PM script. "
                        "The tool result itself is always a JSON string."
                    ),
                }
            },
            "additionalProperties": False,
        },
    }


CRYPTO_BOT_PM_STATUS_SCHEMA = _schema(
    "crypto_bot_pm_status",
    (
        "Read-only crypto_bot PM status with live Gitea read context. "
        "Does not mutate Gitea, start runners, run workflows, deploy, touch "
        "runtime surfaces, access secrets, or perform financial actions."
    ),
)

CRYPTO_BOT_PM_ISSUE_LIFECYCLE_SCHEMA = _schema(
    "crypto_bot_pm_issue_lifecycle",
    (
        "Read-only lifecycle attestation for crypto_bot Issue #1. It checks "
        "existing PM issue state through GET-only Gitea reads and does not "
        "create issues, labels, comments, projects, workflows, runners, "
        "runtime actions, secrets, or financial actions."
    ),
)

CRYPTO_BOT_PM_KANBAN_PACKET_SCHEMA = _schema(
    "crypto_bot_pm_kanban_packet",
    (
        "Generate a proposal-only crypto_bot Kanban packet. Does not create "
        "issues, edit projects, mutate Gitea, start runners, run workflows, "
        "deploy, touch runtime surfaces, access secrets, or perform financial "
        "actions."
    ),
)

CRYPTO_BOT_PM_BACKLOG_EXPANSION_SCHEMA = _schema(
    "crypto_bot_pm_backlog_expansion",
    (
        "Generate a proposal-only crypto_bot PM backlog expansion from "
        "existing Issue #1. It proposes 3 to 5 candidates and does not create "
        "issues, edit projects, mutate Gitea, start runners, run workflows, "
        "deploy, touch runtime surfaces, access secrets, invoke branch "
        "writers, invoke issue executors, or perform financial actions."
    ),
)

CRYPTO_BOT_PM_BACKLOG_SELECTION_SCHEMA = _schema(
    "crypto_bot_pm_backlog_selection",
    (
        "Generate a read-only crypto_bot PM backlog selection packet from the "
        "proposal-only backlog candidates. It selects no candidates by "
        "default, recommends one safest candidate for future review, and does "
        "not create issues, mutate Gitea, start runners, run workflows, "
        "deploy, touch runtime surfaces, access secrets, invoke branch "
        "writers, invoke issue executors, or perform financial actions."
    ),
)

CRYPTO_BOT_PM_CANDIDATE_APPROVAL_SCOPE_SCHEMA = _schema(
    "crypto_bot_pm_candidate_approval_scope",
    (
        "Generate a read-only crypto_bot PM approval scope for selected "
        "candidate pm8-002. It prepares exact future create_issue constraints "
        "for Operator review and does not create issues, mutate Gitea, start "
        "runners, run workflows, deploy, touch runtime surfaces, access "
        "secrets, invoke branch writers, invoke issue executors, or perform "
        "financial actions."
    ),
)

CRYPTO_BOT_PM_FORGE_PLAN_SCHEMA = _schema(
    "crypto_bot_pm_forge_plan",
    (
        "Generate a dry-run forge-write plan for review only. It renders "
        "future operation intent without executing Gitea writes, branch "
        "writers, workflow actions, runtime actions, or financial actions."
    ),
)

CRYPTO_BOT_PM_FORGE_APPROVAL_PACKET_SCHEMA = _schema(
    "crypto_bot_pm_forge_approval_packet",
    (
        "Generate a review-only forge approval packet from the dry-run plan. "
        "It is not approval and cannot execute a write."
    ),
)

CRYPTO_BOT_PM_CAPABILITY_MAP_SCHEMA = _schema(
    "crypto_bot_pm_capability_map",
    (
        "Build a read-only Gitea forge capability map for crypto_bot. Does "
        "not call Gitea write APIs, create issues, start runners, run "
        "workflows, deploy, touch runtime surfaces, access secrets, or perform "
        "financial actions."
    ),
)

CRYPTO_BOT_PM_DEVELOPMENT_WORKSTREAM_SCHEMA = _schema(
    "crypto_bot_pm_development_workstream",
    (
        "Generate a read-only crypto_bot completion workstream packet. It "
        "selects the first safe development slice for Operator review and "
        "does not create issues, mutate Gitea, start runners, run workflows, "
        "deploy, touch runtime surfaces, access secrets, invoke branch "
        "writers, invoke issue executors, or perform financial actions."
    ),
)

CRYPTO_BOT_PM_DEVELOPMENT_SLICE_SCHEMA = _schema(
    "crypto_bot_pm_development_slice",
    (
        "Generate a non-mutating implementation-ready development slice "
        "packet for the recommended crypto_bot completion candidate. It "
        "requires future Operator approval before any branch write and does "
        "not call Gitea write APIs or write files."
    ),
)


TOOL_SCHEMAS = {
    "crypto_bot_pm_issue_lifecycle": CRYPTO_BOT_PM_ISSUE_LIFECYCLE_SCHEMA,
    "crypto_bot_pm_status": CRYPTO_BOT_PM_STATUS_SCHEMA,
    "crypto_bot_pm_kanban_packet": CRYPTO_BOT_PM_KANBAN_PACKET_SCHEMA,
    "crypto_bot_pm_backlog_expansion": CRYPTO_BOT_PM_BACKLOG_EXPANSION_SCHEMA,
    "crypto_bot_pm_backlog_selection": CRYPTO_BOT_PM_BACKLOG_SELECTION_SCHEMA,
    "crypto_bot_pm_candidate_approval_scope": (
        CRYPTO_BOT_PM_CANDIDATE_APPROVAL_SCOPE_SCHEMA
    ),
    "crypto_bot_pm_forge_plan": CRYPTO_BOT_PM_FORGE_PLAN_SCHEMA,
    "crypto_bot_pm_forge_approval_packet": CRYPTO_BOT_PM_FORGE_APPROVAL_PACKET_SCHEMA,
    "crypto_bot_pm_capability_map": CRYPTO_BOT_PM_CAPABILITY_MAP_SCHEMA,
    "crypto_bot_pm_development_workstream": (
        CRYPTO_BOT_PM_DEVELOPMENT_WORKSTREAM_SCHEMA
    ),
    "crypto_bot_pm_development_slice": CRYPTO_BOT_PM_DEVELOPMENT_SLICE_SCHEMA,
}
