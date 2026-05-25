"""Canonical phase and issue plan for Trend Discovery Center."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseSpec:
    phase_id: str
    name: str
    duration_days: int
    objective: str


@dataclass(frozen=True)
class IssueSpec:
    issue_id: str
    phase_id: str
    title: str
    owner_role: str


PHASES: tuple[PhaseSpec, ...] = (
    PhaseSpec("P0", "Approval And Baseline Audit", 1, "Confirm runtime targets and baseline health."),
    PhaseSpec("P1", "Anti-Forget System", 7, "Persist the plan, reminders, watchdogs, logs, and health gates."),
    PhaseSpec("P2", "Reliable Multi-Source Scanner", 7, "Scan through independent source adapters with fallback controls."),
    PhaseSpec("P3", "Business Intelligence And Knowledge Capture", 7, "Turn raw discoveries into scored, deduped project knowledge."),
    PhaseSpec("P4", "Review, Hardening, And Wow Layer", 7, "Audit reliability, delivery, recovery, and final compliance evidence."),
)


ISSUES: tuple[IssueSpec, ...] = (
    IssueSpec("TD-0001", "P0", "Select deployment target: local-first localhost with optional VPS health URL.", "System Architect"),
    IssueSpec("TD-0002", "P0", "Select notification target: local receipt log with optional webhook/gateway handoff.", "Notification Engineer"),
    IssueSpec("TD-0003", "P0", "Verify Hermes runtime prerequisites for local execution.", "QA Engineer"),
    IssueSpec("TD-0004", "P0", "Verify dashboard/localhost target can be health checked when configured.", "QA Engineer"),
    IssueSpec("TD-0005", "P0", "Select storage: profile-scoped SQLite under HERMES_HOME.", "Data Engineer"),
    IssueSpec("TD-0006", "P0", "Seed first trend scopes and default source definitions.", "Research Strategist"),
    IssueSpec("TD-0007", "P0", "Define numeric compliance output contract.", "QA Engineer"),
    IssueSpec("TD-0008", "P0", "Record baseline risks and default choices.", "SRE"),
    IssueSpec("TD-1001", "P1", "Create persistent phase registry with issue percentages.", "Data Engineer"),
    IssueSpec("TD-1002", "P1", "Create run log schema with evidence fields.", "Data Engineer"),
    IssueSpec("TD-1003", "P1", "Create phase reminder schedule.", "SRE"),
    IssueSpec("TD-1004", "P1", "Create watchdog rules for overdue and repeated failures.", "SRE"),
    IssueSpec("TD-1005", "P1", "Create notification dispatcher with primary and fallback behavior.", "Notification Engineer"),
    IssueSpec("TD-1006", "P1", "Create delivery receipt log.", "Notification Engineer"),
    IssueSpec("TD-1007", "P1", "Create status command for phase progress.", "Product Owner"),
    IssueSpec("TD-1008", "P1", "Create local health check command.", "QA Engineer"),
    IssueSpec("TD-1009", "P1", "Test scheduler, watchdog, reminder, and delivery failure paths.", "QA Engineer"),
    IssueSpec("TD-1010", "P1", "Verify local runtime before phase delivery.", "QA Engineer"),
    IssueSpec("TD-2001", "P2", "Define source adapter interface.", "Crawler Engineer"),
    IssueSpec("TD-2002", "P2", "Implement RSS/news adapter.", "Crawler Engineer"),
    IssueSpec("TD-2003", "P2", "Implement search provider adapter hooks.", "Crawler Engineer"),
    IssueSpec("TD-2004", "P2", "Implement direct webpage fetch adapter.", "Crawler Engineer"),
    IssueSpec("TD-2005", "P2", "Implement Open Crawl as optional adapter only.", "Crawler Engineer"),
    IssueSpec("TD-2006", "P2", "Implement n8n as optional import/webhook adapter only.", "Crawler Engineer"),
    IssueSpec("TD-2007", "P2", "Enforce timeout per source.", "SRE"),
    IssueSpec("TD-2008", "P2", "Add retry with backoff metadata.", "SRE"),
    IssueSpec("TD-2009", "P2", "Add circuit breaker for repeated source failures.", "SRE"),
    IssueSpec("TD-2010", "P2", "Add fallback rule so one failed source cannot fail the pipeline.", "SRE"),
    IssueSpec("TD-2011", "P2", "Track source health score.", "Data Engineer"),
    IssueSpec("TD-2012", "P2", "Test simulated source failure.", "QA Engineer"),
    IssueSpec("TD-2013", "P2", "Verify scan succeeds with partial source failure.", "QA Engineer"),
    IssueSpec("TD-3001", "P3", "Extract company or entity names from findings.", "Data Engineer"),
    IssueSpec("TD-3002", "P3", "Dedupe findings by URL, domain, and normalized title.", "Data Engineer"),
    IssueSpec("TD-3003", "P3", "Compute relevance score.", "Research Strategist"),
    IssueSpec("TD-3004", "P3", "Compute novelty score against stored findings.", "Research Strategist"),
    IssueSpec("TD-3005", "P3", "Apply trend tags.", "Knowledge Engineer"),
    IssueSpec("TD-3006", "P3", "Store source provenance for every finding.", "Knowledge Engineer"),
    IssueSpec("TD-3007", "P3", "Write review queue markdown.", "Knowledge Engineer"),
    IssueSpec("TD-3008", "P3", "Write project-specific digest output.", "Knowledge Engineer"),
    IssueSpec("TD-3009", "P3", "Generate daily digest.", "Product Owner"),
    IssueSpec("TD-3010", "P3", "Generate weekly trend report.", "Product Owner"),
    IssueSpec("TD-3011", "P3", "Test dedupe, scoring, and writeback.", "QA Engineer"),
    IssueSpec("TD-3012", "P3", "Verify report generation on local runtime.", "QA Engineer"),
    IssueSpec("TD-4001", "P4", "Generate source reliability report.", "SRE"),
    IssueSpec("TD-4002", "P4", "Generate insight quality review.", "Research Strategist"),
    IssueSpec("TD-4003", "P4", "Generate phase completion audit.", "QA Engineer"),
    IssueSpec("TD-4004", "P4", "Generate notification audit.", "Notification Engineer"),
    IssueSpec("TD-4005", "P4", "Expose dashboard-ready JSON status output.", "System Architect"),
    IssueSpec("TD-4006", "P4", "Generate evidence pack.", "QA Engineer"),
    IssueSpec("TD-4007", "P4", "Generate operator runbook.", "SRE"),
    IssueSpec("TD-4008", "P4", "Test recovery from source/service failure.", "SRE"),
    IssueSpec("TD-4009", "P4", "Run final local smoke test.", "QA Engineer"),
    IssueSpec("TD-4010", "P4", "Generate final numeric compliance report.", "QA Engineer"),
)


DEFAULT_SCOPES: tuple[str, ...] = (
    "agentic workflow",
    "AI infrastructure",
    "robotics automation",
    "climate technology",
    "developer productivity",
)
