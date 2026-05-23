# Failure Pattern Library

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18
**Priority:** 2 of 7

## Purpose

Provide a queryable, categorized catalog of past failure modes so Hermes agents can recognize recurring problems, apply known fixes, and avoid re-investigating solved issues.

## Preconditions

- This is a reference catalog, not an active monitoring system.
- Entries are added manually after post-mortem or audit, not automatically.
- Each entry must have a confirmed root cause before being added (no speculative entries).

---

## Failure Category Taxonomy

| Category | Code | Description |
| --- | --- | --- |
| Data staleness | DS | Input data exceeded freshness threshold |
| Cache miss / corruption | CM | Cache file missing, truncated, or stale |
| Pipeline timeout | PT | Job exceeded time limit |
| Logic error | LE | Code produced wrong output from correct input |
| Doc-sync gap | DG | Documents contradicted each other or production state |
| Infrastructure failure | IF | Service unreachable, API down, credential expired |
| Governance lapse | GL | Process skipped, gate bypassed, approval missing |
| Naming collision | NC | Same concept referenced by different names across docs |
| Misattribution | MA | Metric or claim attributed to wrong source or scope |

---

## Failure Entry Schema

```
failure_id: F-YYYY-NNN
category: [code from taxonomy]
first_seen: YYYY-MM-DD
recurrence_count: N
severity: LOW | MEDIUM | HIGH | CRITICAL
summary: One-line description
root_cause: What actually went wrong
affected_systems: [list of skills/agents/pipelines]
resolution: What fixed it (or "UNRESOLVED")
prevention_rule: What should change to prevent recurrence
related_specs: [spec numbers if applicable]
related_findings: [C1, W3, G5 from doc review if applicable]
```

---

## Catalog

### F-2026-001 | NC | Selector Terminology Mismatch

- **First seen:** 2026-05-17 (doc review Run 1)
- **Recurrence:** 4+ documents affected
- **Severity:** HIGH
- **Summary:** "sponsorship_score_z" and "momentum_delta_z" used in external docs; production uses "coinvest_score_z" and "inst_delta_z" since v1.14.0 rename
- **Root cause:** Signal renaming in code did not propagate to all documentation layers
- **Affected systems:** selector-ranker, institutional-signal, all .docx exports
- **Resolution:** CON-1 cross-reference note added to selector-ranker skill. GitHub/docx not updated.
- **Prevention rule:** Any signal rename must include a doc-propagation checklist: skill -> GitHub model docs -> .docx exports -> agent prompts
- **Related findings:** C1 (doc review), W6 (RULESET_CHANGELOG lacks naming note)

### F-2026-002 | MA | IC Tooling Scope Conflation

- **First seen:** 2026-05-13 (Spec 095)
- **Recurrence:** Pervasive in all prior IC claims
- **Severity:** CRITICAL
- **Summary:** run_rank_ic_backtest.py measured composite_score IC, not production final_score IC. Composite_rank correlates only 0.25 with actionable_rank.
- **Root cause:** Tool was built to measure one signal; was silently assumed to measure the production signal
- **Affected systems:** ic-evaluation, selector-ranker, all historical IC claims
- **Resolution:** Spec 100 committed (2faa88e6, 2026-05-17). Prior claims invalidated. Checklist v2 battery rerun deferred post-freeze.
- **Prevention rule:** Any IC measurement tool must explicitly declare which score field it measures in its output header. No IC claim is valid without matching the declared field to the production sort key.
- **Related findings:** Spec 095, Spec 100, G2

### F-2026-003 | DG | inst_delta_z Scope Misattribution

- **First seen:** 2026-05-16 (Code Review H3)
- **Recurrence:** 2 (selector-ranker skill + institutional-signal skill contradicted)
- **Severity:** HIGH
- **Summary:** selector-ranker stated inst_delta_z "excluded from ranker since Spec 051" when it was excluded from the SELECTOR, not the ranker. inst_delta_z remains active in the ranker (NW-t = +3.32).
- **Root cause:** Imprecise language in skill update -- "ranker" used when "selector" was meant
- **Affected systems:** selector-ranker, institutional-signal
- **Resolution:** Code Review H3 fix noted; skill text corrected
- **Prevention rule:** When describing signal status changes, always specify the exact engine layer (selector vs. ranker vs. composite vs. decision engine)
- **Related findings:** C2 (doc review)

### F-2026-004 | PT | AACT Pipeline Timeout

- **First seen:** Pre-May 2026 (recurring)
- **Recurrence:** Multiple (especially Mondays)
- **Severity:** MEDIUM
- **Summary:** 4500s timeout was killing the daily pipeline mid-AACT ingestion, particularly on Monday runs (weekend AACT batch)
- **Root cause:** Timeout set too aggressively for worst-case AACT ingestion duration
- **Affected systems:** screener-ops daily pipeline, catalyst-resolution
- **Resolution:** Timeout increased from 4500s to 6000s
- **Prevention rule:** Monday pipeline runs should have extended timeout or dedicated monitoring. Any timeout change should be validated against 4-week Monday duration distribution.

### F-2026-005 | IF | Herald Digest Extended Outage

- **First seen:** ~2026-04-14 (estimated; 5+ weeks dark as of May 17)
- **Recurrence:** Ongoing (5+ consecutive weeks)
- **Severity:** CRITICAL
- **Summary:** Herald Digest has produced zero output for 5+ consecutive weeks. No deduped or classified JSONL files generated.
- **Root cause:** UNRESOLVED as of 2026-05-18
- **Affected systems:** Herald pipeline, press release monitoring, downstream news-driven signals
- **Resolution:** UNRESOLVED. Hermes mail bridge last tested May 3. 6 stale agents identified May 16 diagnostic.
- **Prevention rule:** Herald should have a max-dark-days SLA (proposed: 3 days) with automatic escalation. See operational-health-baselines skill.
- **Related findings:** G6, weekly signal counts memory

### F-2026-006 | IF | CI Pipeline Extended Red State

- **First seen:** ~2026-05-08
- **Recurrence:** Ongoing (~10 days as of May 18)
- **Severity:** HIGH
- **Summary:** CI has been red since approximately May 8. PR #285 remains open/unmerged. phase2-daily-production cron is dark.
- **Root cause:** Not fully diagnosed. CI Diagnostic Report and CI Fix Checklist produced May 14-16 but remediation not confirmed complete.
- **Affected systems:** All merge gates, production deployment confidence
- **Resolution:** UNRESOLVED. PR #285 open. CI Fix Checklist in biotech-screener collection.
- **Prevention rule:** CI red > 5 days should trigger merge block and operator escalation. See operational-health-baselines skill.
- **Related findings:** C4 (highest operational risk, doc review)

### F-2026-007 | DG | Clinical Score Denominator Confusion

- **First seen:** 2026-05-16 (Code Review H1)
- **Recurrence:** 2 (clinical-scoring skill vs GitHub model docs)
- **Severity:** MEDIUM (RESOLVED)
- **Summary:** GitHub referenced denominator 117; clinical-scoring skill showed 120. Both internally consistent at different layers: GitHub 117 = pre-H1-fix state, skill 120 = post-H1-fix (execution base 12 -> 15).
- **Root cause:** Two fixes applied at different layers without cross-referencing each other
- **Affected systems:** clinical-scoring, GitHub model_documentation_root.md
- **Resolution:** RESOLVED (C7, Run 4). Both references documented as internally consistent.
- **Prevention rule:** When a fix touches a score denominator or weight at one layer, check all other layers that reference the same total.
- **Related findings:** C3/C7 (doc review)

### F-2026-008 | NC | Agent Fleet Count Inconsistency

- **First seen:** 2026-05-17 (doc review Run 1)
- **Recurrence:** 5 different counts across documents (17, 26, 27, 28, 30)
- **Severity:** MEDIUM
- **Summary:** Agent count appears as 17, 26, 27, 28, and 30 across different documents
- **Root cause:** Agent count is a moving target as agents are added/suppressed/retired, and documents capture point-in-time counts without dating them
- **Affected systems:** All governance documents, compliance memos, exec overview
- **Resolution:** agent_governance.md (GitHub, 2026-05-17) designated as most authoritative (30 agents: 27 active + 1 shadow + 1 suppressed + 1 retired). Other docs remain stale.
- **Prevention rule:** Agent count should always be sourced from agent_governance.md with a dated citation. All other documents should say "see agent_governance.md for current count" rather than hardcoding a number.
- **Related findings:** C6/W3 (doc review)

---

## Usage Rules

1. Before investigating a new error, search this catalog by category and keywords first.
2. If a match exists and resolution is documented, apply the known fix before re-investigating.
3. If a match exists but resolution is UNRESOLVED, add recurrence count and any new diagnostic information.
4. New entries require confirmed root cause. Do not add speculative entries.
5. Entries with recurrence_count >= 3 should be considered for prevention rule promotion into the relevant skill document.