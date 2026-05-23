# Content Library Promotion Pipeline

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18
**Priority:** 6 of 7 (defer until governance sync is stable)

## Purpose

Define a lightweight process for evaluating Content Library documents and determining whether their knowledge should be promoted into a skill, archived as historical, or left as-is.

---

## Content Classification

### Tier A: Durable Knowledge (Promote to Skill)

Knowledge that is generally applicable across runs, not tied to a specific date, and referenced by multiple agents or processes.

**Current candidates (biotech-screener):**
- Biotech Screener Build Specs (Export Correctness) -- ~410 lines of schema/coverage/export correctness rules. Supplements biotech-validation but is not linked. **Decision needed:** merge into biotech-validation as a companion file, or create a new skill?

**Current candidates (ai-projects):**
- ccft-aware-routing-policy -- defines the 7-tier CCFT routing. Referenced by W4 (tier numbering confusion). Should be internalized into a governance skill or cross-referenced from openclaw-agent-optimize.

### Tier B: Time-Bounded Reference (Keep, Date-Stamp, Review Periodically)

Knowledge correct at a point in time but may go stale. Useful for historical context.

**Current examples:**
- Ranker Research Prep Pack -- ~350 lines including 3 unbuilt scripts for Spec 072. Has a deadline (May 22). After that, actionable content expires.
- CI Diagnostic Report (May 14) -- Root-cause analysis for CI red state. Historical after CI is green.
- CI Fix Checklist (May 16) -- Remediation steps. Same lifecycle.
- Full Sweep Audit Report (May 17) -- Comprehensive audit. Useful as baseline for comparison.
- Hermes Stale Agent Diagnostic Checklist (May 16) -- 6 stale agents identified. Useful until all confirmed resolved.

### Tier C: Reference Library (Archive, No Maintenance)

Background reading not operationally consumed.

**Current examples:**
- ai-agent-trends-2026
- ai-equity-portfolio-management-2026
- ai-family-office-* research digests (6 files)
- weekly-ai-research-digest-* (4 files, including duplicates)
- openclaw-rise-and-fall-timeline
- openclaw-post-collapse-architecture
- local-llm-cost-vs-performance-agents-2026
- frontier-vs-local-llm-performance-2026

### Tier D: Stale / Superseded (Archive or Delete)

- Agent_0.5.2_Dossier_Generator_v1.0.0.zip (Dec 2025) -- Pre-dates current architecture by 5+ months
- WakeRobin_DEM_Executive_Overview (1).docx -- Duplicate of (2).docx
- WakeRobin_Model_Documentation_v1.7.0 (2).docx -- Stale (v1.13.0 when production is v1.14.0)
- DEM compliance memo 17-agent draft -- Not tagged SUPERSEDED per W3

---

## Promotion Criteria

A Content Library document should be promoted to a skill (or companion file) when ALL of the following are true:

1. The knowledge is generally applicable (not tied to a single date or event)
2. It is referenced by 2+ agents or processes
3. It contains rules, thresholds, or procedures (not just narrative)
4. It is actively maintained or has a clear maintenance owner
5. The operator approves the promotion

---

## Housekeeping Rules

1. **Duplicates:** Identify and consolidate. The weekly-ai-research-digest has 4 entries for 2 dates. The .docx files have (1) and (2) copies.
2. **Stale date-stamps:** Any Tier B document older than 30 days without an update should be flagged for review.
3. **Cross-references:** When a skill references a Content Library document, use the document URL, not a prose description. This makes the dependency traceable.
4. **Archive path:** `content://collections/archive/{year}/` for documents no longer active but worth preserving.