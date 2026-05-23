# Skill Maturity Metadata and Freshness Tracking

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18
**Priority:** 7 of 7 \(last; do not automate freshness checks until the skill set is stable\)

## Purpose

Define a lightweight metadata model for each skill so that agents and operators can assess skill maturity, freshness, and known gaps at a glance.

---

## Metadata Schema \(Conceptual Only\)

This schema is a design artifact. It does NOT imply adding frontmatter to existing skills or creating an automated freshness system.

```
skill_name: [slug]
last_reviewed: YYYY-MM-DD
last_substantive_change: YYYY-MM-DD
maturity: DRAFT | BETA | STABLE | MATURE
known_open_issues: [list of finding IDs]
downstream_dependencies: [skills or agents that consume this skill]
upstream_sources: [Layer 0/1 documents this skill derives from]
refresh_sla_days: N
operational_state_freshness: YYYY-MM-DD
```

---

## Maturity Definitions

| Level | Meaning | Criteria |
| --- | --- | --- |
| DRAFT | Initial structure, not validated | Has purpose and schema but incomplete or unverified |
| BETA | Functional but has known gaps | Operational but with open issues or pending validation |
| STABLE | Production-grade, validated | Reviewed, cross-referenced, no critical open issues |
| MATURE | Stable + battle-tested | 3+ months in use, no critical issues, operator-confirmed |

---

## Current Skill Maturity Assessment \(Manual, as of 2026-05-18\)

### Biotech Screener Pipeline Skills

| Skill | Last Modified | Last Reviewed | Maturity | Open Issues | Refresh SLA |
| --- | --- | --- | --- | --- | --- |
| financial-health | 2026-05-14 | 2026-05-17 | STABLE | L8, M1 applied | 30 days |
| clinical-scoring | 2026-05-09 | 2026-05-17 | STABLE | H1 applied, C7 resolved | 30 days |
| biotech-validation | 2026-05-17 | 2026-05-17 | STABLE | H2 applied, W7 documented | 30 days |
| institutional-signal | 2026-05-14 | 2026-05-17 | STABLE | Ops state needs refresh | 14 days \(13F cycle\) |
| catalyst-resolution | 2026-05-14 | 2026-05-17 | BETA | Spec 097/098 monitoring, EV gates pending | 14 days |
| screener-ops | 2026-05-17 | 2026-05-17 | STABLE | C4 \(CI red\), G6 \(stale agents\) | 7 days |
| ic-evaluation | 2026-05-17 | 2026-05-17 | BETA | Spec 100 pending, ranker IC unmeasured | 14 days |
| selector-ranker | 2026-05-14 | 2026-05-17 | STABLE | C1 naming, C2 scope \(both fixed\) | 30 days |

### Investment Framework Skills

| Skill | Last Modified | Last Reviewed | Maturity | Open Issues | Refresh SLA |
| --- | --- | --- | --- | --- | --- |
| pe-pacing | 2026-05-09 | 2026-05-17 | STABLE | G4: approaching 30-day stale \(\~Jun 8\) | 30 days |
| sfo-liquidity-architecture | 2026-05-09 | 2026-05-17 | STABLE | G4: approaching 30-day stale \(\~Jun 8\) | 30 days |
| spending-liquidity | 2026-05-09 | 2026-05-17 | STABLE | G4: approaching 30-day stale \(\~Jun 8\) | 30 days |

### Agent Infrastructure & Meta Skills

| Skill | Last Modified | Last Reviewed | Maturity | Open Issues | Refresh SLA |
| --- | --- | --- | --- | --- | --- |
| memory-steward | 2026-05-09 | 2026-05-17 | STABLE | None | 60 days |
| openclaw-agent-optimize | 2026-05-15 | 2026-05-17 | STABLE | External landscape section time-sensitive | 30 days |
| self-improving | 2026-05-09 | 2026-05-17 | BETA | No structured correction history; dual-store reconciliation undefined | 30 days |

### Self-Improvement Skills \(NEW, this session\)

| Skill | Created | Maturity | Open Issues | Refresh SLA |
| --- | --- | --- | --- | --- |
| operational-health-baselines | 2026-05-18 | DRAFT | Thresholds need operator calibration | 14 days |
| failure-patterns | 2026-05-18 | DRAFT | 8 seeded entries; needs ongoing maintenance | 14 days |
| document-lineage | 2026-05-18 | DRAFT | Per-fact authority table needs validation | 30 days |
| decision-audit-trail | 2026-05-18 | DRAFT | 2 entries need rationale backfill | 30 days |
| town-hermes-feedback | 2026-05-18 | DRAFT | Design only, no implementation | 60 days |
| content-promotion-pipeline | 2026-05-18 | DRAFT | Tier classifications need operator review | 60 days |
| skill-maturity-metadata | 2026-05-18 | DRAFT | This skill -- self-referential | 30 days |

---

### Domain Skills \(NEW, this session\)

| Skill | Created | Maturity | Domain | Open Issues | Refresh SLA |
| --- | --- | --- | --- | --- | --- |
| backtest-framework | 2026-05-18 | DRAFT | Quant finance methodology | Needs codebase cross-validation | 30 days |
| sec-edgar-mechanics | 2026-05-18 | DRAFT | SEC filing ingestion | Dedup state in memories, not structured | 14 days \(filing cycles\) |
| coding-standards | 2026-05-18 | DRAFT | Dev workflow & conventions | Cross-repo consistency not verified | 30 days |
| trade-execution | 2026-05-18 | DRAFT | Portfolio action bridge | Shadow portfolio mechanics need code review | 30 days |
| regime-detection | 2026-05-18 | DRAFT | Market regime framework | No codebase yet; research-only | 60 days |
| real-estate-intel | 2026-05-18 | DRAFT | Wake Robin core business | Low domain data in system; needs web research seeding | 60 days |
| asset-allocation-methodology | 2026-05-18 | DRAFT | Strategic allocation | 567KB MODEL\_DOCUMENTATION.md not yet distilled | 30 days |
| options-derivatives | 2026-05-18 | DRAFT | Options & vol analysis | Dead lanes documented; limited active use case | 60 days |
| hermes-runtime | 2026-05-18 | DRAFT | Agent runtime mechanics | Runtime internals from 83-file repo, partially covered | 30 days |
| performance-attribution | 2026-05-18 | DRAFT | Return measurement | performance-validation repo is empty shell | 60 days |
| ai-landscape-monitoring | 2026-05-18 | DRAFT | AI industry intelligence | Evaluation framework needs operator calibration | 30 days |

---

## Freshness Rules \(Manual Only\)

1. Check `operational_state_freshness` date when citing any Section 2 data from a skill
2. If older than the skill's `refresh_sla_days`, operational data is potentially stale -- verify against production before citing
3. Investment-framework skills \(pe-pacing, sfo-liquidity, spending-liquidity\) have a 30-day SLA and were last modified May 9 -- they hit the stale threshold around June 8

---

## What This Skill Does NOT Do

- Does NOT add metadata frontmatter to existing skills
- Does NOT create an automated freshness cron
- Does NOT modify any agent prompts
- Does NOT create CI checks or hooks
- It is a manual reference table maintained by the operator or during periodic reviews