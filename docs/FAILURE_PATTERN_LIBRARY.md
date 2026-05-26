# Failure Pattern Library

status: DRAFT / NOT ACTIVE
created: 2026-05-18
updated: 2026-05-26
scope: reference documentation only

## Purpose

This document is a memory index of known operational failure modes. It is a
reference catalog, not an active monitoring system, not automation, and not a
Hermes skill.

Agents should search this catalog before re-investigating recurring failures.
If a match exists and its resolution is documented, apply the known fix before
re-investigating. If a match exists but its resolution is `UNRESOLVED`, update
only `recurrence_count` and diagnostic information; do not add speculative
fixes.

Entries require confirmed root cause before inclusion. Unresolved entries may
only receive recurrence and diagnostic updates until root cause is confirmed.

Prevention rules are reference guidance only. If `recurrence_count >= 3`, an
agent may propose promotion of a prevention rule into the relevant operational
skill, but must not apply that promotion automatically.

## Taxonomy

- DS = Data staleness
- CM = Cache miss / corruption
- PT = Pipeline timeout
- LE = Logic error
- DG = Doc-sync gap
- IF = Infrastructure failure
- GL = Governance lapse
- NC = Naming collision
- MA = Misattribution

## Entry Schema

```yaml
failure_id: F-YYYY-NNN
category: DS | CM | PT | LE | DG | IF | GL | NC | MA
first_seen: YYYY-MM-DD
recurrence_count: N
severity: LOW | MEDIUM | HIGH | CRITICAL
summary: One-line description
root_cause: Confirmed cause, or UNRESOLVED / PARTIALLY RESOLVED
affected_systems:
  - system_or_skill
resolution: Documented fix, UNRESOLVED, or PARTIALLY RESOLVED
prevention_rule: Rule that would prevent recurrence
related_specs:
  - spec_or_empty
related_findings:
  - finding_or_empty
```

## Catalog

### F-2026-001

```yaml
failure_id: F-2026-001
category: NC
first_seen: 2026-05-17
recurrence_count: 4
severity: HIGH
summary: Selector terminology mismatch between external docs and production signal names.
root_cause: Signal renaming in code did not propagate to all documentation layers.
affected_systems:
  - selector-ranker
  - institutional-signal
  - docx exports
resolution: PARTIALLY RESOLVED. CON-1 cross-reference note added to selector-ranker skill; GitHub and docx references not fully updated.
prevention_rule: Any signal rename must include a doc-propagation checklist covering skill docs, GitHub model docs, docx exports, and agent prompts.
related_specs: []
related_findings:
  - C1
  - W6
```

### F-2026-002

```yaml
failure_id: F-2026-002
category: MA
first_seen: 2026-05-13
recurrence_count: 1
severity: CRITICAL
summary: IC tooling measured composite_score IC, not production final_score IC.
root_cause: Tool was built to measure one signal but was assumed to measure the production sort signal.
affected_systems:
  - ic-evaluation
  - selector-ranker
  - historical IC claims
resolution: PARTIALLY RESOLVED. Spec 100 committed at 2faa88e6 and prior claims invalidated; checklist v2 battery rerun deferred post-freeze.
prevention_rule: Any IC measurement tool must declare the measured score field in its output header; no IC claim is valid unless the declared field matches the production sort key.
related_specs:
  - Spec 095
  - Spec 100
related_findings:
  - G2
```

### F-2026-003

```yaml
failure_id: F-2026-003
category: DG
first_seen: 2026-05-16
recurrence_count: 2
severity: HIGH
summary: inst_delta_z was described as excluded from ranker when it was excluded from selector.
root_cause: Imprecise skill wording used "ranker" where "selector" was meant.
affected_systems:
  - selector-ranker
  - institutional-signal
resolution: RESOLVED. Code Review H3 fix noted and skill text corrected.
prevention_rule: When describing signal status changes, specify the exact engine layer: selector, ranker, composite, or decision engine.
related_specs: []
related_findings:
  - C2
```

### F-2026-004

```yaml
failure_id: F-2026-004
category: PT
first_seen: 2026-05-01
recurrence_count: 2
severity: MEDIUM
summary: AACT ingestion exceeded daily pipeline timeout, especially on Monday runs.
root_cause: Timeout was set too aggressively for worst-case AACT ingestion duration.
affected_systems:
  - screener-ops daily pipeline
  - catalyst-resolution
resolution: RESOLVED. Timeout increased from 4500s to 6000s.
prevention_rule: Monday pipeline runs should have extended timeout or dedicated monitoring; any timeout change should be validated against four-week Monday duration distribution.
related_specs: []
related_findings: []
```

### F-2026-005

```yaml
failure_id: F-2026-005
category: IF
first_seen: 2026-04-14
recurrence_count: 5
severity: CRITICAL
summary: Herald Digest produced no output for more than five consecutive weeks.
root_cause: UNRESOLVED
affected_systems:
  - herald pipeline
  - press release monitoring
  - downstream news-driven signals
resolution: UNRESOLVED. Hermes mail bridge was last tested May 3; six stale agents were identified in the May 16 diagnostic.
prevention_rule: Herald should have a max-dark-days SLA, proposed at three days, with operator escalation.
related_specs: []
related_findings:
  - G6
  - weekly signal counts memory
```

### F-2026-006

```yaml
failure_id: F-2026-006
category: IF
first_seen: 2026-05-08
recurrence_count: 10
severity: HIGH
summary: CI remained red for approximately ten days and blocked deployment confidence.
root_cause: UNRESOLVED
affected_systems:
  - merge gates
  - production deployment confidence
  - phase2-daily-production cron
resolution: UNRESOLVED. CI diagnostic report and fix checklist were produced May 14-16; remediation not confirmed in this entry.
prevention_rule: CI red for more than five days should trigger merge block and operator escalation.
related_specs: []
related_findings:
  - C4
```

### F-2026-007

```yaml
failure_id: F-2026-007
category: DG
first_seen: 2026-05-16
recurrence_count: 2
severity: MEDIUM
summary: Clinical score denominator documentation differed across layers.
root_cause: Two fixes were applied at different layers without cross-referencing each other.
affected_systems:
  - clinical-scoring
  - GitHub model documentation root
resolution: RESOLVED. C7 / Run 4 documented both references as internally consistent at their respective layers.
prevention_rule: When a fix touches a score denominator or weight in one layer, check every other layer that references the same total.
related_specs: []
related_findings:
  - C3
  - C7
```

### F-2026-008

```yaml
failure_id: F-2026-008
category: NC
first_seen: 2026-05-17
recurrence_count: 5
severity: MEDIUM
summary: Agent fleet count appeared as 17, 26, 27, 28, and 30 across documents.
root_cause: Agent count is a moving target and documents captured point-in-time counts without dating or source citation.
affected_systems:
  - governance documents
  - compliance memos
  - executive overview
resolution: PARTIALLY RESOLVED. GitHub agent_governance.md dated 2026-05-17 was designated most authoritative; other docs may remain stale.
prevention_rule: Agent count should be sourced from the current registry or canonical governance document with a dated citation; other docs should avoid hardcoded counts.
related_specs: []
related_findings:
  - C6
  - W3
```

## Usage Rules

1. Search this catalog by category and keywords before investigating a new
   operational error.
2. If a match exists and resolution is documented, apply the known fix before
   re-investigating.
3. If a match exists but resolution is `UNRESOLVED`, update only
   `recurrence_count` and diagnostic information.
4. New entries require confirmed root cause.
5. Do not add speculative fixes.
6. Entries with `recurrence_count >= 3` may be proposed for prevention-rule
   promotion into a relevant skill.
7. Do not promote prevention rules into skills automatically.
8. Do not make this document a Hermes skill without a separate approval.

## Non-Goals

- This is not a live monitor.
- This does not trigger cron jobs.
- This does not mutate `.learnings/`, knowledge ledgers, SOUL files, or skills.
- This does not replace postmortems, held-spec ledgers, or operator approval.
