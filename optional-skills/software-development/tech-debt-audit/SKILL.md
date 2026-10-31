---
name: tech-debt-audit
description: Use when ranking tech debt into a cited, rerunnable ledger.
version: 0.1.0
author: rlaope (github.com/rlaope), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [tech-debt, code-quality, maintenance, refactoring, audit, churn]
    category: software-development
    related_skills: [codebase-inspection, simplify-code, requesting-code-review]
---

# Tech Debt Audit Skill

Turn a codebase's accumulated debt into a ranked ledger: observed orientation (manifests, churn,
size, existing gates), findings across nine dimensions each cited to `file:line`, severity × effort
grading, top fixes and quick wins, and a "looks bad but is fine" list. A rerun reconciles against the
previous ledger (RESOLVED / CARRIED / NEW) instead of starting over. The audit fixes nothing and
never recommends a rewrite.

Ported from [rlaope/oh-my-hermes](https://github.com/rlaope/oh-my-hermes) (MIT), `omh-tech-debt-audit`.

## When to Use

- "Audit our tech debt and tell me what to fix first — we have two weeks of cleanup budget."
- "Re-run last quarter's debt audit; what got fixed, what's still there?"
- Before planning a cleanup sprint or a refactor roadmap for a repo or a scoped path.

Don't use for: reviewing one diff (`requesting-code-review`), cleaning code up now (`simplify-code`),
or just counting lines per language (`codebase-inspection`).

## Prerequisites

`git` on PATH and a local clone. `scripts/churn_rank.py` is stdlib Python. Per-stack detectors
(linters, type checkers, dead-code finders, `npm audit`, `pip-audit`, coverage) are optional and named
from the manifests at audit time.

## How to Run

```bash
python3 ${HERMES_SKILL_DIR}/scripts/churn_rank.py --repo . --since "12 months ago" --top 30
python3 ${HERMES_SKILL_DIR}/scripts/churn_rank.py --repo . --path src/ --json
```

## Procedure

1. **Scope.** Repo root or the scoped path list; out-of-scope findings are reported as out of scope,
   never audited silently. Find the previous ledger if one exists.
2. **Orient (observed, not remembered).** Read manifests with `read_file`; run `churn_rank.py` via
   `terminal`; list test suites, type checkers, linters and CI entry points with `search_files`.
   High churn + large size + thin tests is the priority corner. Done when manifests, top churn,
   largest files and gates are named from tool output.
3. **Audit by dimension** (`references/debt-dimensions.md`): architectural decay, consistency rot,
   type/contract gaps, test debt, dependency/config debt, performance/resource debt,
   error-handling/observability debt, security hygiene, documentation drift. Spend effort on the
   priority corner first. Run a detector only when its output will be read; a detector not run is
   not evidence.
4. **Write findings.** One row each: stable `id` (dimension prefix + path + slug), dimension,
   `file:line`, severity (critical/high/medium/low), effort (S/M/L), bounded recommendation. A
   finding without a `file:line` is an open question, kept out of the table.
5. **Rank.** Top fixes by severity; quick wins = severity ≥ medium with effort S. A critical/L
   finding gets a phased plan suggestion, never "rewrite".
6. **Looks bad but is fine.** Generated files, intentional duplication, documented workarounds — with
   the reason, so the next audit does not rediscover them.
7. **Reconcile on rerun.** Map prior ids by dimension + path; mark each RESOLVED (evidence gone —
   name the commit or observed absence), CARRIED (age +1), or superseded; new rows are NEW. Nothing
   prior is silently dropped.
8. **Save** the ledger with `write_file` (e.g. `docs/tech-debt-ledger.md`, or where the user keeps it)
   so the next run can reconcile.

## Pitfalls

- **Debt a gate already catches** is a gate-configuration finding (the gate is off, skipped or
  ignored), not new debt.
- **Churn needs history.** A shallow clone or a squash-only history understates churn; say so.
- **Renames split churn** across old and new paths (`--no-renames`); merge them by hand when a
  top file was recently moved.
- **Carried three audits** is a finding about prioritisation, not about the code.
- The ledger is analysis: not a cleanup, a measured quality gain, or CI evidence.

## Verification

- Orientation cites `churn_rank.py` output and the manifests actually read.
- Every ranked row has id, dimension, `file:line`, severity, effort and a bounded fix.
- The looks-bad-but-fine section exists (may be empty with a sentence why).
- On rerun, every prior id appears as RESOLVED, CARRIED or superseded.
