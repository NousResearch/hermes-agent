---
name: denji-governance
description: Governing KENSEI agent profiles — review tiers, worker failure analysis, self-eval cycle, quarterly audits, Profile Change Ledger, and cron wiring.
version: 1.0.0
platforms: [linux]
metadata:
  hermes:
    tags: [governance, denji, profile-review, wfa, self-eval, cron, watchdog]
    related_skills: [kanban-orchestrator, kanban-worker]
---

# Denji — Governance Operations

This skill covers the operational machinery for governing KENSEI agent profiles: performance review tiers, worker failure analysis, self-evaluation cycles, quarterly audits, and the Profile Change Ledger.

## Performance Review Tiers

| Tier | Trigger | Cadence | Action |
|------|---------|---------|--------|
| 1 — Flag-driven | `Refinement-Needed` tag on task output | Immediate (same week) | Denji reviews the profile that produced the flagged output |
| 2 — Rolling weekly | Rotation schedule across all profiles | Weekly (Friday reminder) | Each profile gets a full review at least monthly |
| 3 — Quarterly deep audit | Calendar trigger (1st Mon of Mar/Jun/Sep/Dec) | Quarterly | Full system-wide audit cross-referenced against Logboard trend data |

## Self-Evaluation Protocol

1. Agent writes self-eval using the schema template at `~/.hermes/governance/self-eval-schema.md`
2. Saved to `~/.hermes/governance/logboard/` as `self-eval-{agent}-{YYYY-MM-DD}.md`
3. Denji-reviewer reads and categorises: keep / refine / escalate
4. `Refinement-Needed` entries get Tier 1 review within the week
5. Profile changes go on the Profile Change Ledger with a 2-week follow-up date

## Worker Failure Analysis (WFA)

- **Script**: `~/.hermes/scripts/worker-failure-analysis.py`
- **Cron**: `denji-worker-failure-analysis` — Mon 09:00, `no_agent: true`, delivers to Discord
- **Output**: JSON snapshot in `~/.hermes/governance/logboard/worker-failure-analysis-{timestamp}.json` + `worker-failure-analysis-latest.json`
- **Threshold rules** (from SOUL.md):
  - 2+ `needs_reasoning` on same worker → flag for reasoning bump (off→low)
  - 3+ `decomposition_gap` from same lead → flag lead for profile review
  - No improvement after reasoning bump → revert bump, fix decomposition upstream

### WFA Watchdog Pattern

The WFA script prints `[SILENT]` when there are zero findings, zero integrity failures, and zero scan errors. In `no_agent: true` cron mode, empty stdout means nothing is delivered — this is the correct watchdog convention (stay quiet when healthy, report only when there's an issue).

## Quarterly Audit

- **Script**: `~/.hermes/scripts/quarterly-audit.py`
- **Cron**: `denji-quarterly-audit` — 1st Monday of Mar/Jun/Sep/Dec 09:00, `no_agent: true`, delivers to Discord
- **Produces**: System profile count, kanban health, cron health, WFA trend summary

## Cron Wiring Pitfalls

1. **Script paths must be relative to `~/.hermes/scripts/`**. Absolute paths or paths under `~/.hermes/governance/scripts/` are rejected by the cronjob tool. Copy or symlink governance scripts into `~/.hermes/scripts/`.
2. **`no_agent: true` requires `script` field.** The script IS the job — no `prompt` or `skills` are used when `no_agent` is true.
3. **`deliver` targets.** Use `discord` for governance crons (posts to #governance). Use `local` for follow-up crons that should only report locally.
4. **Profile field.** Always set `profile: denji` for governance crons so they run under Denji's config and use the right toolset.
5. **Quarterly cron schedule.** Cron expression `0 9 1-7 3,6,9,12 1` means "09:00 on days 1-7 of Mar/Jun/Sep/Dec, only if it's a Monday" — this is the standard pattern for "first Monday of the quarter".

## Profile Change Ledger

- **Location**: `~/.hermes/governance/profile-change-ledger.md`
- **Format**: Table with columns: Date | Profile | Trigger | Change | Follow-up date | Outcome
- **Follow-up rule**: Every change gets a 2-week follow-up date. If follow-up shows worse evals → immediate rollback.
- **Append new entries at the bottom** of the table (before the Register section).

## Meta-Tag Taxonomy

- **State tags**: Review-Required, Refinement-Needed, FutureIdea, Decomposition-Gap, Needs-Reasoning, Execution-Error, Scope-Creep
- **Routing tags**: Lead-Octacon, Lead-Remii, Lead-CeeCee, Lead-Quan, Lead-Wesker, Lead-Gojo, Lead-Denji, Lead-Light
- **Priority tags**: P0-Critical through P3-Low
- **Artefact tags**: Artefact-Spec, Artefact-Design, Artefact-Research, Artefact-Audit, Artefact-Code, Artefact-Config, Artefact-Docs, Artefact-Report
- **Full taxonomy**: `~/.hermes/governance/meta-tag-taxonomy.md`

## Key Paths

| Resource | Path |
|----------|------|
| Self-eval schema | `~/.hermes/governance/self-eval-schema.md` |
| Self-eval outputs | `~/.hermes/governance/logboard/self-eval-{agent}-{date}.md` |
| WFA outputs | `~/.hermes/governance/logboard/worker-failure-analysis-*.json` |
| WFA latest | `~/.hermes/governance/logboard/worker-failure-analysis-latest.json` |
| Profile Change Ledger | `~/.hermes/governance/profile-change-ledger.md` |
| Meta-Tag taxonomy | `~/.hermes/governance/meta-tag-taxonomy.md` |
| Cron scripts (canonical) | `~/.hermes/scripts/` |
| Governance scripts (source) | `~/.hermes/governance/scripts/` |

## References

- [cron-setup-2026-05-28.md](references/cron-setup-2026-05-28.md) — Session log: creating the three governance crons, job IDs, script locations, key lesson