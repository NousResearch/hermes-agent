# Self-Heal Coding Agent Setup

This document describes the setup and mechanics of the Self-Heal CI pipeline.

## Overview
The Self-Heal Agent continuously adapts to project drift by automatically running linters, formatters, lockfile updates, and type checks on failing CI runs or via a periodic schedule.

## Triggers
1. **Scheduled:** Runs on a dynamic schedule based on telemetry (PR churn/failure rates).
2. **Reactive:** Triggers on CI failure (via `workflow_run` on `Tests`).
3. **Manual:** Triggered via `workflow_dispatch`.

## Self-Scheduling
The schedule cadence is dynamic. The `.github/workflows/compute-schedule.yml` workflow recalculates the necessary frequency weekly based on PR merge/churn telemetry and updates `.github/self-heal-schedule.yml`.
* **Tiers:** High-churn (daily), Standard (weekly).

### Overriding the Schedule
To manually override the cadence:
1. Open `.github/self-heal-schedule.yml`.
2. Edit the `cron: "..."` string directly.
3. The marker `# AUTO-UPDATED` allows the system to update around it, but if you change the value, the system will respect your edits as long as your changes match the valid cron pattern.

## Reviewer Checklist for Self-Heal PRs
When the agent opens a PR labeled `self-heal`, verify:
- [ ] No secrets or high-entropy patterns are in the diff.
- [ ] Modifications are confined to allowed paths (e.g., lockfiles, formats).
- [ ] CI workflow logic remains intact.
- [ ] Tests successfully passed on the new branch.

## Architecture
The system relies on idempotent, side-effect-free scripts inside `scripts/`, generating safe edits and relying on GitHub Actions to securely open PRs without direct branch pushes.
