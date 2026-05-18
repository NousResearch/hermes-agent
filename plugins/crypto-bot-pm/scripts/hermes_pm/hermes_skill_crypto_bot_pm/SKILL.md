---
name: crypto-bot-pm
description: Use when managing the crypto_bot repository through Hermes PM status, read-only Gitea snapshots, completion workstream packets, development slice packets, Kanban proposal packets, forge-plan dry runs, and approval review packets while keeping PM work separate from runtime, trading, secrets, workflows, and deploys.
version: 1.0.0
author: Hermes PM
license: MIT
metadata:
  hermes:
    tags: [project-management, crypto-bot, gitea, kanban, forge-plan]
    related_skills: [kanban-orchestrator]
---

# Crypto Bot PM Bridge

## Overview

Use this skill when the Operator asks Hermes to manage, discuss, triage, or plan work for the `crypto_bot` project from Telegram or another Hermes chat surface.

The first managed project is:

- Project: `crypto_bot`
- Repository: `/Users/preston/robinhood/crypto_bot`
- PM mode: read, summarize, propose, and request approval before writes

Keep a hard line between the Hermes PM platform and the managed product runtime:

- Hermes PM platform: reads project status, summarizes Gitea state, proposes Kanban cards, generates forge dry-run plans, generates approval packets, explains blockers, and asks the Operator before any mutation.
- `crypto_bot` daemon/runtime: trading daemon, broker paths, runtime services, launchd, CI runner execution, workflows, and live or paper trading behavior. These are out of scope for this skill.

## When To Use

Use this skill when the Operator asks for:

- PM status for `crypto_bot`
- Issue #1 lifecycle status after the PM-6B seed issue write
- Gitea issue, PR, check, or blocker summaries
- Proposal-only backlog expansion from existing Issue #1
- Operator-reviewed backlog selection packets for proposed candidates
- Completion-oriented development workstream packets
- Non-mutating development slice packets for the recommended first slice
- Selected-candidate approval scopes for future issue-only writes
- Kanban proposal packets
- Forge-write dry-run plans
- Forge approval review packets
- Capability maps for future forge planning
- Next safe PM action
- Telegram rehearsal of PM workflows

Do not mutate Gitea by default. Do not use this skill to administer the runtime, place trades, run CI, deploy, inspect credentials, or mutate Gitea without a future exact approval.

## Quick Reference

Run commands from `/Users/preston/robinhood/crypto_bot`.

```bash
python3 scripts/hermes_pm/hermes_pm_status.py --repo-root . --live-gitea-read --format json
```

```bash
python3 scripts/hermes_pm/hermes_pm_issue_lifecycle.py --issue-index 1 --expected-title "[Hermes PM] Establish initial PM-managed backlog item" --format json
```

```bash
python3 scripts/hermes_pm/generate_kanban_proposal_packet.py --repo-root . --project-id crypto_bot --live-gitea-read --format json
```

```bash
python3 scripts/hermes_pm/generate_backlog_expansion_proposal.py --repo-root . --project-id crypto_bot --live-gitea-read --issue-index 1 --expected-title "[Hermes PM] Establish initial PM-managed backlog item" --format json
```

```bash
python3 scripts/hermes_pm/generate_backlog_selection_packet.py --repo-root . --project-id crypto_bot --live-gitea-read --issue-index 1 --expected-title "[Hermes PM] Establish initial PM-managed backlog item" --format json
```

```bash
python3 scripts/hermes_pm/generate_development_workstream_packet.py --repo-root . --project-id crypto_bot --live-gitea-read --format json
```

```bash
python3 scripts/hermes_pm/generate_development_slice_packet.py --repo-root . --project-id crypto_bot --format json
```

```bash
python3 scripts/hermes_pm/generate_backlog_candidate_approval_scope.py --repo-root . --project-id crypto_bot --candidate-id pm8-002 --live-gitea-read --format json
```

```bash
python3 scripts/hermes_pm/generate_forge_write_plan.py --repo-root . --project-id crypto_bot --format json
```

```bash
python3 scripts/hermes_pm/generate_forge_approval_packet.py --repo-root . --project-id crypto_bot --format json
```

```bash
python3 scripts/hermes_pm/map_gitea_forge_capabilities.py --base-url http://127.0.0.1:3005 --owner preston --repo crypto_bot --format json
```

If the `crypto-bot-pm` plugin is enabled, prefer the safe plugin tools when they are available:

- `crypto_bot_pm_status`
- `crypto_bot_pm_issue_lifecycle`
- `crypto_bot_pm_kanban_packet`
- `crypto_bot_pm_backlog_expansion`
- `crypto_bot_pm_backlog_selection`
- `crypto_bot_pm_development_workstream`
- `crypto_bot_pm_development_slice`
- `crypto_bot_pm_candidate_approval_scope`
- `crypto_bot_pm_forge_plan`
- `crypto_bot_pm_forge_approval_packet`
- `crypto_bot_pm_capability_map`

The slash command `/crypto-bot-pm-status` may be available after the gateway loads the plugin.

## Procedure

1. Start with PM status.
   Use `crypto_bot_pm_status` or the PM status command above. Summarize the branch, dirty state, latest checkpoint docs, local CI evidence, runner readiness, blockers, warnings, and the next safe PM action.

2. Add Gitea context only through read-only surfaces.
   PM status and Kanban packet generation may include `--live-gitea-read`, which is intended to use read-only Gitea requests. Treat blockers and warnings as evidence, not as permission to write.

3. Check the seed issue lifecycle.
   Use `crypto_bot_pm_issue_lifecycle` or the issue lifecycle command above. Confirm Issue #1 already exists, matches the PM seed title, is not a pull request, and was checked without any Gitea write API.

4. Generate a Kanban proposal packet.
   Use `crypto_bot_pm_kanban_packet` or the Kanban packet command above. Explain proposed cards, blocked items, stale work, PR attention, and approval candidates. A Kanban packet is proposal-only.

5. Generate a proposal-only backlog expansion.
   Use `crypto_bot_pm_backlog_expansion` or the backlog expansion command above. It should propose 3 to 5 PM/platform candidates from existing Issue #1, dedupe the seed issue, block runtime/trading/secret/deploy candidates, and keep all candidates as non-mutating proposals.

6. Generate an Operator-reviewed backlog selection packet.
   Use `crypto_bot_pm_backlog_selection` or the backlog selection command above. It should review the 5 proposal-only candidates, select none by default, recommend one safest first PM/platform candidate, and explain that future issue creation needs the exact candidate ID, exact operation ID, exact plan hash, and explicit approval token.

7. Generate a completion-oriented development workstream packet.
   Use `crypto_bot_pm_development_workstream` or the development workstream command above. It should identify 5 to 8 concrete candidates, distinguish crypto_bot completion work from PM/process mechanics, block trading/broker/runtime/secret/deploy candidates, and recommend the first safe development slice. This packet is proposal-only and performs no writes.

8. Generate a non-mutating development slice packet.
   Use `crypto_bot_pm_development_slice` or the development slice command above. It should default to the workstream's recommended candidate, define allowed and forbidden paths, include a definition of done, and state that future implementation requires explicit Operator branch-write approval.

9. Generate a selected-candidate approval scope when the Operator explicitly focuses `pm8-002`.
   Use `crypto_bot_pm_candidate_approval_scope` or the approval-scope command above. It should propose title/body text, constrain the future operation to exactly one `create_issue`, exclude labels/projects/comments/PRs/workflows/runners/runtime/financial/secret actions, and state that PM-11 needs exact approval before any write.

10. Generate a forge-write dry-run plan.
   Use `crypto_bot_pm_forge_plan` or the forge plan command above. Explain planned operation IDs, target endpoints, payload summaries, hashes, blockers, and why the plan is still non-mutating.

11. Generate the approval packet.
   Use `crypto_bot_pm_forge_approval_packet` or the approval packet command above. Show the review scope, exact plan hash, operation IDs, blockers, and non-action proof. The approval packet is review-only and does not execute a write.

12. Map forge capabilities when planning future writes.
   Use `crypto_bot_pm_capability_map` or the capability map command above. Interpret endpoint readiness, blocked or unknown operation types, and issue-only fallback guidance. Readiness evidence is not approval.

13. Ask the Operator before any future write.
   Before any Gitea mutation, branch write, runner action, workflow trial, deploy, runtime action, secret-adjacent action, or financial action, pause and request explicit Operator approval for the exact plan hash and selected operation IDs.

## Blocker Interpretation

- Dirty worktree: continue with read-only PM summaries and proposal packets; do not overwrite or discard changes.
- Gitea read blockers: summarize unavailable endpoints and continue with local PM status.
- Project endpoint blockers: prefer issue-only fallback planning; do not recommend project cards unless endpoint capability is proven.
- Runner or workflow blockers: report them as future approval gates; do not start runners or workflows.
- Missing token or auth blockers: report the blocker without asking to reveal token values.
- Runtime, trading, broker, or secret blockers: mark out of scope and ask for a separate explicitly scoped checkpoint.

## Forbidden Actions

Do not:

- Place trades or call Robinhood, broker, exchange, live-market, account, order, position, wallet, or other financial APIs.
- Start app servers, trading workers, schedulers, launchd services, qmd, Docker builds, Kubernetes, Flux, Harbor, production services, Hermes gateway, or Gitea runners.
- Run workflows, CI wrappers in normal execution mode, or workflow trials.
- Edit `.gitea/workflows` or create workflow files.
- Deploy.
- Inspect `.env` files, token files, Keychain material, private keys, SQLite runtime DBs, runtime logs, browser credential stores, cookies, caches, generated OTA artifacts, or credential material.
- Invoke `apply_approved_write_plan.py`, `branch_local_writer.py`, or any branch writer against real repo files.
- Create, edit, close, label, comment on, merge, or move Gitea issues, pull requests, labels, milestones, projects, releases, checks, statuses, webhooks, packages, or board cards unless a future checkpoint explicitly approves the exact mutation.
- Push the product repo.
- Change trading, broker, or runtime code.
- Print or store token values.

## Operator Approval Rule

Approval must be explicit, current, and scoped. A future approval must identify the exact plan hash, operation IDs, repo, endpoint class, and expiration rules. Endpoint readiness, a generated approval packet, or a previous dry-run plan is not approval to mutate anything.

When in doubt, answer with the current evidence, the next safe PM action, and the approval question the Operator would need to answer in a future checkpoint.
