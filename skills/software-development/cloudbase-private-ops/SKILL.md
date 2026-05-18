---
name: cloudbase-private-ops
description: Use when operating CloudBase for this user's production/dev projects with project-local MCP isolation, multi-account safety, route-A wrapper, and CLI deprecation rules. Load official cloudbase first, then this overlay; this overlay wins for this profile's production operations.
version: 1.0.0
author: Hermes Agent
license: Private
metadata:
  hermes:
    tags: [cloudbase, mcp, isolation, production, private]
    related_skills: [cloudbase]
---

# CloudBase Private Ops Overlay

## Overview

This is a private overlay on top of the official `cloudbase` skill. Use the official skill for general CloudBase capabilities and product/API references. Use this overlay for this user's production and multi-environment operations.

When rules conflict, this overlay wins for this profile's CloudBase production operations.

## When to Use

Load this skill after `cloudbase` when the task involves:

- Vibe Photoing or meme CloudBase operations;
- any new CloudBase project that shares this machine/profile with other CloudBase projects;
- CloudBase MCP setup, deployment, backfill, DB/storage operations, or environment/account isolation;
- migration from `tcb` CLI to MCP;
- questions about route-A, project-local MCP, mcporter, preflight, diff, rollback, or write gates.

Do not use this overlay as a replacement for official CloudBase development docs. It is an operational safety layer.

## Non-negotiable Rules

- Do not use raw `tcb` for production operations.
- Do not call `npx mcporter call ...` directly for production operations.
- Do not register multiple CloudBase projects as Hermes profile-level MCP servers for production use.
- Do not use `auth action=set_env` or `auth action=logout` to fix environment mismatch.
- Use route-A wrapper only: `scripts/cloudbase-mcp-call.mjs <project> <tool> ...`.
- Every project must have a fixed EnvId, project-local MCP config, project-local credentials, and preflight checks.
- Write operations require explicit user authorization and `--write-ok`.

## Required References

Read as needed:

- `references/route-a-policy.md` — wrapper, blocked actions, required files.
- `references/operation-levels.md` — read-only vs code deploy vs high-risk writes.
- `references/project-template.md` — onboarding a new CloudBase project.
- `references/vibe-meme-example.md` — validated Vibe/meme reference implementation.
- `references/route-a-migration-closeout.md` — how to stop migration-mode over-testing, run the closeout audit, and operate afterward.
- `references/sync-upstream.md` — keeping this overlay aligned with official CloudBase skills.

## Default Production Path

```text
scripts/cloudbase-mcp-call.mjs
  -> cloudbase-preflight.mjs
  -> project config/mcporter.json
  -> project-local CloudBase MCP
  -> fixed EnvId + project-local credentials
```

## Communication Cadence for This User

For long CloudBase migration/ops runs over chat, work in batches and avoid reporting after every small substep unless the user asks for a live trace. Prefer compact checkpoints after meaningful milestones, with evidence paths and pass/fail counts.

If the user says they did not receive the latest message, do not blindly repeat an older status. First distinguish whether any new work happened after the last delivered milestone. If no new action happened, say so plainly; if new action happened, resend only that newest result.

## Operation Levels

Use the least-heavy safe workflow for the operation:

- Level 0 read-only: wrapper call only.
- Level 1 single-function code-only deploy: preflight, function diff, code update, post-check.
- Level 2 config/env/trigger writes: per-operation runbook, rollback, explicit confirmation.
- Level 3 DB/storage/backfill/destructive writes: dry-run/read-only counts, runbook, rollback, explicit confirmation.

## Verification Checklist

Before reporting success, verify:

- [ ] Target project is in `config/cloudbase-projects.json`.
- [ ] EnvId matches the project and CloudBase namespace.
- [ ] Operation used route-A wrapper, not raw CLI/direct mcporter.
- [ ] Dangerous actions were blocked or explicitly authorized.
- [ ] For writes, post-check confirms function status/health/diff.
- [ ] No real secrets were written into shareable docs or chat output.

## Common Pitfalls

1. Treating MCP availability as permission to skip project isolation. MCP still needs route-A guards.
2. Using `auth.set_env` to switch environments. Fix config or credentials instead.
3. Passing a single function directory as `functionRootPath`. It must be the directory containing function subdirectories.
4. Deploying config/envVariables just to test MCP. Code-only writes are enough for migration validation.
5. Copying live function detail into docs without redacting secrets.
6. Modifying the official CloudBase skill heavily. Keep this private policy in the overlay to reduce upstream sync conflicts.
