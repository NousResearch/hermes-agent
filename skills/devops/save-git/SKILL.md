---
name: save-git
description: "Pre-merge/pre-deploy safety gate. Run one real 5-stage Git/GitLab/VPS check before push, merge, or deploy and return one decision token plus a Grid that names the real blocking layer, so the owner clicks merge only when it is truly safe. Use whenever the task involves push, merge, MR/PR, deploy, or 'is production ready'."
version: 2.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [git, gitlab, devops, deploy, vps, merge, safety-gate, ci-cd]
    related_skills: [kanban-orchestrator]
---

# Save Git — 5-Stage Merge/Ship Safety Gate

## Overview

This skill stops the "merge first, find out later" loop. Instead of letting GitLab merge be trial-and-error, you run ONE gate that checks everything, then report a single decision. The human owner clicks merge; the agent never merges — it only certifies.

It was rebuilt (v2) after 11 real projects lost hours to merge/push-to-production loops. It is the Hermes Agent native twin of the cross-AI prompt shortcut `Use Save Git` (Obsidian `skills/prompt-shortcuts/references/use-save-git.md`). Both drive the same engine.

`gate` = ด่านตรวจ · `decision token` = คำตัดสินคำเดียว · `Grid` = ตารางสรุปทุกชั้น · `SHA` = เลข commit เทียบโค้ด · `dry-run` = ลองรันบน VPS โดยยังไม่แตะ production

## When to use

Trigger this skill when the task involves any of: `git push`, opening or merging an MR/PR, deploy/ship to a server, or answering "is this ready for production / can I merge". It works on ANY project the agent is operating in.

## The engine (bundled, runs on any project)

`SKILL_DIR` is the folder holding this SKILL.md.

```bash
# before merge: stages 1-3 (local, mr sanity, ci)
python3 ${HERMES_SKILL_DIR}/scripts/save_git_gate.py --stage merge-gate

# before deploy: stages 1-5 (adds vps dry-run + production)
python3 ${HERMES_SKILL_DIR}/scripts/save_git_gate.py --stage ship-gate

# single stage
python3 ${HERMES_SKILL_DIR}/scripts/save_git_gate.py --stage local   # add --fast to skip build/test (hook use)
python3 ${HERMES_SKILL_DIR}/scripts/save_git_gate.py --stage mr
python3 ${HERMES_SKILL_DIR}/scripts/save_git_gate.py --stage prod --health-url https://example.com/api/health
```

If a browser/localhost URL is open, resolve the serving project first — do not use the shell cwd if the URL points to another project:

```bash
python3 ${HERMES_SKILL_DIR}/scripts/save_git_gate.py --url http://127.0.0.1:7421/ui --stage merge-gate
```

The gate exits 0 only on `SAFE_TO_MERGE` / `SAFE_TO_DEPLOY` / `PRODUCTION_VERIFIED`; otherwise exit 1.

## Per-project adapter · `.savegit.json` (repo root)

The gate reads `.savegit.json` so it knows each project's real stack and commands. Every field is optional; unconfigured stages show `skip`.

```json
{
  "project": "contentfactory",
  "remote_must_match": "contentfactory.git",
  "default_target": "main",
  "stack": "pnpm",
  "checks": {
    "install": "pnpm install --frozen-lockfile",
    "lint": "pnpm lint",
    "typecheck": "pnpm typecheck",
    "build": "pnpm build",
    "test": "pnpm test",
    "audit": "pnpm audit --audit-level high"
  },
  "test_timeout_sec": 600,
  "forbidden_paths": [".env", "*.key", "*.pem"],
  "bundle_globs": ["web/dist/**/*.js"],
  "bundle_must_not_contain": ["localhost:", "127.0.0.1:"],
  "scope_guard": { "max_commits": 5, "max_files": 30 },
  "ci": { "enabled": false },
  "deploy": {
    "branch_only": "origin/main",
    "health_url": "https://aicontentfac.com/api/health",
    "health_commit_field": "commitSha",
    "container_health": { "container": "openclaw", "command": "curl -fsS http://127.0.0.1:18789/health" },
    "vps_service": { "name": "venture-radar-api", "expect_workdir": "/srv/projects/contentfactory/runtime" }
  }
}
```

If a project has no `.savegit.json`, create one first (ask the owner for the stack/commands if unknown — do not guess).

## The 5 stages

1. **Local** — git clean, diff in scope, secret scan, forbidden paths, build/test/lint/typecheck/audit, test timeout split from fail, bundle not pointing to localhost.
2. **MR sanity** — right project/remote, source/target branch correct (never guess GitLab default), branch synced with `origin/<target>`, commit + files within scope_guard (bloat ⇒ likely WRONG TARGET), no conflict.
3. **CI** — latest-commit pipeline passed, not stuck/pending, migration/schema safe. Only enforced when `ci.enabled`.
4. **VPS dry-run** — build candidate in a separate worktree, check env/port/service, run the real `container_health` command (e.g. `docker exec ... curl /health`) — not reading yaml. Do not restart production here.
5. **Production** — deploy from `origin/main` only, deployed SHA = origin SHA, health endpoint returns `commitSha` matching the latest commit (no 200-but-old-commit), service points to the right path, rollback exists.

## Decision tokens (answer with exactly one)

- Merge: `SAFE_TO_MERGE` · `BLOCKED_DO_NOT_MERGE` · `OWNER_DECISION_REQUIRED`
- Deploy: `SAFE_TO_DEPLOY` · `PRODUCTION_VERIFIED` · `PRODUCTION_NOT_VERIFIED`

## Required output

Pass the gate's Grid to the owner verbatim, then in Thai state: decision, the real blocking layer, what to fix, and the owner action (กด merge ได้ / ห้าม merge / ต้องตัดสินใจ). Never say "merge ได้" while Blocking layer is not `none`.

## Autonomous remediation (do not dead-end on BLOCKED)

`BLOCKED_*` is not a final answer — it means stop the risky action and fix the root cause, then rerun the gate. If dirty files are the agent's own safety updates, classify, verify, commit on a separate branch, rerun. If they are someone else's work, do not touch them — isolate with a branch/worktree. Only stop to ask the owner for secret / permission / destructive / business decisions that the agent genuinely cannot make.

## Notes

- The agent runs the gate and reports; the human owner performs the actual merge/deploy click.
- Engine provenance: identical file is mirrored at Obsidian `99-System/scripts/save_git_gate.py` (used by the cross-AI shortcut + git pre-push hooks). Keep the two in sync when editing.
