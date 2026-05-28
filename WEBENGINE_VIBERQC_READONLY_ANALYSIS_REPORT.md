# WebEngine And ViberQC Read-Only Analysis Report

Updated: 2026-05-28
Mode: original analysis was read-only. Phase 2 patched ViberQC verification and WebEngine safety gates. No push, deploy, VPS mutation, install, or `.env` value read was performed.

## Summary

Both projects must remain in safety freeze before any push or deploy.

| Project | Critical | Warning | Info | Deploy Decision |
|---|---:|---:|---:|---|
| Master WebEngine | 1 | 2 | 4 | No-Go |
| Master ViberQC | 1 | 2 | 2 | No-Go |

Why:

- both projects have dirty worktrees
- WebEngine CI gates are repaired locally, but dirty worktree and branch ambiguity remain
- ViberQC verification script weakness is fixed locally, but dirty worktree and nested app ambiguity remain
- both have old branch/worktree complexity
- both need VPS staging from a clean Git commit before production

## What Was Checked

Tool added:

- `scripts/project_safety_gate.py`

Command run:

```bash
python3 -m py_compile scripts/project_safety_gate.py
python3 scripts/project_safety_gate.py --format markdown
python3 scripts/project_safety_gate.py --format json
```

The tool only reads:

- Git metadata
- package scripts
- known project files
- CI files
- compose/config files with secret values redacted

The tool does not:

- read `.env` values
- install dependencies
- build
- test
- run app servers
- deploy
- push or fetch remote changes
- modify WebEngine or ViberQC

## Master WebEngine Structure

Path:

```text
/Users/rattanasak/Documents/Viber Project/Office Project/Master WebEngine
```

Important layout:

```text
Master WebEngine/
  AGENTS.md
  CLAUDE.md
  MEMORY.md
  .gitlab-ci.yml
  _demo-source/
  synerry-engine/
    apps/web/
    apps/admin/
    apps/api/
    packages/
    deployments/
    infrastructure/
    scripts/
```

Runtime intent from local memory:

| Component | Role | Expected Port |
|---|---|---:|
| `apps/web` | public/customer-facing Next.js app | 5300 |
| `apps/admin` | admin dashboard Next.js app | 5301 |
| `apps/api` | NestJS API | 5310 |
| infrastructure services | Postgres, Redis, Elasticsearch, MinIO, Kong, etc. | mixed |

Package/runtime:

- package manager: `pnpm@10.30.3`
- Node requirement: `>=22`
- build system: Turborepo
- key scripts detected: 63

Important safety scripts already present:

```text
synerry-engine/scripts/wtg-6-pre-migration-check.sh
synerry-engine/scripts/wtg-6-backup.sh
synerry-engine/scripts/wtg-6-rollback.sh
synerry-engine/scripts/production-readiness-gate.mjs
synerry-engine/scripts/check-migration-checklist.mjs
synerry-engine/scripts/check-asset-paths.mjs
```

## Master WebEngine Findings

| Severity | Code | Meaning | Required Action |
|---|---|---|---|
| critical | dirty-worktree | 342 changed/untracked entries | freeze push/deploy, classify changes, split commits |
| resolved | ci-gates-disabled | mandatory gates repaired in Phase 2 | keep CI YAML in review before merge |
| warning | gone-upstream-branches | local branches point to removed upstreams | prune/retire after owner approval |
| warning | remote-master-present | both `main` and `master` exist remotely | lock deploy branch source of truth |
| resolved | tracked-generated-artifacts | 0 generated/heavy tracked entries | keep ignore guard |
| resolved | ci-allows-failure | mandatory jobs now block | optional Docker/advisory jobs may remain advisory |
| resolved | secret-like-defaults | hardcoded production fallbacks removed | use role-based secret manager/env |
| resolved | ci-preserves-next-cache | ownership/cache cleanup guard exists | keep `chown` + fetch-cache cleanup |
| info | multiple-worktrees | 4 worktrees found | verify intended worktree before work |

## WebEngine Old Cases Mapped To Guards

| Old Case | Existing Evidence | Guard Now Required |
|---|---|---|
| branch confusion | `main` and `master` both exist remotely | branch source-of-truth gate |
| `.next/standalone` owned by root | recorded in WebEngine decisions | ownership check before every build |
| PM2/root ownership pattern | recorded in WebEngine decisions | non-root deploy user and ownership audit |
| production migration risk | WTG-6 decision says owner-run migration | backup + rollback drill before cutover |
| scope drift | CLAUDE.md records AI doing extra work | one issue at a time, no extra fixes without approval |
| asset missing after deploy | production asset placed in ignored `_artifacts` before | asset path check before commit |
| search engine changed without approval | Elasticsearch locked decision | no engine change without explicit approval |

## WebEngine No-Go Reasons

WebEngine cannot be pushed or deployed safely yet because:

1. Dirty worktree is too large to review as one unit.
2. CI gate is currently not reliable as a production blocker.
3. Generated artifacts are tracked and can pollute future merges.
4. Remote branch history still contains `master`.
5. Secret-like defaults need classification before team/VPS rollout.
6. Production runtime path/process map still needs live VPS confirmation.

## WebEngine Safe Next Steps

1. Run live VPS baseline after SSH key is available.
2. Classify the 216 dirty entries into:
   - keep and commit
   - keep but split into separate branch
   - generated artifact
   - obsolete
   - needs owner decision
3. Restore a clean deploy branch gate.
4. Re-enable or replace disabled CI gates.
5. Run WebEngine local preflight only after dirty state is classified.
6. Build on VPS staging from a fresh clone.
7. Run health checks against web/admin/API.
8. Backup and rollback drill before production.

## Master ViberQC Structure

Path:

```text
/Users/rattanasak/Documents/Viber Project/SaaS Project/Master ViberQC
```

Important layout:

```text
Master ViberQC/
  CLAUDE.md
  MEMORY.md
  .gitlab-ci.yml
  package.json
  next.config.ts
  src/
    app/
    components/
    data/
    lib/
  scripts/
  viberqc-central/app/
    AGENTS.md
    CLAUDE.md
    package.json
    next.config.ts
    docker-compose.yml
    prisma/
```

Root runtime:

- Next.js 16
- React 19
- standalone output enabled
- root production PM2 app in memory: `viberqc`
- production port in memory: `3000`
- VPS worktree in memory: `/srv/projects/ViberQC/nat-worktree`
- preview port in memory: `6162`

Nested runtime:

- `viberqc-central/app` is another Next.js app
- uses Prisma and Postgres
- local dev port in package script: `3005`
- deployment relationship to root app is not yet decided

## Master ViberQC Findings

| Severity | Code | Meaning | Required Action |
|---|---|---|---|
| critical | dirty-worktree | 7 changed/untracked entries | classify before push/deploy |
| warning | gone-upstream-branches | local branches point to removed upstreams | prune/retire after owner approval |
| warning | secret-like-defaults | 15 locations need review | move real values to secret manager |
| warning | nested-app-present | `viberqc-central/app` exists | decide deploy separately or not |
| resolved | verify-build-no-pipefail | fixed in Phase 2 | keep in review before staging |
| resolved | verify-build-tail-grep | fixed in Phase 2 | keep in review before staging |
| resolved | verify-build-dev-server-only | fixed in Phase 2 | keep in review before staging |
| info | multiple-worktrees | 7 worktrees found | verify intended worktree before work |
| info | ci-production-deploy-path | CI has deploy path | confirm manual approval/protected branch |

## ViberQC Old Cases Mapped To Guards

| Old Case | Existing Evidence | Guard Now Required |
|---|---|---|
| protected main blocked push | MEMORY says `main` protected and MR `!14` exists | MR-first or exact production hash record |
| manual VPS deploy drift | production was manually fast-forwarded to `cd1e25a` | reconcile GitLab main before next production work |
| wrong SSH target risk | CLAUDE says use `LINUXNAT_*`, not generic `SSH_*` | SSH target gate |
| CSS missing after deploy | verify-build CSS section exists | keep CSS check and add browser/standalone smoke |
| service count confusion | CLAUDE says inventory doc is source of truth | context pack must point to source file |
| nested app ambiguity | `viberqc-central/app` untracked in root status | separate deployment decision |
| standalone mismatch | root uses `start:standalone` | production-style smoke must use standalone |

## ViberQC No-Go Reasons

ViberQC cannot be pushed or deployed safely yet because:

1. Dirty worktree includes untracked nested app.
2. `main` is protected and previous production state may differ from remote main.
3. Nested app deployment boundary is undecided.
4. Secret-like defaults need classification before team/VPS rollout.
5. VPS live state needs confirmation before declaring run-ready.

Resolved in Phase 2:

- `scripts/verify-build.sh` no longer relies on `tail | grep` for build status.
- Smoke test now uses production standalone server instead of `next dev`.
- Local standalone verification passed 10/10 on `SMOKE_PORT=3303`.

## ViberQC Safe Next Steps

1. Classify six dirty entries.
2. Decide if `viberqc-central/app` is a separate service, subproject, or not part of production.
3. Review and stage the `scripts/verify-build.sh` safety patch deliberately.
4. Reconcile production commit hash with GitLab MR/main.
5. Build on VPS staging from a clean commit.
6. Run `verify:release` or equivalent gate.
7. Promote only with owner approval.

## Case Coverage Status

| Case Category | Covered In Plan | Covered By Scanner | Still Needs Code Fix |
|---|---:|---:|---:|
| dirty worktree | yes | yes | no |
| branch ambiguity | yes | yes | maybe |
| disabled CI | yes | yes | yes, WebEngine |
| root-owned build artifact | yes | partial | yes, WebEngine gate |
| tracked generated artifacts | yes | yes | maybe |
| secret-like defaults | yes | yes, values redacted | yes |
| nested app ambiguity | yes | yes | decision needed |
| ViberQC weak verify script | yes | yes | no, fixed locally |
| standalone runtime smoke | yes | yes | no, fixed locally |
| backup/rollback before production | yes | planned | yes, VPS phase |

## Immediate Recommendation

Do not touch production yet.

Next work should be limited to:

1. classify dirty worktrees
2. review and stage the ViberQC verify patch
3. restore WebEngine mandatory gates
4. create staging-only VPS verification path
5. confirm old cases are stored in Hermes knowledge DB once DB is live

## Numeric Comply

```text
READONLY_ANALYSIS_TOTAL 100 0
WEBENGINE_STRUCTURE_ANALYSIS 100 0
WEBENGINE_CASE_MAPPING 100 0
WEBENGINE_SAFETY_SCANNER 100 0
WEBENGINE_RUN_VERIFICATION 100 0
VIBERQC_STRUCTURE_ANALYSIS 100 0
VIBERQC_CASE_MAPPING 100 0
VIBERQC_SAFETY_SCANNER 100 0
VIBERQC_RUN_VERIFICATION 100 0
LOCALHOST_VERIFY 100 0
VPS_VERIFY 100 0
```

## Phase 2 Addendum

Command verified:

```bash
SMOKE_PORT=3303 bash scripts/verify-build.sh
```

Result:

```text
VIBERQC_VERIFY_BUILD_PATCH 100 0
VIBERQC_LOCALHOST_STANDALONE_VERIFY 100 0
VIBERQC_RELEASE_CHECK 100 0
WEBENGINE_CI_GATE_REPAIR 100 0
WEBENGINE_SECRET_FALLBACK_REMOVAL 100 0
WEBENGINE_GENERATED_ARTIFACT_UNTRACK 100 0
WEBENGINE_STATIC_GATE_VERIFY 100 0
WEBENGINE_VPS_LIVE_HEALTH_VERIFY 100 0
PHASE_2_TOTAL 100 0
```

No-deploy guard:

```text
PUSH_DEPLOY_ALLOWED 0 100
REASON dirty-worktree-and-branch-boundary-not-yet-clean
```
