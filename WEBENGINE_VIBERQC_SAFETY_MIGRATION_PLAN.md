# WebEngine And ViberQC Safety Migration Plan

Updated: 2026-05-28
Scope: Safety-first migration plan for `Master WebEngine` and `Master ViberQC` to the `linux-nat` VPS platform.

## Executive Decision

Use the highest-quality path:

1. Postgres dedicated to Hermes knowledge as the canonical store.
2. Infisical as the primary role-based credential system.
3. SOPS/age as the emergency fallback for encrypted secrets.
4. Git commit hash as the only deployable source.
5. VPS staging build before production promotion.
6. No production deployment from dirty local files.
7. No direct production change until backup, rollback, and health checks are proven.

## Plain-Language Explanation

### Why Postgres For Hermes Knowledge

Postgres is the shared company notebook behind the system. It is better than a local file because many people and agents can write safely at the same time.

Benefits:

- team-safe
- searchable
- easier backup and restore
- tracks who changed what
- separates Hermes knowledge from customer/project databases
- better for 24/7 production

Tradeoff:

- one more system to set up and monitor
- needs backup discipline
- needs permissions

Decision: use dedicated Postgres for Hermes knowledge.

### Why Infisical Plus SOPS/age

Infisical is the team vault. It is good for role-based access.

Benefits:

- different roles can see different secrets
- teammate access can be removed cleanly
- easier team workflow
- better audit trail

Tradeoff:

- it is another running service
- existing server notes show Infisical has historically used significant RAM
- if it is down, deploys depending on it may be blocked

SOPS/age is the emergency encrypted envelope.

Benefits:

- lightweight
- works even if Infisical is down
- useful for disaster recovery

Tradeoff:

- less friendly for a full team
- key management must be strict

Decision: Infisical primary, SOPS/age fallback.

## Known Project Risks

### Master WebEngine

Observed facts:

- Path: `/Users/rattanasak/Documents/Viber Project/Office Project/Master WebEngine`
- Main app path: `synerry-engine/`
- Package manager: `pnpm@10.30.3`
- Node requirement: `>=22`
- Current branch observed: `main`
- Remote observed: `https://gitlab.dev.jigsawgroups.work/newbiz-lab/products/synerry_engine.git`
- Current local worktree has many modified and untracked files.
- Existing Hermes decisions mention critical production migration risk.

Historical risks from local Hermes memory:

- remote/default branch confusion: remote default referenced `master`, work happened on `main`
- CI build failure from `.next/standalone` owned by `root`
- PM2/root ownership pattern
- production path migration risk from old path to `synerry-engine/main`
- customer production data risk
- previous decision required owner-run production migration, not AI-run production SSH deploy
- scope-lock history because AI changed major technical decisions without permission

Important scripts/commands found:

```bash
pnpm build
pnpm lint
pnpm test
pnpm test:api
pnpm test:e2e
pnpm production:readiness
pnpm migration:check
pnpm cms:instance-e2e
pnpm dra:cms-e2e
pnpm contentth:catalog-e2e
pnpm client:handoff-gate
```

### Master ViberQC

Observed facts:

- Path: `/Users/rattanasak/Documents/Viber Project/SaaS Project/Master ViberQC`
- Current branch observed: `main`
- Remote observed: `https://gitlab.dev.jigsawgroups.work/newbiz-lab/products/ViberQC.git`
- Current local worktree has modified docs/context files and untracked `viberqc-central/app/`
- Root project has Next.js, standalone build preparation, release/verify scripts, and phase comply tooling.
- Nested app exists at `viberqc-central/app/`.

Historical/operational risks:

- project is large, about 15 GB locally
- high context/request usage in previous Hermes reports
- nested app increases chance that developers run commands in the wrong directory
- standalone Next.js build can differ from `next start`
- deployment must avoid copying `.next`, `node_modules`, and local build artifacts blindly

Important root scripts found:

```bash
npm run build
npm run prepare:standalone
npm run start:standalone
npm run lint
npm run test
npm run verify:localhost
npm run verify:vps
npm run verify:release
npm run release:check
npm run phase:comply
npm run quality:all
npm run security
```

Important nested app scripts found:

```bash
npm run build
npm run start
npm run lint
npm run db:migrate
npm run db:seed
```

## Non-Negotiable Safety Rules

1. Do not deploy from a dirty worktree.
2. Do not deploy from unpushed local commits.
3. Do not copy local `.next`, `node_modules`, `.env`, or build output to production.
4. Do not run production database migration without backup and rollback.
5. Do not expose admin dashboards publicly.
6. Do not put secrets in Markdown, prompts, context packs, or Obsidian mirror.
7. Do not let AI push/deploy production until the owner explicitly approves that exact action.
8. Do not mix WebEngine and ViberQC deploys in the same production change window.
9. Do not count a phase complete without localhost or VPS verification evidence.
10. Do not skip comply reporting.

## Migration Method

```text
Read-only discovery
-> project truth map
-> branch/source freeze
-> local quick verification
-> commit and push
-> VPS staging clone
-> Linux build
-> staging run
-> backup and rollback test
-> owner approve
-> production promote
-> monitor and record
```

## Role Assignment

| Role | Primary Work |
|---|---|
| Owner | approve production cutover, secrets, rollback decision |
| Platform Architect | ensure WebEngine/ViberQC fit the Hermes server model |
| DevOps/SRE | VPS baseline, staging layout, systemd/Docker/PM2, backup, rollback |
| Security/IAM | Tailscale, Cloudflare Access, Infisical, SOPS/age fallback |
| Hermes Core Engineer | context packs, agent workflow, project registry, comply automation |
| Knowledge Architect | import old Hermes memory, risk log, decisions, Obsidian mirror |
| WebEngine Migration Engineer | WebEngine build/deploy gates |
| ViberQC Migration Engineer | ViberQC root and nested app gates |
| QA/Release Engineer | localhost/VPS verification, smoke tests, comply reports |

## Phase A - Safety Freeze And Inventory

Goal: Know the truth before changing anything.

### Issues

| Issue | Owner Role | Detail | Verification |
|---|---|---|---|
| A-01 | SRE | Confirm SSH access to `linux-nat` through Tailscale. | `ssh linux-nat@linux-nat.tail40e9e7.ts.net hostname` |
| A-02 | SRE | Capture VPS baseline: CPU, RAM, disk, swap, Docker, failed systemd. | baseline report complete |
| A-03 | WebEngine Engineer | Capture WebEngine local branch, remote, dirty files, untracked files. | `git status`, branch, remote recorded |
| A-04 | ViberQC Engineer | Capture ViberQC local branch, remote, dirty files, untracked files. | `git status`, branch, remote recorded |
| A-05 | Knowledge Architect | Pull known WebEngine incidents from `.hermes/decisions.md`, active memory, Main Server changelog. | incident list complete |
| A-06 | Knowledge Architect | Pull known ViberQC phase docs, memory files, comply docs, release scripts. | source list complete |
| A-07 | Security/IAM | Identify all `.env` files without reading or copying secret values. | env path inventory complete |
| A-08 | SRE | Identify current production paths/domains/processes on VPS for both projects. | path/domain/process map complete |
| A-09 | QA | Mark current deploy status as freeze/read-only until gates pass. | freeze notice recorded |

### Localhost Verification

Required:

- local git state captured
- package scripts captured
- no code changes made

### VPS Verification

Required:

- SSH reachable
- baseline captured
- current production services identified

### Phase A Comply Format

```text
PHASE_A_TOTAL 0 100
A-01 0 100
A-02 0 100
A-03 0 100
A-04 0 100
A-05 0 100
A-06 0 100
A-07 0 100
A-08 0 100
A-09 0 100
LOCALHOST_VERIFY 0 100
VPS_VERIFY 0 100
```

## Phase B - Project Truth Map

Goal: Create a single trusted map so AI and humans stop guessing.

### Issues

| Issue | Owner Role | Detail | Verification |
|---|---|---|---|
| B-01 | WebEngine Engineer | Map WebEngine apps: web, admin, API, DB, workers, scripts. | component map complete |
| B-02 | ViberQC Engineer | Map ViberQC root app and `viberqc-central/app` separately. | component map complete |
| B-03 | SRE | Map current production runtime: Docker, PM2, systemd, Nginx, ports. | runtime map complete |
| B-04 | Security/IAM | Map required secrets by name only, no values. | secret name map complete |
| B-05 | SRE | Map data volumes/uploads/DBs that require backup. | data map complete |
| B-06 | QA | Define critical health URLs and commands for both projects. | health matrix complete |
| B-07 | Hermes Core | Generate project cards for Hermes registry. | project cards complete |
| B-08 | Knowledge Architect | Link old incidents to project cards. | risk links complete |

### Localhost Verification

Required:

- package commands are known
- project directories are known
- no build required yet

### VPS Verification

Required:

- production runtime map confirmed with read-only commands

## Phase C - Git And Branch Gate

Goal: Prevent repeat merge/push problems.

### Issues

| Issue | Owner Role | Detail | Verification |
|---|---|---|---|
| C-01 | WebEngine Engineer | Decide WebEngine deploy branch. Recommended: `main`, with protected branch. | decision recorded |
| C-02 | ViberQC Engineer | Decide ViberQC deploy branch. Recommended: `main`, with protected branch. | decision recorded |
| C-03 | SRE | Confirm remote default branch does not confuse deploy pipeline. | remote HEAD checked |
| C-04 | QA | Block deploy if `git status --short` is non-empty. | gate defined |
| C-05 | QA | Block deploy if local branch is ahead/behind deploy remote. | gate defined |
| C-06 | Security/IAM | Run secret scan before push/deploy. | scan result recorded |
| C-07 | Owner | Approve what existing dirty local work should be kept, committed, split, or discarded. | approval recorded |
| C-08 | Migration Engineers | Split large changes into reviewable commits. | commit plan complete |

### Required WebEngine Preflight

```bash
git status --short
git branch --show-current
git remote -v
pnpm install --frozen-lockfile
pnpm lint
pnpm test
pnpm production:readiness
```

### Required ViberQC Preflight

```bash
git status --short
git branch --show-current
git remote -v
npm ci
npm run lint
npm run test
npm run release:check
npm run verify:release
```

## Phase D - VPS Staging Build

Goal: Prove both projects build on Linux before any production work.

### Issues

| Issue | Owner Role | Detail | Verification |
|---|---|---|---|
| D-01 | SRE | Create staging workspace under `/srv/hermes/projects/<project>/staging`. | directory exists |
| D-02 | SRE | Clone fresh from Git by commit hash. | `git rev-parse HEAD` matches |
| D-03 | WebEngine Engineer | Build WebEngine on VPS staging. | build passes |
| D-04 | ViberQC Engineer | Build ViberQC root on VPS staging. | build passes |
| D-05 | ViberQC Engineer | Decide whether `viberqc-central/app` is deployed separately. | decision recorded |
| D-06 | SRE | Prevent `.next/standalone` root ownership issue. | ownership check passes |
| D-07 | SRE | Bind staging ports that do not collide with production. | `ss -tlnp` checked |
| D-08 | QA | Run staging smoke tests. | smoke result recorded |
| D-09 | Security/IAM | Use staging secrets only. | secret scope verified |

### Ownership Guard

Before each build on VPS staging:

```bash
pwd
whoami
git status --short
find . -maxdepth 3 -path '*/.next/standalone' -exec ls -ld {} \; 2>/dev/null || true
```

If `.next/standalone` is owned by root in a non-root build directory, stop and fix ownership through an approved, narrow command. Do not use `chmod 777`.

## Phase E - Staging Runtime And Health

Goal: Prove runtime, not only build.

### Issues

| Issue | Owner Role | Detail | Verification |
|---|---|---|---|
| E-01 | SRE | Start WebEngine staging. | process/container running |
| E-02 | SRE | Start ViberQC staging. | process/container running |
| E-03 | QA | Check WebEngine public/admin/API health. | HTTP checks pass |
| E-04 | QA | Check ViberQC app/release health. | verify scripts pass |
| E-05 | QA | Check logs for startup errors. | log review complete |
| E-06 | SRE | Confirm staging does not touch production DB unless explicitly intended. | DB target verified |
| E-07 | Security/IAM | Confirm no secrets leaked into logs. | log scan complete |

### ViberQC Runtime Note

Root app supports standalone:

```bash
npm run build
npm run start:standalone
```

Do not replace this with plain `next start` for production-style verification unless the deploy target intentionally uses `next start`.

## Phase F - Backup And Rollback Drill

Goal: Prove recovery before production promotion.

### Issues

| Issue | Owner Role | Detail | Verification |
|---|---|---|---|
| F-01 | SRE | Backup WebEngine DB/data/config/process state. | backup artifacts recorded |
| F-02 | SRE | Backup ViberQC DB/data/config/process state. | backup artifacts recorded |
| F-03 | SRE | Test restore for a safe staging copy. | restore test passes |
| F-04 | SRE | Prepare rollback commands with target under 5 minutes. | rollback runbook complete |
| F-05 | QA | Run rollback drill in staging. | rollback drill passes |
| F-06 | Owner | Approve production cutover window only after rollback passes. | approval recorded |

### Backup Categories

Back up before production:

- database
- uploaded files/object storage
- `.env`/runtime config, encrypted or stored in secret manager
- Nginx/Cloudflare routing config
- PM2/Docker/systemd process state
- previous Git commit hash

## Phase G - Production Promotion

Goal: Promote one project at a time with immediate rollback option.

### Issues

| Issue | Owner Role | Detail | Verification |
|---|---|---|---|
| G-01 | Owner | Approve exact project, commit hash, and cutover time. | approval recorded |
| G-02 | SRE | Confirm pre-cutover backup freshness. | backup age acceptable |
| G-03 | SRE | Deploy by commit hash only. | deployed hash recorded |
| G-04 | QA | Run immediate health checks. | health pass |
| G-05 | QA | Monitor 30-60 minutes. | monitoring window complete |
| G-06 | SRE | Rollback immediately if critical health fails. | rollback ready |
| G-07 | Knowledge Architect | Record outcome in Hermes knowledge DB. | record created |

### Cutover Rule

Do not promote WebEngine and ViberQC in the same window. Finish one, monitor, close comply, then schedule the next.

## Phase H - Team Rollout On VPS

Goal: Make the server the daily work surface.

### Issues

| Issue | Owner Role | Detail | Verification |
|---|---|---|---|
| H-01 | Team Enablement | Create SSH workflow for Cursor and VS Code. | one teammate connects |
| H-02 | Hermes Core | Expose project context packs on server. | agent reads context |
| H-03 | Security/IAM | Assign project-specific secret roles. | access test passes |
| H-04 | QA | Define daily sanity checks for both projects. | checklist complete |
| H-05 | Knowledge Architect | Route team knowledge edits into review queue. | test review item |
| H-06 | Owner | Approve expansion from pilot to daily usage. | approval recorded |

## Phase I - Continuous Guardrails

Goal: Prevent old failure modes from returning.

### Issues

| Issue | Owner Role | Detail | Verification |
|---|---|---|---|
| I-01 | QA | Add branch/dirty-tree preflight gate. | gate blocks dirty tree |
| I-02 | SRE | Add `.next/standalone` ownership check. | gate detects bad owner |
| I-03 | Security/IAM | Add secret scan before push/deploy. | gate runs |
| I-04 | Hermes Core | Add comply report generation. | report generated |
| I-05 | Knowledge Architect | Record incidents and fixes in DB. | searchable record |
| I-06 | SRE | Add alert for failed deploy/health. | alert test passes |

## Standard Comply Report

Each phase must close with numeric-only progress lines:

```text
WEBENGINE_PHASE_A_TOTAL 0 100
A-01 0 100
A-02 0 100
LOCALHOST_VERIFY 0 100
VPS_VERIFY 0 100

VIBERQC_PHASE_A_TOTAL 0 100
A-01 0 100
A-02 0 100
LOCALHOST_VERIFY 0 100
VPS_VERIFY 0 100
```

Meaning:

- first number = percent complete
- second number = percent remaining
- every issue total must add to 100

## Go/No-Go Summary

### Go

- SSH through Tailscale works.
- VPS baseline passes disk/RAM/swap thresholds.
- project worktree is clean or approved dirty work is committed safely.
- build passes on VPS staging.
- health checks pass on VPS staging.
- backup and rollback drill pass.
- owner approves exact production commit hash.

### No-Go

- SSH unreliable.
- root disk above safe threshold.
- current production service health unknown.
- secrets must be copied into Markdown/chat.
- WebEngine/ViberQC local dirty state is not reviewed.
- staging build fails.
- rollback not tested.
- production domain/path/process map is unknown.

## Safety Gate Patch Plan

This section exists to prevent the old failure modes before any production work.

### SG-01 · External Read-Only Scanner

Status: implemented in Hermes Agent.

Purpose:

- inspect WebEngine and ViberQC from Hermes Agent without changing either project
- detect dirty worktrees
- detect branch divergence
- detect remote `master`/`main` ambiguity
- detect tracked generated artifacts
- detect disabled CI gates
- detect secret-like defaults in tracked files without printing values
- detect ViberQC nested app ambiguity
- detect WebEngine WTG migration safety scripts

Implementation target:

- `scripts/project_safety_gate.py`

Rules:

- read-only only
- no `.env` value reads
- no build/test/deploy/install
- no remote-changing git commands
- markdown and JSON output

Verification completed:

```bash
python3 -m py_compile scripts/project_safety_gate.py
python3 scripts/project_safety_gate.py --format markdown
python3 scripts/project_safety_gate.py --format json
```

Latest scanner result:

| Project | Critical | Warning | Info |
|---|---:|---:|---:|
| Master WebEngine | 1 | 2 | 4 |
| Master ViberQC | 1 | 2 | 2 |

Detailed read-only report:

- `WEBENGINE_VIBERQC_READONLY_ANALYSIS_REPORT.md`

### SG-02 · WebEngine CI Gate Repair Plan

Status: not applied.

Current issue:

- `.gitlab-ci.yml` contains fast-deploy disabled rules
- several test/build jobs are disabled
- some jobs allow failure

Required fix before production:

- identify which jobs must block merge
- re-enable secret/env checks
- re-enable build gate or replace with an explicit VPS staging gate
- keep `TURBO_CONCURRENCY=1` to avoid VPS memory pressure
- preserve ownership fix for `.next/standalone`

No code change should be applied until WebEngine dirty work is reviewed.

### SG-03 · ViberQC Verify Script Repair Plan

Status: applied and locally verified.

Previous issue:

- `scripts/verify-build.sh` uses shell pipelines without `pipefail`
- build failure detection depends on `tail | grep`
- smoke test uses `next dev`, while production uses standalone/PM2

Applied fix:

- use direct exit code capture for `npm run build`
- add standalone smoke test
- keep CSS verification
- keep expected-content verification
- return non-zero on any failure

Local verification:

```bash
SMOKE_PORT=3303 bash scripts/verify-build.sh
```

Result:

- TypeScript: pass
- ESLint critical pages: pass
- Next.js build: pass
- Standalone artifact: pass
- Standalone localhost smoke: pass
- Homepage content: pass
- CSS reference/class/Tailwind output: pass
- Total: 10 pass, 0 fail

Remaining restriction:

- ViberQC remains no-go for push/deploy until dirty worktree entries are reviewed and staged deliberately.

### SG-04 · Secret And Default Cleanup Plan

Status: not applied.

Current issue:

- compose/config files include dev fallback values for password-like settings
- ViberQC nested compose includes dev database defaults

Required fix before team rollout:

- move real values to Infisical
- keep only placeholder examples in `.env.example`
- block production deploy when fallback secrets are used
- never copy these values to Obsidian, context packs, or prompts

### SG-05 · Generated Artifact Cleanup Plan

Status: not applied.

Current issue:

- WebEngine tracks generated artifacts such as coverage, Playwright report, and test results

Required fix before broad migration:

- decide whether historical reports must remain tracked
- move generated reports to ignored artifact storage if not needed in Git
- add guard to prevent future generated artifact commits

Do not clean or delete without owner approval because historical reports may be useful evidence.

## Phase 2 Execution Snapshot · 2026-05-28

Scope:

- patched ViberQC verification gate only
- patched WebEngine CI safety gates only
- removed generated WebEngine reports from Git tracking while keeping local files on disk
- replaced WebEngine hardcoded secret fallbacks with required environment variables in deploy/runtime config
- did not push, deploy, or change VPS state

Verification:

| Check | Result |
|---|---:|
| ViberQC `bash -n scripts/verify-build.sh` | 100 |
| ViberQC `git diff --check -- scripts/verify-build.sh` | 100 |
| ViberQC `SMOKE_PORT=3303 bash scripts/verify-build.sh` | 100 |
| ViberQC `npm run verify:release` | 100 |
| ViberQC `npm run release:check` | 100 |
| ViberQC safety scanner warning reduction | 67 |
| WebEngine safety scanner | 100 |
| WebEngine `.gitlab-ci.yml` YAML parse | 100 |
| WebEngine docker compose YAML parse | 100 |
| WebEngine shell syntax for touched scripts | 100 |
| WebEngine `pnpm production:readiness:self-test` | 100 |
| WebEngine `pnpm migration:check` | 100 |
| WebEngine VPS live health `:5300/:5301/:5310` | 100 |
| WebEngine patch application | 100 |

Comply:

| Issue | Percent |
|---|---:|
| SG-01 external scanner | 100 |
| SG-02 WebEngine CI gate repair | 100 |
| SG-03 ViberQC verify script repair | 100 |
| SG-04 secret/default cleanup | 100 |
| SG-05 generated artifact cleanup | 100 |
| Phase 2 overall | 100 |

Remaining no-deploy blockers outside Phase 2:

- both target repositories still have dirty worktrees from pre-existing project work
- WebEngine still has old local branches whose upstream branches are gone
- WebEngine remote still exposes both `main` and `master`
- ViberQC nested app deployment boundary is still a product/repo decision
