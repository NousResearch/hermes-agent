# Pilot Migration Plan

Updated: 2026-05-28
Scope: First 3-5 workloads for moving Hermes Agent, shared knowledge, and selected projects to `linux-nat`.

## Goal

Prove the end-to-end server workflow before migrating all 30-40 projects:

- team can access the server through Tailscale SSH
- Hermes runs as a shared server runtime
- canonical knowledge lives in DB
- Obsidian is generated as a mirror
- selected projects can be opened, tested, monitored, backed up, and rolled back on the server

## Pilot Workloads

Recommended first batch after owner approval on 2026-05-28:

| Order | Workload | Why It Is In Pilot | Risk |
|---:|---|---|---|
| 1 | Hermes Agent | Core AI runtime; everything else depends on it | Medium |
| 2 | HermesAgent knowledge + Obsidian mirror | Proves DB-canonical knowledge model | High |
| 3 | Master WebEngine read-only + staging | Large critical project; must solve old merge/deploy failure modes early | High |
| 4 | Master ViberQC read-only + staging | Large critical SaaS project; known context/deploy complexity | High |
| 5 | VPS Server + Main Server docs | Existing infra knowledge and runbooks support the migration | Low |

Explicitly defer direct production promotion until WebEngine/ViberQC staging gates pass:

| Workload | Reason |
|---|---|
| EmailHunter | good migration candidate, but less important than proving WebEngine/ViberQC safety model |
| ScanlyIQ or Master Content Factory | useful later, after the large-project safety gate exists |
| Tech Tools/AIControlCenter | about 14 GB locally, likely complex; defer until platform guardrails are stable |
| Office Project/Master SynerryNew | about 6.7 GB; defer |

The dedicated detailed plan for WebEngine and ViberQC is:

- `WEBENGINE_VIBERQC_SAFETY_MIGRATION_PLAN.md`

## Phase 0 - Live Baseline

Status: blocked until SSH identity is available in the local SSH agent or explicit access is provided.

Read-only checks to run after access works:

```bash
ssh linux-nat@linux-nat.tail40e9e7.ts.net 'hostname; whoami; uptime'
ssh linux-nat@linux-nat.tail40e9e7.ts.net 'free -h; df -hT / /home /srv 2>/dev/null || true'
ssh linux-nat@linux-nat.tail40e9e7.ts.net 'docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | sed -n "1,80p"'
ssh linux-nat@linux-nat.tail40e9e7.ts.net 'docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | sed -n "1,80p"'
ssh linux-nat@linux-nat.tail40e9e7.ts.net 'systemctl --failed --no-pager'
ssh linux-nat@linux-nat.tail40e9e7.ts.net 'tailscale status'
```

Baseline outputs required:

- CPU count
- RAM total/available
- disk total/free
- swap usage
- container count and top memory consumers
- failed systemd units
- active public/listening ports
- current Cloudflare tunnel process/service
- current backup status

## Phase 1 - Server Workspace Foundation

Target directories:

```text
/srv/hermes/
  runtime/
  projects/
  knowledge/
  backups/
  ops/
```

Required ownership model:

| Path | Owner | Group | Notes |
|---|---|---|---|
| `/srv/hermes/runtime` | `hermes` | `hermes-admin` | runtime services |
| `/srv/hermes/projects` | `hermes` or project owner | project groups | code workspaces |
| `/srv/hermes/knowledge` | `hermes` | `knowledge-editors` | DB exports and mirror |
| `/srv/hermes/backups` | `hermes-backup` | `hermes-admin` | encrypted backup output |
| `/srv/hermes/ops` | `hermes` | `hermes-admin` | runbooks and checks |

Acceptance:

- directory exists with correct ownership
- no world-writable production paths
- backup destination is clear
- project workspaces are separated from existing production app directories

## Phase 2 - Hermes Agent Runtime

Deploy Hermes Agent as a server-managed runtime.

Required components:

| Component | Required Check |
|---|---|
| Python environment | pinned and reproducible |
| Hermes config | server profile separate from local profile |
| logs | written under Hermes/server log path |
| API/gateway | starts and has health check |
| worker | starts and processes a test job |
| scheduler | systemd timer fires and logs success |
| dashboard/TUI path | accessible through SSH or Cloudflare Access |

Suggested systemd units:

- `hermes-api.service`
- `hermes-worker.service`
- `hermes-knowledge.service`
- `hermes-scheduler.timer`

Acceptance:

- `systemctl status` is clean
- restart survives reboot
- logs show no secret leakage
- basic Hermes command works on server
- one test agent action can read project context

## Phase 3 - Knowledge Pilot

Implement DB-canonical knowledge first, then mirror to Obsidian.

Data domains:

| Domain | Purpose |
|---|---|
| `projects` | project registry and metadata |
| `contexts` | context packs and repo adapters |
| `memory_items` | durable knowledge records |
| `review_items` | pending AI/team writebacks |
| `decisions` | ADR-style decisions |
| `handoffs` | current handoff state |
| `runs` | agent run and job traces |

Pilot rules:

- AI writes go to review items.
- Promoted records are stored in DB first.
- Obsidian mirror is generated from DB.
- Human edits can be imported only through a controlled sync/import command.
- Conflicts are reported as review items, not silently overwritten.

Acceptance:

- can create review item
- can approve/promote review item
- promoted item appears in DB
- Obsidian mirror export is generated
- context pack can be loaded by Hermes/Codex/Qwen/Cursor adapters

## Phase 4 - Project Pilot A: Docs And Infra Knowledge

Workloads:

- `Tech Tools/VPS Server`
- `Tech Tools/Main Server`

Purpose:

- make infrastructure knowledge searchable by Hermes
- generate clean server runbooks
- keep existing docs as source/reference while DB becomes queryable

Acceptance:

- docs are indexed
- server stack registry is represented in canonical DB
- runbooks are mirrored to Obsidian
- no secrets from `.env` or logs are imported into knowledge

## Phase 5 - Project Pilot B: Master WebEngine Safety Staging

Why:

- user specifically wants WebEngine on the VPS
- project is large and critical
- previous Hermes memory records branch, CI ownership, and production migration risks
- solving WebEngine safely creates the standard for all remaining large projects

Checklist:

| Item | Required |
|---|---|
| repository source | confirm `synerry_engine.git`, branch, remote default, dirty state |
| dependencies | Node >=22, pnpm 10.30.3, turbo workspace |
| old incidents | branch confusion, `.next/standalone` root ownership, PM2/root ownership, production data risk |
| secrets | map names only into role-based credential pool |
| data volumes | identify DB/uploads/config/process state |
| staging build | build on VPS from fresh clone by commit hash |
| health check | web/admin/API health checks |
| logs | staging logs visible and scanned |
| rollback | rollback target under 5 minutes before production promote |

Acceptance:

- WebEngine remains read-only until project truth map and git gate pass
- VPS staging build passes from a clean clone
- no production deploy from dirty local files
- backup and rollback drill pass before production approval
- comply report is numeric and complete

## Phase 6 - Project Pilot C: Master ViberQC Safety Staging

Why:

- user specifically wants ViberQC on the VPS
- project is large, about 15 GB locally
- root app and `viberqc-central/app` must not be confused
- root app has release/verify/phase comply scripts that should become migration gates
- previous reports show high context/request usage, so context must be compact and controlled

Acceptance:

- ViberQC remains read-only until project truth map and git gate pass
- root app and nested app deployment decision is recorded
- VPS staging build passes from a clean clone
- `verify:release`, `verify:vps`, or equivalent health gates are used
- no production deploy from dirty local files
- backup and rollback drill pass before production approval

## Migration Factory Template

Each future project should follow this template:

```text
1. Inventory
2. Source control check
3. Dependency check
4. Runtime classification
5. Secrets mapping
6. Data/volume mapping
7. Build test
8. Health check
9. Backup test
10. Rollback test
11. Context pack generation
12. Team access verification
```

## Pilot Timeline

| Day | Work |
|---:|---|
| 1 | regain SSH access, live baseline, decide server capacity |
| 2 | create server workspace foundation and Hermes runtime profile |
| 3 | bring up Hermes API/worker/scheduler |
| 4 | implement knowledge DB and Obsidian mirror pilot |
| 5 | migrate infra docs and generate context packs |
| 6-7 | migrate EmailHunter |
| 8-9 | migrate ScanlyIQ or Content Factory |
| 10 | team onboarding, monitoring, backup/restore drill |

## Go/No-Go Gates

Go to next phase only when:

- current phase health check passes
- logs are clean enough to diagnose failures
- backup exists for any stateful change
- rollback is documented
- owner knows what changed

No-go conditions:

- disk above 85% before migration
- swap pressure sustained
- existing production containers unhealthy due to pilot work
- secrets must be copied manually into markdown or chat
- no working backup for stateful service

## Completion Criteria

Pilot is complete when:

- at least 2 team members can SSH through Tailscale
- Hermes Agent runs on server under systemd
- DB-canonical knowledge flow is working
- Obsidian mirror is generated and readable
- EmailHunter and one complex project have health/log/backup/rollback
- team tools can operate from local apps into server workspaces
- migration factory is ready for the remaining 30-40 projects
