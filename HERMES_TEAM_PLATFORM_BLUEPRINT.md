# Hermes Team AI Platform Blueprint

Updated: 2026-05-28
Scope: Move Hermes Agent, team knowledge, Obsidian mirror, and selected projects from local-first work to a production 24/7 Linux server workflow.

## Executive Summary

The target system is not a simple file migration. It is a shared AI operating platform for the team:

- Team members work from local machines using Cursor, Codex, Qwen, Antigravity, VS Code, and SSH.
- The Linux server becomes the canonical runtime and knowledge backend.
- Hermes runs as a shared control plane with API/gateway, workers, scheduler, project context, and audit trails.
- The canonical knowledge source moves to a database. Obsidian becomes a readable and editable mirror with review controls.
- Projects are migrated in batches, starting with 3-5 pilot workloads before moving all 30-40 projects.

## Known Facts

From local inspection and existing server docs:

| Area | Current Evidence |
|---|---|
| Server name | `linux-nat` |
| Tailscale MagicDNS | `linux-nat.tail40e9e7.ts.net` |
| Tailscale IP | `100.70.103.59` |
| Public/server IP in docs | `103.142.150.185` |
| OS in docs | Ubuntu 24.04.4 LTS |
| Docker in docs | Docker 29.3.0 |
| Server resources in docs | about 58 GiB RAM, 785 GB root disk |
| Existing access layer | Tailscale verified, Cloudflare Tunnel verified |
| Existing monitoring | Prometheus, Grafana, Loki, Alertmanager, Lark Alert Bridge |
| Existing secrets tool | Infisical present but historically RAM-heavy |
| Live SSH check | blocked from this local session because no SSH identity is loaded |

## Target Principles

1. Server-first: production state lives on the VPS, not on a single local Mac.
2. Zero-trust access: team access goes through Tailscale SSH and Cloudflare Access, not open admin ports.
3. DB-canonical knowledge: database is the source of truth for shared AI memory and review state.
4. Obsidian mirror: Obsidian remains the human-readable team knowledge surface.
5. Review-before-promote: AI writes go through review queues before becoming durable team knowledge.
6. Role-based credentials: credentials are assigned by role and project, not copied into shared `.env` files.
7. Batch migration: migrate 3-5 pilot projects first, verify, then scale.
8. Every project needs health, logs, backup, rollback, and context.

## Architecture

```text
Team Machines
Cursor / Codex / Qwen / Antigravity / VS Code
        |
        | SSH over Tailscale
        | HTTPS via Cloudflare Access for dashboards
        v
linux-nat VPS
        |
        |-- Access Plane
        |     |-- Tailscale SSH
        |     |-- Cloudflare Tunnel
        |     |-- Cloudflare Access
        |
        |-- Hermes Control Plane
        |     |-- Hermes API / Gateway
        |     |-- Hermes TUI / Web entry
        |     |-- Worker service
        |     |-- Scheduler service
        |     |-- Tool/plugin runtime
        |
        |-- Knowledge Plane
        |     |-- Postgres canonical DB
        |     |-- Obsidian mirror exporter
        |     |-- Review queue
        |     |-- Search/vector index
        |     |-- Context pack generator
        |
        |-- Project Plane
        |     |-- /srv/hermes/projects/<project>
        |     |-- containerized runtime
        |     |-- per-project secrets mapping
        |     |-- per-project logs/backups
        |
        |-- Ops Plane
              |-- monitoring
              |-- audit logs
              |-- backup/restore
              |-- incident runbooks
```

## Access Plane

### Tailscale

Use Tailscale as the primary engineering access path.

Recommended usage:

- SSH from local tools to `linux-nat.tail40e9e7.ts.net`.
- Cursor and VS Code use Remote SSH through Tailscale.
- Codex/Qwen/Antigravity use server workspaces through SSH shells or remote extension workflows.
- Admin services bind to `127.0.0.1` or Tailscale-only interfaces where possible.

Required policy:

- Each teammate has their own SSH key.
- No shared SSH private keys.
- Disable password SSH after rollout is verified.
- Use named Linux accounts or controlled shared service accounts with audit logs.
- Tailscale ACLs should separate admins, maintainers, developers, and read-only users.

### Cloudflare Access

Use Cloudflare Tunnel and Access for browser-facing internal services:

- `hermes.synerry.com` - Hermes Team Cockpit
- `api.hermes.synerry.com` - Hermes API/gateway if needed
- `vault.hermes.synerry.com` - Obsidian mirror or published knowledge surface if needed
- `monitor.hermes.synerry.com` - optional monitoring entry if not already handled elsewhere

Rule:

- Never expose raw Portainer, Grafana, Infisical, Postgres, Redis, or Docker sockets publicly.
- All dashboards require Cloudflare Access identity policy.
- Production service domains should be separate from admin domains.

## Runtime Plane

Recommended services:

| Service | Purpose | Runtime |
|---|---|---|
| `hermes-api` | Shared API/gateway for team tools and dashboards | systemd + venv/container |
| `hermes-worker` | Long-running AI jobs, project scans, migrations | systemd worker |
| `hermes-scheduler` | Timers replacing local launchd jobs | systemd timer |
| `hermes-knowledge` | DB sync, indexing, Obsidian mirror export | worker/timer |
| `hermes-cockpit` | Team dashboard | web service behind Cloudflare |
| `hermes-health` | Single health command and endpoint | CLI + HTTP |

Hermes local macOS `launchd` jobs must be translated to Linux `systemd` timers or cron jobs. For production 24/7, prefer systemd timers because logs, status, restart policy, and dependencies are easier to audit.

## Knowledge Plane

The user approved DB-canonical knowledge with Obsidian as mirror. Recommended model:

```text
Hermes events / team edits / agent findings
        |
        v
Canonical DB
        |
        |-- review queue
        |-- durable memory
        |-- project registry
        |-- decision log
        |-- runbook registry
        v
Obsidian mirror export
        |
        v
Team-readable Markdown vault
```

Recommended DB: Postgres for the shared production platform.

Why not SQLite for the shared platform:

- SQLite is fine for one host and light write concurrency.
- Team edits, dashboard writes, agents, workers, and indexing jobs will create concurrent write pressure.
- Postgres gives role isolation, auditability, backup tooling, and better future growth.

Obsidian remains valuable for:

- human-readable project capsules
- long-lived context packs
- runbooks
- decision records
- handoffs
- team learning material

But AI should not directly write durable Obsidian notes in production. AI should write to review queues or canonical DB records first.

## Project Plane

Target layout:

```text
/srv/hermes/
  runtime/
    hermes-agent/
    services/
    logs/
  knowledge/
    obsidian-mirror/
    exports/
    indexes/
  projects/
    hermes-agent/
    main-server/
    vps-server/
    emailhunter/
    scanlyiq/
  secrets/
    templates/
  backups/
    db/
    projects/
    knowledge/
  ops/
    runbooks/
    health/
```

Every migrated project needs a project card:

| Field | Required |
|---|---|
| Owner | yes |
| Repo/source | yes |
| Runtime type | yes |
| Build command | yes |
| Start command | yes |
| Health check | yes |
| Required secrets | yes |
| Ports/domains | yes |
| Data volumes | yes |
| Backup scope | yes |
| Rollback command | yes |
| Hermes context pack | yes |

## Secrets And Credentials

Target: role-based credential pool.

Recommended roles:

- `owner-admin`: full production, secrets, rollback, approvals
- `platform-admin`: infra and Hermes runtime, no business secrets unless needed
- `project-maintainer`: project deploy and logs, scoped secrets
- `developer`: project workspace and dev/test secrets
- `knowledge-editor`: edit/promote knowledge, no production deploy rights
- `viewer`: read-only docs and dashboards

Credential storage:

- Prefer Infisical if it is stable after resource limits and backup verification.
- If Infisical remains too heavy or unreliable, use a simpler interim model:
  - encrypted `.env.age` or SOPS files in private ops repo
  - server-side decrypted runtime files owned by service users
  - strict file permissions

Never put secrets into:

- Obsidian mirror
- Git-tracked markdown
- project context packs
- agent prompts
- public dashboard payloads

## Observability

The server already has a monitoring stack. Hermes should integrate instead of adding a separate stack first.

Required signals:

- Hermes API health
- worker queue depth
- failed jobs
- scheduler last run
- knowledge export last success
- pending review queue count
- per-project health
- per-project disk growth
- agent/tool error rate
- LLM/API failure rate
- backup age

Minimum production alerts:

- Hermes API down
- worker not processing jobs
- scheduler missed runs
- knowledge DB backup older than threshold
- pending critical review items above threshold
- disk above 80/90/95%
- project health check failing
- secret manager unavailable

## Backup And Restore

Production requirement is 24/7, so backup must include restore testing.

Back up:

- Postgres canonical knowledge DB
- Hermes config
- role/credential metadata, not plaintext secrets unless encrypted
- Obsidian mirror
- project compose files and env templates
- project data volumes
- audit logs where required

Recommended schedule:

- hourly DB WAL or frequent logical snapshot for canonical DB
- daily full DB dump
- daily Obsidian mirror snapshot
- weekly project config backup
- monthly restore drill

Acceptance:

- A backup is not valid until a restore command has been tested.
- Each pilot project must have a documented restore path before team rollout.

## Domain Plan

Suggested domains under `synerry.com`:

| Domain | Purpose | Exposure |
|---|---|---|
| `hermes.synerry.com` | Team cockpit | Cloudflare Access |
| `api.hermes.synerry.com` | API/gateway | Cloudflare Access or internal only |
| `vault.hermes.synerry.com` | Knowledge mirror | Cloudflare Access |
| `health.hermes.synerry.com` | limited status page | Access-controlled |

Avoid exposing admin services directly. Use Tailscale for engineering, Cloudflare Access for browser tools.

## Migration Strategy

Start with a pilot of 3-5 workloads:

1. Hermes Agent
2. HermesAgent knowledge/Obsidian mirror
3. Master WebEngine read-only inventory and VPS staging
4. Master ViberQC read-only inventory and VPS staging
5. VPS Server + Main Server documentation knowledge

WebEngine and ViberQC are intentionally included early because the owner wants them on the VPS and they represent the hardest merge/deploy failure modes. They must not go straight to production. They first go through read-only inventory, branch/source-of-truth cleanup, VPS staging build, health checks, backup, and rollback drill.

Dedicated plan:

- `WEBENGINE_VIBERQC_SAFETY_MIGRATION_PLAN.md`

## Acceptance Criteria

The platform is ready for team pilot when:

- SSH via Tailscale works for at least owner and one teammate.
- `hermes.synerry.com` is protected by Cloudflare Access.
- Hermes API/gateway has a health endpoint.
- Hermes worker and scheduler are managed by systemd.
- canonical knowledge DB has backup and restore test.
- Obsidian mirror export runs without conflicts.
- at least one project can be opened from Cursor Remote SSH.
- pilot project has health, logs, backup, rollback, and context pack.
- role-based credential access is verified.
- monitoring alerts fire to the expected channel.

## Open Decisions

| Decision | Recommended Default |
|---|---|
| Canonical DB | Postgres |
| Container runtime | Docker Compose first, reuse existing server pattern |
| Access for developers | Tailscale SSH |
| Browser dashboard access | Cloudflare Tunnel + Access |
| Secret manager | Infisical if stable, otherwise SOPS/age interim |
| Knowledge writes | DB/review queue first, Obsidian mirror second |
| Migration order | pilot 3-5 projects first |
