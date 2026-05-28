# Team Roles And Access Model

Updated: 2026-05-28
Scope: Roles, responsibilities, and access boundaries for the Hermes Team AI Platform.

## Operating Model

The team will work from local machines using:

- Cursor
- Codex
- Qwen
- Antigravity
- VS Code
- SSH

The server is the shared production and knowledge runtime. Local machines are clients, not the canonical work location.

## Core Roles

### Owner / Platform Sponsor

Primary responsibility:

- final approval for production changes
- final approval for durable knowledge promotion
- domain and identity ownership
- budget and capacity decisions

Key skills:

- product judgment
- infrastructure risk assessment
- approval discipline
- incident decision-making

Access:

- full Cloudflare/Tailscale admin
- full Hermes admin
- full credential approval
- production rollback authority

### Platform Architect

Primary responsibility:

- system design
- migration architecture
- access/runtime/knowledge tradeoffs
- service boundary decisions

Key skills:

- Linux architecture
- Docker/service design
- AI agent workflow design
- database-backed knowledge systems
- production risk management

Access:

- architecture docs
- server read access
- limited admin access as approved
- no blanket secret access unless required

### DevOps / SRE

Primary responsibility:

- Linux baseline
- systemd services
- Docker Compose
- monitoring and alerts
- backups and restore drills
- incident response

Key skills:

- Ubuntu
- Docker
- systemd
- Prometheus/Grafana/Loki
- Nginx/Cloudflare/Tailscale
- backup/restore

Access:

- production shell
- Docker management
- monitoring
- backup system
- scoped secrets needed for deployment

### Security / IAM Engineer

Primary responsibility:

- SSH policy
- Tailscale ACLs
- Cloudflare Access policies
- role-based credential pool
- audit logs
- secret redaction rules

Key skills:

- zero-trust access
- SSH key management
- secret management
- RBAC
- incident containment

Access:

- identity provider/admin tools
- secret manager admin
- audit logs
- no need for app source write access unless also maintainer

### Hermes Core Engineer

Primary responsibility:

- Hermes server runtime
- gateway/API integration
- tool/plugin integration
- context pack loading
- agent job execution
- scheduler migration from local launchd to Linux systemd

Key skills:

- Python
- Hermes internals
- tool registries
- plugins
- gateway patterns
- testing

Access:

- Hermes Agent repo
- Hermes runtime logs
- non-secret config
- scoped service credentials

### Knowledge Architect

Primary responsibility:

- DB canonical knowledge model
- Obsidian mirror structure
- review queue workflow
- project context packs
- knowledge promotion policy

Key skills:

- Obsidian
- markdown information architecture
- database schema design
- search/index design
- prompt/context engineering

Access:

- knowledge DB
- Obsidian mirror
- review queue
- no production app secrets

### Migration Engineer

Primary responsibility:

- project inventory
- dependency mapping
- Docker/build migration
- workspace setup
- health check creation
- migration factory templates

Key skills:

- Node/Python
- Docker Compose
- Git
- environment mapping
- debugging builds

Access:

- assigned project workspaces
- scoped project secrets
- project logs
- no platform-wide secret access

### QA / Release Engineer

Primary responsibility:

- smoke tests
- acceptance criteria
- rollback verification
- deploy gates
- regression checks

Key skills:

- test planning
- CI/CD
- endpoint validation
- release discipline
- incident rollback

Access:

- test/staging environments
- read logs
- release dashboards
- scoped deploy approvals if assigned

### Team Enablement Lead

Primary responsibility:

- onboarding docs
- team workflow training
- editor setup
- support process
- usage standards

Key skills:

- developer experience
- documentation
- training
- workflow simplification

Access:

- docs
- knowledge mirror
- onboarding dashboards
- no production shell unless needed

## Role-Based Credential Pool

Recommended credential groups:

| Credential Group | Who Can Use | Examples |
|---|---|---|
| `platform-admin` | owner, SRE | Cloudflare tunnel, system deploy tokens |
| `hermes-runtime` | Hermes service account | DB URL, queue credentials |
| `knowledge-editor` | owner, knowledge architect | knowledge DB write token |
| `project-maintainer:<project>` | maintainers | project deploy/env secrets |
| `developer:<project>` | developers | dev/test keys |
| `readonly:<project>` | viewers/QA | read-only DB/API tokens |

Rules:

- no shared personal API keys
- no secrets in Obsidian mirror
- no secrets in context packs
- no secrets pasted into agent prompts
- rotate credentials when team membership changes
- log credential access where possible

## SSH Access Model

Recommended:

- Tailscale SSH or normal OpenSSH over Tailscale
- unique SSH key per teammate
- groups determine directory access
- no shared private keys
- no root login for routine work
- sudo limited to admins

Suggested Linux groups:

| Group | Access |
|---|---|
| `hermes-admin` | platform runtime administration |
| `hermes-dev` | project development workspaces |
| `knowledge-editors` | knowledge DB/mirror tools |
| `project-<slug>-maintainers` | scoped project deploy |
| `project-<slug>-developers` | scoped project development |
| `readers` | read-only docs/logs where appropriate |

## Cloudflare Access Model

Use Cloudflare Access for browser tools:

| App | Domain | Policy |
|---|---|---|
| Hermes Cockpit | `hermes.synerry.com` | team login required |
| Hermes API | `api.hermes.synerry.com` | service token or team login |
| Knowledge mirror | `vault.hermes.synerry.com` | team login required |
| Health page | `health.hermes.synerry.com` | admin/maintainer |

Admin services should remain behind Tailscale unless there is a strong need.

## Responsibility Matrix

| Work Item | Owner | Architect | SRE | Security | Hermes Eng | Knowledge | Migration | QA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| access baseline | A | C | R | R | C | C | C | C |
| service layout | A | R | R | C | R | C | C | C |
| credential model | A | C | C | R | C | C | C | C |
| knowledge DB | A | C | C | C | R | R | C | R |
| Obsidian mirror | A | C | C | C | C | R | C | R |
| Hermes runtime | A | C | R | C | R | C | C | R |
| project migration | A | C | C | C | C | C | R | R |
| production rollout | A | R | R | R | R | R | R | R |

Legend:

- R = responsible
- A = accountable
- C = consulted

## Team Workflow

### Normal development

```text
1. Developer connects via Tailscale SSH.
2. Opens assigned project in Cursor/VS Code Remote SSH.
3. Uses Codex/Qwen/Antigravity with server project context.
4. Creates branch and changes.
5. Runs project checks.
6. Opens review request.
7. Maintainer reviews and merges.
```

### Knowledge update

```text
1. Agent or human proposes new knowledge.
2. Item enters review queue.
3. Knowledge editor or project maintainer reviews.
4. Approved item enters canonical DB.
5. Obsidian mirror and context packs regenerate.
```

### Production change

```text
1. Migration/release plan is created.
2. Backup and rollback are confirmed.
3. Change is applied.
4. Health checks run.
5. Monitoring is watched.
6. Outcome is recorded in knowledge DB.
```

## Minimum Onboarding Checklist

For each teammate:

- Tailscale installed and joined
- SSH key registered
- Cloudflare Access login verified
- Cursor or VS Code Remote SSH verified
- project access group assigned
- credential role assigned
- can read knowledge mirror
- can create a review item
- understands no-secret-in-prompt rule

## Access Review Cadence

Recommended:

- weekly during pilot
- monthly after stable rollout
- immediate review when teammate role changes
- immediate rotation after suspected secret leak

## Hard Rules

1. Do not edit production config without backup.
2. Do not expose admin dashboards publicly.
3. Do not put secrets in markdown or prompts.
4. Do not let AI directly promote durable knowledge without review.
5. Do not migrate a project without health, backup, and rollback.
6. Do not give broad credential access when project-scoped access is enough.

