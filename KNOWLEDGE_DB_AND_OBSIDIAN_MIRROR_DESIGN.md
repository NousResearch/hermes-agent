# Knowledge DB And Obsidian Mirror Design

Updated: 2026-05-28
Scope: Shared knowledge architecture for Hermes Agent team usage on `linux-nat`.

## Decision

Use a database as the canonical knowledge source. Use Obsidian as a human-readable mirror and editing surface with controlled import/promotion.

This matches the user's direction:

- DB can be canonical.
- Obsidian can be mirror.
- team members may adjust notes.
- owner/team should still control what becomes durable knowledge.

## Why This Model

Direct multi-user Obsidian editing plus AI writeback is high-risk:

- many agents may write at once
- markdown conflicts are hard to merge semantically
- AI may overwrite human context
- secrets can leak into notes if there is no intake boundary
- project context becomes inconsistent across tools

The DB-canonical model gives:

- clear ownership of durable records
- audit trail
- review queue
- conflict detection
- role-based access
- easier indexing/search
- structured export into Obsidian

## Data Flow

```text
Agent finding / team note / project event
        |
        v
Intake
        |
        |-- validate
        |-- classify
        |-- redact secrets
        |-- assign project/owner
        v
Review Queue
        |
        |-- approve
        |-- reject
        |-- needs-edit
        v
Canonical DB
        |
        |-- search index
        |-- context pack generator
        |-- Obsidian mirror exporter
        v
Obsidian Mirror
```

## Core Entities

### projects

Represents one project/workspace.

Suggested fields:

| Field | Purpose |
|---|---|
| `id` | stable project id |
| `slug` | filesystem and URL-safe name |
| `name` | display name |
| `category` | SaaS, Customer, Office, Private, Tech Tools |
| `source_path` | original local/server path |
| `repo_url` | Git remote |
| `owner_role` | owner/maintainer role |
| `runtime_type` | python, node, docker, docs, mixed |
| `status` | pilot, active, archived, blocked |
| `created_at` | audit |
| `updated_at` | audit |

### knowledge_items

Durable records approved for reuse.

Suggested fields:

| Field | Purpose |
|---|---|
| `id` | stable id |
| `project_id` | related project |
| `type` | memory, decision, runbook, context, issue, handoff |
| `title` | human title |
| `body` | canonical markdown or structured text |
| `source` | agent, human, import, system |
| `confidence` | low, medium, high |
| `status` | active, superseded, archived |
| `tags` | search/filter |
| `created_by` | user/agent |
| `approved_by` | user |
| `approved_at` | audit |

### review_items

Unapproved incoming knowledge.

Suggested fields:

| Field | Purpose |
|---|---|
| `id` | stable id |
| `project_id` | target project |
| `title` | proposed title |
| `body` | proposed content |
| `source` | agent/human/import |
| `risk_level` | low, medium, high |
| `detected_secrets` | boolean or redaction summary |
| `conflicts_with` | related knowledge ids |
| `status` | pending, approved, rejected, needs-edit |
| `reviewer` | assigned reviewer |

### context_packs

Generated compact context for agents.

Suggested fields:

| Field | Purpose |
|---|---|
| `id` | stable id |
| `project_id` | related project |
| `version` | generated version |
| `content` | compact markdown |
| `source_hash` | detect stale generation |
| `generated_at` | audit |

### decisions

Architecture and operational decisions.

Suggested fields:

| Field | Purpose |
|---|---|
| `id` | DEC-xxx |
| `project_id` | related project |
| `decision` | what was decided |
| `reason` | why |
| `alternatives` | rejected options |
| `status` | proposed, accepted, superseded |
| `decided_by` | owner/reviewer |

### runs

Agent jobs and automation executions.

Suggested fields:

| Field | Purpose |
|---|---|
| `id` | run id |
| `project_id` | target project |
| `agent` | codex/qwen/hermes/etc |
| `task` | summary |
| `status` | started, succeeded, failed |
| `logs_ref` | log path or object key |
| `created_at` | audit |
| `completed_at` | audit |

## Obsidian Mirror Layout

Generated mirror path:

```text
/srv/hermes/knowledge/obsidian-mirror/
  MOC.md
  projects/
    hermes-agent/
      README.md
      active-memory.md
      decisions.md
      handoff.md
      runbooks/
    main-server/
    emailhunter/
    scanlyiq/
  review-queue/
    pending.md
  reports/
  indexes/
```

Rules:

- generated files include a header that marks them as generated
- manual edits should go into designated editable intake files or through the dashboard
- mirror exporter should not overwrite manual-only files
- secrets must be redacted before export

## Human Edits

Because the team should be able to edit notes, use one of two controlled paths.

### Preferred path: dashboard/edit form

Team edits through Hermes UI:

```text
edit form -> review item -> approval -> DB -> Obsidian export
```

### Acceptable path: Obsidian intake folder

Team edits markdown files under:

```text
obsidian-mirror/intake/
```

Then an importer reads them into review items. The importer does not directly overwrite durable knowledge.

## Conflict Policy

When imported Obsidian content conflicts with DB content:

- do not auto-merge
- create review item
- show old value, new value, and source
- require owner/maintainer approval

Conflict examples:

- same decision id with different body
- project owner changed
- runtime command differs
- health check changed
- handoff state diverges

## Secret Redaction

Before any content enters the mirror:

- scan for API-key-like strings
- scan for `.env` style assignments
- scan for private keys
- scan for tokens in URLs
- scan for passwords and credentials

Blocked patterns should create high-risk review items.

Never export:

- `.env`
- logs containing tokens
- SSH keys
- API keys
- database passwords
- OAuth client secrets
- session cookies

## Search And Indexing

Recommended:

- Postgres full-text search for basic text search
- vector index later if needed
- project/category/tag filters
- separate index visibility by role

Indexable content:

- approved knowledge
- project metadata
- runbooks
- decisions
- handoffs
- generated context packs

Do not index:

- rejected review items
- secrets
- raw logs unless redacted
- private credentials

## Context Pack Generation

Every project should have a generated compact context pack.

Minimum sections:

- project identity
- current status
- runtime commands
- important paths
- active decisions
- known risks
- recent handoff
- verification commands
- forbidden actions

Context pack consumers:

- Hermes Agent
- Codex
- Qwen
- Cursor
- VS Code workflows
- Antigravity if it supports project context files

## Approval Roles

| Role | Can Create Review | Can Approve | Can Export Mirror |
|---|---:|---:|---:|
| Owner | yes | yes | yes |
| Platform Admin | yes | yes for ops/project | yes |
| Project Maintainer | yes | yes for own project | no or scoped |
| Knowledge Editor | yes | yes for docs | no |
| Developer | yes | no | no |
| Viewer | no | no | no |

## Backup

Back up in this order:

1. Postgres canonical DB
2. Obsidian mirror
3. generated context packs
4. import/export logs
5. review queue attachments if any

Restore priority:

1. DB
2. Hermes knowledge service
3. context pack generator
4. Obsidian mirror export
5. search index rebuild

## Health Checks

Required health checks:

- DB reachable
- last backup age
- pending review count
- mirror export last success
- context pack generation last success
- redaction scan status
- import conflict count

Example status target:

```text
knowledge_db: ok
last_db_backup_age_hours: < 24
pending_review_items: visible
mirror_export: ok
context_pack_generation: ok
secret_redaction: ok
```

## Open Implementation Choices

| Choice | Recommended For Pilot |
|---|---|
| DB | Postgres |
| Mirror format | Markdown files |
| Import method | review queue only |
| Conflict resolution | manual approval |
| Search | Postgres FTS first |
| Vector index | later after pilot |
| UI | Hermes Team Cockpit |

