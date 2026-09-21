---
name: gbrain-knowledge
description: |
  Browse, search, and update the structured knowledge wiki at ~/brain/.
  This is the canonical structured knowledge layer — Markdown files organised
  by category (people, projects, apps, properties, accounts, conventions,
  concepts, timeline). The agent uses this for durable facts about Sahil's
  world. Updated by the memory-promotion skill weekly, or directly when
  significant new facts emerge.
trigger:
  manual: ["what do you know about", "check brain", "update brain", "knowledge"]
  tags: ["knowledge", "brain", "gbrain", "memory", "structured"]
adoption_status: permanent
---

# GBrain Knowledge Wiki

## What this is

`~/brain/` is a flat Markdown knowledge base — no database, no binary, no service.
Hermes reads and writes it directly via the file tools. It is the canonical structured
knowledge tier, sitting above conversational memory.

## Architecture

| Tier | System | Storage | Access |
|------|--------|---------|--------|
| Conversational | Mnemosyne | `~/.hermes/mnemosyne/data/mnemosyne.db` | Vector + FTS5 recall via tools |
| Structured canonical | GBrain wiki | `~/brain/*.md` | Direct file read/write |
| Human-authored | Obsidian vault | `~/vaults/obsidian-master/` (branch: `main`) | READ ONLY — never write |

DO NOT write to `~/vaults/obsidian-master/`. That is Sahil's space. The vault is checked out to the `main` branch, which has full project specs for CoachOS, MatchdayMaestro, Plenishd, Kick-tionary, Big-Rollerz, and plenishd-price-scraper. When you need app-level detail beyond what brain pages hold, read the vault's project docs — they are the canonical source for feature specs, technical decisions, and architecture.

The vault is synced to the VPS via daily git pull (cron at 07:15) and may be up to 24h behind Sahil's laptop. Treat it as authoritative but possibly stale within that window.

## Directory structure

```
brain/
├── RESOLVER.md              # Filing decision tree
├── accounts/
│   └── connected-accounts.md   # Connected accounts (Gmail, Outlook, Tailscale)
├── apps/
│   └── portfolio.md            # Active app inventory + tech preferences
├── concepts/
│   └── <topic>.md              # Individual fact pages (migrated from memory)
├── conventions/
│   └── infrastructure.md       # Operational conventions and gotchas
├── people/
│   ├── sahil-family.md         # Family info (privacy-sensitive)
│   └── sahil-saghir.md         # Sahil's profile
├── projects/
│   ├── mission-control.md      # Mission Control dashboard build docs (migrated from vault)
│   ├── content-pipeline.md     # Social content engine architecture
│   └── job-hunt.md             # Job search tracking
├── properties/
│   └── sahil-properties.md     # Property and finance plan
├── references/
│   ├── operating-model.md      # KENSEI multi-profile operating model (historical)
│   ├── profile-checklist.md    # Profile implementation checklist reference
│   ├── profile-pilot-log.md    # Pilot A+B results and handoff analysis
│   ├── mission-control-arch.md # Mission Control detailed architecture reference
│   └── backlog-may-2026.md     # Old backlog items for Sahil review
├── timeline/
│   └── kensei-setup.md         # Dated infrastructure events
└── inbox/                      # Unfiled items (creates as needed)
```

`references/` holds historical docs migrated from the vault's old `master` branch. These are static references — update them only when the underlying system changes significantly. `projects/` contains active workstream documents.

## How to read the brain

### Full overview
Read `RESOLVER.md` first, then the files you need based on the RESOLVER decision tree:

- Person info → `people/<name>.md`
- Project/app info → `apps/portfolio.md` or `projects/<name>.md`
- Infrastructure/MCP/auth → `conventions/infrastructure.md`
- Connected accounts → `accounts/connected-accounts.md`
- Property/finance → `properties/sahil-properties.md`
- Timeline/history → `timeline/kensei-setup.md`
- Individual fact → `concepts/<topic>.md`

### Targeted search
If you need something specific, use `search_files(target='content', path='/home/kensei/brain', pattern='keyword')` to find it.

## How to write to the brain

### When to write
- A new durable fact emerges that belongs in structured knowledge (not conversational scratch)
- An existing page needs updating (status changed, new account connected, etc.)
- A promotion candidate is approved by Sahil (see memory-promotion skill)
- A significant infrastructure change (new cron, new MCP, new tool)

### When NOT to write
- Conversational noise ("Sahil is tired", "good question")
- Temporary state (a cron job that ran, a test result)
- Identity/personal information that belongs in Mnemosyne only
- Anything that contradicts an existing page without first flagging it

### How to write
Use the `patch` tool to make targeted edits. NEVER overwrite an entire file unless creating a new page. This prevents accidental data loss.

For new pages, follow this format:
```markdown
---
type: <concept|project|person|concept|timeline>
title: <Page Title>
tags: [tag1, tag2]
---

<Content here>

---

- YYYY-MM-DD: Created by [agent/user]. Reason: [why]
```

## Cross-references

Pages use `[[path/to/page]]` syntax (no `.md` extension) for cross-references.
Example: `[[people/sahil-saghir]]` links to `people/sahil-saghir.md`.

## Privacy rules

- Children's identifying info must NEVER appear in external/public output
- Family information is private — use only for context, never in public content
- Personal finances are private — use only for context
- When in doubt about whether something is private, ask Sahil
