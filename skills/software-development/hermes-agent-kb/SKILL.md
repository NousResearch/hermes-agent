---
name: hermes-agent-kb
description: "Hermes Agent KB Pattern — autonomous KB writes, [ACTIVE] project tracking, two-layer memory model, and hybrid search. The operational knowledge system that lets Hermes survive context resets and compound knowledge across sessions."
version: 1.0.0
author: noivan0
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [knowledge-base, memory, kb, session-continuity, active-tags, hybrid-search, rag-alternative, autonomous]
    category: software-development
    related_skills: [llm-wiki, hermes-agent, hermes-ipc-coordination]
---

# Hermes Agent KB Pattern

The operational knowledge system built into Hermes Agent. Based on the
[Agent KB Pattern](https://gist.github.com/noivan0/2c1129a2b8d829be70cab1439d4c6e18),
extended for always-on agent operation across sessions, projects, and HA deployments.

Reference implementation: [nova-oss nova/kb/](https://github.com/noivan0/NOVA/tree/main/nova/kb)

## When This Skill Activates

Load this skill when:
- Setting up KB for a new Hermes instance
- Deciding whether to write to MEMORY.md vs KB
- Implementing `[ACTIVE]` project tracking for a new project
- Recovering context after an unexpected session reset
- Running KB lint / health audit
- Configuring multi-agent KB sync (Primary ↔ Standby)
- Integrating KB writes into a new workflow or cron job

Do NOT use this for general wiki/research KBs — use the `llm-wiki` skill instead.
This skill is specifically for **Hermes Agent's own operational knowledge**.

---

## Architecture: Two-Layer Memory

```
MEMORY.md  (hot cache, injected EVERY turn)
  < 10,000 chars hard limit
  Compressed facts + [[KB links]]
  [ACTIVE] project tags
  Overflows → promote to KB

KB pages  (cold storage, read on demand)
  ~/.hermes/kb/
  Unlimited pages, full detail
  Queryable via hybrid search
  Source of truth for all operational knowledge
```

**Promotion rule** — a MEMORY.md fact graduates to KB when:
1. It exceeds ~200 chars and needs full explanation
2. It has multiple sub-components (root cause / fix / prevention)
3. It's project-specific (not universally relevant every turn)
4. MEMORY.md > 8,000 chars → audit and promote oldest entries

After promotion, MEMORY.md keeps one line:
```
SSL cert fix: REQUESTS_CA_BUNDLE in .env. KB: [[fixes/ssl-cert-error]]
```

---

## KB Directory Structure

```
~/.hermes/kb/
├── SCHEMA.md              # Read this first every new session
├── index.md               # Every page listed here (one-line summary)
├── log.md                 # Append-only. Rotate at 500 entries → log-YYYY.md
├── config/                # Discovered system facts
├── fixes/                 # Bug root causes + solutions
├── projects/              # Per-project operational context
│   └── _registry.md       # Active project list (P0–P3 priority)
├── user/                  # User preferences + recurring patterns
├── weekly/                # YYYY-WNN.md summaries
└── _archive/              # Superseded pages (removed from index)
```

**Search tool:** `python3 /root/.hermes/bin/kb_unified_search.py "query"`
Spans 4 namespaces: KB + project KB + past sessions + skills.

---

## KB Page Format

```markdown
---
title: Descriptive Title
created: YYYY-MM-DD
updated: YYYY-MM-DD
type: config | fix | project | user | concept | weekly
tags: [from-schema-taxonomy-only]
status: active | resolved | archived
---

## [Section heading — H2 = chunk boundary for search]
Content here. Use [[wikilinks]] to cross-reference other pages.
Minimum 2 outbound [[wikilinks]] per page.
```

Every new page → add to `index.md` → append to `log.md`.

---

## [ACTIVE] Tag System

Active projects live in MEMORY.md with a structured tag:

```
[ACTIVE] project-name — current phase. Next: exact next action. KB: [[projects/project-name]]
```

**Rules:**
- Max 5 `[ACTIVE]` entries (oldest rotates to KB on overflow)
- Pause → remove tag, save state to KB "## Current State" section
- Resume → reload KB page, add `[ACTIVE]` tag back
- Complete → remove tag, set KB status: `resolved`, log in weekly

**Example:**
```
[ACTIVE] blog-pipeline — publishing. Next: add Bing ping. KB: [[projects/blog-pipeline]]
[ACTIVE] nova-oss — v1.2.0. Next: update README. KB: [[projects/nova-oss]]
```

---

## Session Continuity Protocol

### Session Start (ALWAYS do this first)

```
1. Scan MEMORY.md → find all [ACTIVE] tags
2. For each [ACTIVE]: read_file kb/projects/{name}.md
3. kb_unified_search("today's task") → load top-2 results
4. Check kb/log.md last 5 entries → know recent changes
→ Context restored. No session replay needed.
```

### During Session

```
- Write KB immediately on discovery (don't batch — crashes happen)
- Update [ACTIVE] tag when phase changes
- When MEMORY.md > 8K → audit + promote to KB
```

### Session End — MANDATORY BEFORE /new

```
1. Each [ACTIVE] project → update kb/projects/{name}.md:
   ## Current State: [exactly where we are]
   ## Next Action: [single most important next step]
2. Update [ACTIVE] tags in MEMORY.md
3. Append to kb/log.md
4. ONLY THEN reset (/new)
```

**Recovery when session-end was skipped:**
```
kb_unified_search("current project state")
read_file kb/projects/_registry.md
→ Reconstruct from last known state
```

---

## Write Triggers (agent records automatically)

| Event | Target | Record |
|-------|--------|--------|
| Endpoint/URL/auth discovered | `config/` | URL, auth method, response format, quirks |
| Error root cause found | `fixes/` | Symptom → root cause → fix → prevention |
| Patch applied | `fixes/` or `config/` | What changed, why, files |
| New project starts | `projects/_registry.md` + `projects/name.md` | Goals, stack, phase |
| Session ending | Active project pages | Current state + exact next action |
| User preference seen 2+ times | `user/` | The pattern, examples |
| Contradiction found | Existing page | Both claims with dates, `contested: true` |
| Week ends | `weekly/YYYY-WNN.md` | Completed, new KB, carry-over |

**Do NOT record:** intermediate reasoning, raw error messages without diagnosis, one-off lookups, info expiring < 7 days, conversation summaries (use `session_search`).

---

## Hybrid Search

```bash
# Hybrid (keyword + semantic, recommended)
python3 /root/.hermes/bin/kb_unified_search.py "ssl certificate error"

# Semantic only (meaning-based)
python3 /root/.hermes/bin/kb_unified_search.py "ssl error" --mode semantic

# Keyword only (instant, no API call)
python3 /root/.hermes/bin/kb_unified_search.py "ssl error" --mode keyword
```

Searches 4 namespaces simultaneously:
1. `kb_embeddings` — main KB (config/fixes/projects)
2. `doosi_kb_emb` — project-specific KB
3. `session_embeddings` — past session summaries
4. `skill_embeddings` — skill descriptions

Chunks at H2 headings → finds the specific section, not just the page.

---

## KB Maintenance (Lint)

Run after major project completions and weekly:

```bash
# Check for orphan pages, broken wikilinks, stale content
python3 /root/.hermes/bin/kb_stats.py

# Manual audit checklist:
# 1. Orphans: pages with no inbound [[wikilinks]]
# 2. Broken: [[links]] pointing to non-existent pages
# 3. Index: every page in kb/ appears in index.md
# 4. Stale: updated < 30 days ago but heavily referenced
# 5. Contested: pages with contradictions: field set
# 6. Oversized: pages > 200 lines → split
# 7. Tags: all tags in SCHEMA.md taxonomy
# 8. Log: if log.md > 500 entries → rotate to log-YYYY.md
```

---

## Weekly Cycle

Every Monday, create `kb/weekly/YYYY-WNN.md`:

```markdown
---
title: Week 2026-W19 Summary
created: 2026-05-11
type: weekly
---

## Completed
- [[projects/nova-oss]] v1.2.0 — nova/kb/ module shipped

## New KB Pages (N)
- [[config/hmg-gemini]] — internal Gemini endpoint details

## Active Projects
- [[projects/blog-pipeline]] — 81 posts, adding Bing ping

## Carry-over
- [[projects/krayt]] — BB draft pending review
```

---

## Multi-Agent KB Sync (Hermes HA)

Primary → Standby sync every 5 minutes:

```bash
# Sync: KB + memories + skills (newer-wins)
/root/.hermes/scripts/sync_bidirectional.sh

# NEVER sync SOUL.md between instances (identity separation)
# Each agent keeps its own SOUL.md
```

On failover: Standby reads last `[ACTIVE]` tags + project pages → resumes from exact state.
On failback: Primary pulls Standby's KB changes before reclaiming active role.

---

## Anti-Patterns

- **Skipping session-end writes** — biggest failure mode. Loss of days of context.
- **Letting MEMORY.md grow unbounded** — over 8K chars becomes noise. Audit regularly.
- **Creating pages for passing mentions** — follow page thresholds in SCHEMA.md.
- **Using freeform tags** — add to SCHEMA.md taxonomy first. Tag sprawl breaks lint.
- **Silent contradiction overwrites** — always record both claims with dates.
- **Batching KB writes** — write immediately. Crashes happen.
- **Forgetting index.md** — orphaned pages are invisible. Update on every write.
- **Treating KB as session log** — KB is for stable, reusable knowledge only.

---

## Pitfalls

- `kb_unified_search.py` requires `~/.hermes/embeddings.db` to exist (created by `kb_sync.py`)
- If embeddings.db is missing, search falls back to keyword-only automatically
- MEMORY.md newline format: entries separated by `§` delimiter — don't use `§` in content
- `log.md` entries use `## [YYYY-MM-DD] action | subject` format — keep consistent
- Index.md must stay < ~500 entries — split into sub-indexes at that point
- Weekly files: use ISO week numbering `YYYY-WNN` not calendar month
