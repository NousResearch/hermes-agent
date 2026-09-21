---
name: memory-promotion
description: |
  Weekly ritual that reviews high-confidence Mnemosyne memories and proposes
  which ones should be promoted into structured GBrain pages. Run this every
  Sunday morning, or whenever the user requests a "memory consolidation review."
trigger:
  manual: ["promote memories", "memory review", "consolidate memories"]
  schedule: "0 9 * * 0"  # Sunday 9am
adoption_status: permanent
---

# Memory Promotion Ritual

## What this skill does

Mnemosyne holds conversational facts and episodic memories; the GBrain wiki (`~/brain/`)
holds canonical structured knowledge as plain Markdown files. This skill bridges them —
periodically reviewing high-confidence Mnemosyne facts to decide which deserve promotion
into GBrain pages.

Also loads skills: gbrain-knowledge (for brain structure and filing rules)

## Architecture

| Tier | System | Storage | Access |
|------|--------|---------|--------|
| Conversational | Mnemosyne | `~/.hermes/mnemosyne/data/mnemosyne.db` | Vector + FTS5 recall via tools |
| Structured canonical | GBrain wiki | `~/brain/*.md` | Direct file read/write |
| Human-authored | Obsidian | `~/vaults/obsidian-master/` | READ ONLY — never write |

DO NOT write to `~/vaults/obsidian-master/`. That is Sahil's personal space.

## Two Modes of Operation

This skill supports TWO operating modes, selected by how it's invoked:

### Mode 1: Manual review (default when invoked interactively)
Follow the full workflow below: query → map → check → present for approval → write → tag. This is the original design — Sahil reviews candidates and approves/rejects.

### Mode 2: Cron auto-promotion (when running as a daily cron)
The `memory-promotion-daily` cron (job `6b64e27d9073`, 06:00 daily) runs autonomously. It cannot ask for approval. Mode 2 uses stricter auto-promotion criteria:

**Guardrails for auto-promotion:**
- Importance MUST be ≥ 0.8 (not just ≥ 0.7)
- Veracity MUST be "stated" or "imported" — never "modeled", "inferred", or null
- Fact MUST map cleanly to a single existing brain page — if ambiguous, file a kanban task instead
- Fact MUST NOT be time-bounded, temporary, or conversational noise
- MUST check existing brain page content before writing — skip if already covered (even loosely)
- Use targeted `patch` operations only — never overwrite entire files
- Tag promoted facts with `PROMOTED to ~/brain/<path>` via memory tool to prevent re-promotion

**For borderline cases** (high importance, no clear page match, or uncertain coverage):
File a kanban task: `hermes kanban create "Memory promotion: <topic>" --triage --assignee wesker --priority 3`

**Output format for cron mode:**
- Summary of what was promoted (which brain pages got new facts)
- Summary of what was filed as kanban tasks (if any)
- If nothing to promote: respond [SILENT]

## When NOT to promote

- Low-confidence memories (importance < 0.7) — or < 0.8 in cron mode
- Time-bounded facts ("Sahil is currently job hunting" — stale quickly)
- Conversational noise ("Sahil mentioned he's tired", tool output fragments)
- Anything already represented in a GBrain page
- Auto-generated `[USER]` or `[ASSISTANT]` conversation fragments from sync_turn

## When TO promote

- Importance >= 0.8 (high-value facts)
- Facts mentioned across multiple sessions (check recall_count in memory metadata)
- Facts about: people, projects, infrastructure, preferences, brand voice, cron jobs
- Facts that map cleanly to a GBrain page type (via RESOLVER.md)
- Stated facts with veracity="stated" or veracity="imported"

## Workflow

### Step 1: Query Mnemosyne for candidates

Use the `mnemosyne_recall` tool with a broad query to surface recent high-importance memories:

```
mnemosyne_recall(query="*", limit=50)
```

Or query specific domains:
```
mnemosyne_recall(query="project configuration infrastructure", limit=20)
```

### Step 2: Map each candidate to a GBrain page using RESOLVER.md

Read `~/brain/RESOLVER.md` to understand the folder structure. For each candidate,
determine the target page:

- If about a specific app (Plenishd, CoachOS, MatchdayMaestro, Kick-tionary) -> `~/brain/apps/portfolio.md`
- If about infrastructure (VPS, cron, tools, agents) -> `~/brain/conventions/infrastructure.md`
- If about accounts -> `~/brain/accounts/connected-accounts.md`
- If about a person -> `~/brain/people/<name>.md`
- If about a concept or convention -> `~/brain/conventions/<topic>.md`
- If about property -> `~/brain/properties/sahil-properties.md`
- If about a project -> `~/brain/projects/<name>.md`

If no clear mapping exists, file it as a new page under the appropriate category folder.

### Step 3: Check for existing coverage

Read the target GBrain page. If the candidate fact is already represented (same meaning,
even if worded differently), skip it.

### Step 4: Present batch for approval

Format as:

```
PROMOTION CANDIDATES (sourced from Mnemosyne, week of YYYY-MM-DD)

1. Fact: "Plenishd uses Whisk SDK for basket injection"
   Source: Mnemosyne (importance 0.9, veracity=stated, recalled 3 times)
   Target: ~/brain/apps/portfolio.md -> ## Plenishd section
   Current coverage: "Plenishd uses RN+Expo+Convex" (no mention of Whisk SDK)
   Action: ADD as bullet: "- Whisk SDK for basket injection"

2. Fact: "KENSEI heartbeat audit runs hourly at 0 * * * *"
   Source: Mnemosyne (importance 0.85, veracity=stated)
   Target: ~/brain/conventions/infrastructure.md -> ## Model Routing or new section
   Current coverage: existing cron entry already listed
   Action: SKIP (already covered)

[...]

Reply with: APPROVE 1,3 or REJECT 2,4 or EDIT 5 [your edit text]
```

### Step 5: After approval, write to GBrain

Use the `patch` tool to add content to the target GBrain page. Never overwrite the
entire file — only append or insert the new fact.

### Step 6: Tag the promoted Mnemosyne fact

Use `mnemosyne_remember` to tag that this fact was promoted:

```
mnemosyne_remember(
    content="PROMOTED to ~/brain/apps/portfolio.md on YYYY-MM-DD: <original fact>",
    importance=0.9,
    veracity="stated",
    scope="global",
    metadata={"promoted_to_gbrain": "path", "original_memory_id": "id"}
)
```

This prevents the same fact being re-proposed next week.

## Anti-patterns to avoid

- DO NOT write to Obsidian vault. Ever.
- DO NOT auto-promote without approval in manual review mode — cron mode has its own stricter guardrails (see Mode 2 above).
- DO NOT promote conversational scratch ("Sahil said X today").
- DO NOT delete Mnemosyne facts after promotion. Tag them instead.
- DO NOT promote facts that contradict existing GBrain content without flagging the contradiction.
