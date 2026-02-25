# Hermes Agent Memory Architecture

This document explains Hermes Agent's practical "multi-layer memory" system as implemented across the runtime, memory files, skills system, and SQLite session store.

The design goal is simple: keep the active context window small and useful while preserving durable knowledge in progressively cheaper layers.

## Overview

Hermes memory is not one feature. It is four layers with different latency, cost, and write behavior:

1. Active conversation context (fastest, per-turn, token-expensive)
2. Persistent curated memory files (`MEMORY.md`, `USER.md`) (small, durable, explicitly curated)
3. Skills as procedural memory (`SKILL.md` + `skill_manage`) (durable "how-to" workflows)
4. SQLite session store + FTS5 (`state.db`) (large searchable transcript history)

## ASCII Architecture Diagram

```text
                          Hermes Agent Memory Stack

                    (highest relevance / highest token cost)
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Active Conversation Context                                       │
│ - current messages[] in agent loop                                         │
│ - cached system prompt snapshot (stable for session)                       │
│ - tool results, current task state, recent turns                           │
│ - optional context compression when near model limit                        │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │ promotes durable facts/profiles via `memory` tool
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 2: Curated Persistent Memory Files                                   │
│ ~/.hermes/memories/MEMORY.md  (environment/project notes)                  │
│ ~/.hermes/memories/USER.md    (user preferences/profile)                   │
│ - bounded char limits (~800 / ~500 token targets)                          │
│ - injected as frozen snapshot at session start                             │
│ - mid-session writes persist to disk but do not mutate current prompt      │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │ promotes reusable procedures via `skill_manage`
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: Skills (Procedural Memory)                                        │
│ ~/.hermes/skills/**/SKILL.md + references/templates/scripts/assets         │
│ - on-demand loading (progressive disclosure)                               │
│ - created/updated by `skill_manage` after successful complex workflows      │
│ - stores "how to do X", not personal facts                                 │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │ recall path via `session_search`
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 4: SQLite Session Store (`~/.hermes/state.db`)                       │
│ - sessions + messages tables                                               │
│ - FTS5 virtual table (`messages_fts`) + triggers                           │
│ - full transcript history + metadata + system prompt snapshot              │
│ - `session_search` = FTS5 search -> group sessions -> summarize matches    │
└─────────────────────────────────────────────────────────────────────────────┘
                    (largest capacity / lowest immediate prompt impact)
```

## Layer 1: Active Conversation Context

Layer 1 is the live `messages` list used in the agent loop for LLM calls. This is where immediate reasoning happens.

### What is included

- Current user/assistant/tool messages
- Current task outputs (tool results, errors, patches, etc.)
- Cached session system prompt (built once per session)
- Optional ephemeral prompts injected at API-call time

### Key implementation behaviors

- `run_agent.py` builds the system prompt once and caches it for the session (`_cached_system_prompt`)
- The cached system prompt is reused to maximize prefix cache hits
- Context compression can summarize middle turns when token usage approaches the model limit
- On compression, Hermes may split the SQLite session and link it with `parent_session_id`

### Concrete example

User asks: "Fix the Docker compose networking issue in this repo."

Layer 1 contains:
- Current repo path and recent terminal outputs
- `git diff` results
- The latest error messages from `docker compose up`
- The assistant's current plan and tool call results

This is the most useful context for the next tool call, but it is expensive to keep forever in the prompt.

## Layer 2: `MEMORY.md` + `USER.md` (Curated Persistent Memory)

Layer 2 stores short, durable facts that should survive sessions.

### Files and intent

- `MEMORY.md`: agent notes (environment facts, project conventions, quirks, lessons learned)
- `USER.md`: user profile (preferences, style, expectations, habits)

### Token budgets vs enforced limits

Hermes documents these as approximate token budgets, but enforces **character** limits in code:

- `MEMORY.md`: `2200` chars (roughly `~800` tokens)
- `USER.md`: `1375` chars (roughly `~500` tokens)

Why char limits: model-independent, deterministic, and cheap to enforce.

### Frozen snapshot behavior (important)

At session start:
- Hermes loads both files
- Deduplicates entries
- Renders a **frozen snapshot** for system prompt injection

During the session:
- `memory` tool writes persist to disk immediately
- The active system prompt does **not** change mid-session (preserves prefix cache stability)
- New memory becomes part of the prompt on the next session (or after prompt rebuild/compression)

### Write triggers (when Hermes is expected to save)

From the `memory` tool schema + runtime nudges, Hermes is encouraged to write when:

- The user shares preferences/habits/personal details
- The agent learns environment facts (OS, installed tools, repo conventions)
- The user corrects the agent ("remember this", "don't do X")
- A project-specific quirk/workflow is discovered
- A complex task completes (brief diary-style note)

Additional runtime triggers:

- Periodic memory nudge after several exchanges (if enabled)
- Pre-compression memory flush: before context compression, Hermes prompts itself to save anything worth remembering

Capacity behavior:

- Tool guidance says consolidate entries when usage exceeds ~80%
- Add/replace operations are rejected if they exceed the configured char limit

### Concrete examples

`USER.md` example entries:

- "User prefers Turkish explanations for operational steps."
- "User wants autonomous execution without confirmation prompts when safe."

`MEMORY.md` example entries:

- "This repo uses `apply_patch` for focused edits; avoid touching `.py` logic unless requested."
- "`gh` may be missing from PATH on this machine; Homebrew install required in some sessions."

## Layer 3: Skills as Procedural Memory (`skill_manage`)

Layer 3 stores reusable procedures, not facts.

If Layer 2 is "what is true", Layer 3 is "how to do this again".

### What a skill is

A skill is a directory under `~/.hermes/skills/` containing:

- `SKILL.md` (required)
- Optional `references/`, `templates/`, `scripts/`, `assets/`

Hermes treats skills as portable procedural memory compatible with the `agentskills.io` style of `SKILL.md` frontmatter + markdown instructions.

### How skills are loaded (progressive disclosure)

Hermes avoids dumping all skills into the prompt. The normal flow is:

1. `skills_categories()` (very cheap)
2. `skills_list(category)` (metadata only)
3. `skill_view(name)` (full skill only when needed)

Separately, the system prompt includes a compact generated skill index (names + short descriptions) to help the model decide what to load.

### `skill_manage` flow (write/update procedural memory)

`skill_manage` can:

- `create`
- `patch` (preferred for targeted fixes)
- `edit` (full rewrite)
- `delete`
- `write_file`
- `remove_file`

Practical flow after a successful complex task:

1. Hermes completes a non-trivial workflow (often 5+ tool calls)
2. Runtime nudges remind it to consider saving a skill
3. Hermes writes a new `SKILL.md` with trigger conditions + exact steps
4. Future sessions can rediscover and load that workflow on demand

### Concrete example

Task solved: "Review PRs with `gh`, run tests, patch failing files, and write a summary."

Layer 3 skill might store:

- When to use: "PR review + fix request"
- Steps:
  - fetch PR branch
  - run tests
  - inspect diff
  - patch minimal changes
  - rerun targeted tests
  - summarize findings
- Pitfalls (e.g., auth, branch naming, flaky tests)

This is far more reusable than writing a short fact into `MEMORY.md`.

## Layer 4: SQLite Session Store with FTS5 Search

Layer 4 is Hermes' large-capacity recall layer for full conversation history.

### Storage model

`hermes_state.py` stores sessions and messages in `~/.hermes/state.db`:

- `sessions` table: source, model, system prompt snapshot, token counters, parent links, timestamps
- `messages` table: role/content/tool metadata/timestamps
- `messages_fts` virtual table (FTS5) for full-text search over message content

FTS is maintained via SQLite triggers on insert/update/delete.

### Why this is a separate layer

- Too large to inject directly into every prompt
- Usually only a small subset is relevant
- FTS5 makes retrieval cheap; summarization makes results token-efficient

### `session_search` recall flow

`session_search` is not raw transcript dump. It does:

1. FTS5 search over `messages_fts`
2. Filter/group results by session (resolving child sessions to parent sessions)
3. Load the matching sessions' conversations
4. Truncate around query matches (centered window)
5. Summarize each session with a cheap auxiliary model
6. Return compact summaries + metadata (`when`, `source`, `model`)

This gives Hermes "recall" without polluting Layer 1 with full logs.

### Concrete example

User asks: "How did we fix that Docker networking issue?"

Layer 4 response path:

- `session_search(query="Docker OR networking OR compose")`
- FTS5 finds older sessions mentioning the issue
- Hermes receives summarized prior sessions with file names/commands/errors
- Hermes answers from the summary, optionally validating in the current repo

## How the Layers Work Together

A typical lifecycle looks like this:

1. Solve task in Layer 1 (active context)
2. Save durable facts/preferences to Layer 2 (`memory`)
3. Save reusable procedure to Layer 3 (`skill_manage`) when warranted
4. Keep the full transcript in Layer 4 automatically (`state.db`)
5. In future sessions, recall from Layer 4 (`session_search`) and/or load Layer 3 skills

This separation keeps Hermes practical:

- Fast where it must be (Layer 1)
- Small and curated where it should be (Layer 2)
- Reusable for workflows (Layer 3)
- High-capacity and searchable for history (Layer 4)

## Source Pointers

- `run_agent.py` (system prompt caching, memory flush, compression/session split, tool intercepts)
- `tools/memory_tool.py` (Layer 2 storage, limits, frozen snapshot behavior)
- `tools/skill_manager_tool.py` (Layer 3 CRUD/procedural memory)
- `tools/session_search_tool.py` (Layer 4 recall pipeline)
- `hermes_state.py` (SQLite schema + FTS5 search)
- `agent/prompt_builder.py` (skills/context-file prompt assembly)
