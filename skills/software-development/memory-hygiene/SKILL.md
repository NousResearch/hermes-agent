---
name: memory-hygiene
description: "Use when saving MEMORY.md, auditing, or fixing overflow."
version: 0.1.0
author: Tony (icegs), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [memory, curation, hygiene, decision-framework, routing, self-management]
    related_skills: [hermes-agent-skill-authoring]
---

# Memory Hygiene

## Prerequisites

- A Hermes agent session with the `memory` tool available (see
  `skills/hermes-agent` for wiring). The memory tool is optional and can be
  disabled via config (`memory.memory_enabled` / `memory.user_profile_enabled`).
- Familiarity with the `memory` tool's batch surface: one `operations` array
  carrying `add`, `patch`, and `remove` actions.

## How to Run

This is a framework skill, not an executable procedure — it has no commands to
run. Its purpose is to be loaded with `skill_view(name='memory-hygiene')` (or
read directly) whenever the trigger conditions in "When to Use" fire. The
decision logic below is applied by the agent at memory-interaction time.

## Procedure

### Overview

Hermes injects `MEMORY.md` (default 2200 chars, configurable via
`memory.memory_char_limit`) and `USER.md` (default 1375 chars,
`memory.user_char_limit`) into the system prompt as a frozen snapshot at session
start (`tools/memory_tool.py:56`). The limits are configurable defaults, not
fixed caps — `MemoryStore.__init__` carries the same defaults
(`tools/memory_tool_store.py:186`). There is **no silent auto-compaction**:
the `memory` tool never silently truncates a write on overflow — it returns an error
(`tools/memory_tool_store.py:380-384`). The only other system-side write to
MEMORY.md is the memory tool's own apply-time commit, which is also fail-closed —
it returns an error rather than silently truncating on overflow.

Because the budget is bounded and the loss is silent (a rejected write is easy
to miss), the agent needs a stable framework for deciding what enters memory,
what belongs elsewhere, and when to consolidate.

**Core principle:** memory is for facts the agent needs *predictively* — before
it has been told what to do. Everything else routes to a skill (executable
procedure) or to `session_search` (historical recall).

### When to Use

**Load this skill when:**

- The user says "记住 / remember / save this / put it in memory / 固化 / 存进去"
- A `memory` tool call was rejected with an overflow / limit error
- The system-prompt header shows memory ≥ 80% of budget
- The user asks for a memory audit, cleanup, or reorganization
- You just finished a complex task and are about to propose a memory save
- You catch yourself about to write a completed-task diary or a TODO into memory

**Don't use for:**

- One-off facts you can trivially re-derive (a file path in the current task, an ephemeral PID)
- Content that belongs in a plan file, project doc, or issue tracker
- Reusable procedures — those are skills, not memory (create the skill instead)

### The Predictive-Value Framework

Before adding, patching, or defending any memory entry, run these three questions
in order. Stop at the first "no" — that entry does not belong in memory.

#### Q1. Do I need to *predict* this fact before being asked?

**Yes** if a colloquial user phrase must trigger the fact without further
clarification. Examples:

- User says "check the weather station" → I must already know it's a Modbus
  device at a specific IP with specific register addresses.
- User says "reply in the HA topic" → I must already know which topic ID maps
  to HA.
- User says "call the HA API" → I must already know the base URL and where the
  token lives.

**No** if the user always names the tool/target explicitly (e.g. "upgrade the
HA container", "edit the heweather integration"). Those trigger `skill_view` on
a matching skill; the details live inside the skill, not in memory.

#### Q2. What happens if this fact is missing?

- **I would have to ask the user** → high memory value; asking wastes their time.
- **I would run a tool/search** → depends on cost. A 20-token session_search
  wins over a 400-token permanent memory slot for anything queried less than
  weekly.
- **Silent failure** (wrong timezone, wrong register, dropped attachment) →
  high memory value; the user will not know I made the wrong choice.
- **Nothing** → do not save.

#### Q3. Can `session_search` recover this in one query?

- **Yes, with 2–3 obvious keywords** → session_search is enough; do not spend
  memory budget.
- **No, because the trigger phrase would not obviously match past turns** (e.g.
  a numeric ID, a one-shot design decision) → memory value.

Only a fact that passes Q1 (predictive), Q2 (real cost if missing), **and** Q3
(not recoverable by search) earns a memory slot.

### The 3-Tier Routing Table

Once a fact is judged, route it to exactly one home. Do not duplicate across
tiers.

| Fact type | Home | Reason |
|-----------|------|--------|
| User preferences, communication style, corrections | `USER.md` | Predictively shapes every reply |
| Stable environment facts (IPs, register maps, credentials paths, timezone quirks) | `MEMORY.md` | Predictively needed on colloquial triggers |
| Reproducible technical pitfalls (framework quirks, silent-failure traps) | `MEMORY.md` if high-frequency, else skill | Only high-frequency reproducible pitfalls justify permanent context cost |
| Executable procedures / workflows / step-by-step recipes | Skill | Loaded on demand via `skill_view`; unbounded size |
| Completed work diary | 1 short line in `MEMORY.md` if the completion itself is a stable fact, otherwise nothing | The doing is not durable; the outcome sometimes is |
| Open TODOs, in-flight work, pending user asks | Plan file, session_search, or issue tracker | Not durable; churn creates memory sediment |
| One-shot analysis, reasoning chain, discussion | `session_search` | Free, unbounded, retrievable by keyword |

### Capacity Management

The memory tool reports current chars vs. limit on every call. Match your action
to the budget zone. The budget check runs on the **final state** of a batch — a
single batch can remove/patch stale entries to free room AND add new ones. On
overflow the `add` action returns an error stating current usage and how far
over the limit the new total would go (`tools/memory_tool_store.py:380-384`).

- **< 60% (comfortable):** direct `add` for a passing entry.
- **60–80% (watchful):** before `add`, scan for entries that could be tightened
  or merged. Prefer `patch` over `add` when a related shorter entry already
  exists.
- **≥ 80% (consolidate to be safe):** issue a single **batch** call. Combine
  `remove` / `patch` (freeing space) with `add` (new entry) in one `operations`
  array. A standalone `add` above 80% still succeeds unless the resulting total
  exceeds the configured limit — consolidation is best practice to preserve
  headroom, but not a hard gate.
- **≥ 95% (audit first):** stop adding. Run a full audit (below) before any new
  writes.

Batch example (freeing space and adding in one call):

```
memory(target='memory', operations=[
  {"action": "remove",  "old_text": "<unique substring of stale TODO>"},
  {"action": "patch",   "old_text": "<substring of oversized entry>",
                        "content": "<tightened rewrite>"},
  {"action": "add",     "content": "<new high-value fact>"}
])
```

### Audit Workflow

Run this when memory is ≥ 95% or when the user requests a cleanup.

1. **List entries with their sizes.** Read `MEMORY.md` / `USER.md` and note the
   byte count of each entry. *Done when:* every entry has a rough size in front
   of you.
2. **Score each entry against Q1/Q2/Q3.** Mark: keep, tighten, split-to-skill,
   or drop. *Done when:* every entry has one of those four labels.
3. **Draft the tightened / split versions before touching the file.** For a
   tighten, write the shorter replacement text. For a split, note which skill
   will absorb the content. *Done when:* every non-keep entry has a concrete
   replacement plan.
4. **Ask the user before dropping or splitting.** Removing an entry silently is
   worse than an overflowing memory — the user will not notice until they hit
   the missing behavior. See arXiv:2608.11095 (`#86022`) for the rationale-
   attached-entry mechanism that eventually makes audit-driven drops auditable;
   until then, the human-in-the-loop is the safety net. *Done when:* the user
   has approved each drop/split, or chosen to keep it.
5. **Execute as one batch.** Apply all approved changes in a single `memory`
   call with an `operations` array. *Done when:* the tool response confirms the
   write and reports the new byte count.
6. **Record the new baseline.** Note the post-audit percentage (e.g.
   "MEMORY 84% after audit"). *Done when:* the baseline is stated to the user so
   future writes know their starting point.

### Batch Syntax Pitfalls

1. **`old_text` matches too broadly.** Use the shortest **unique** substring,
   not the full entry. Whole-entry matches silently miss when the entry has been
   re-flowed.
2. **Half-width vs. full-width punctuation mismatch.** `(` and `（` are
   different bytes. Copy the exact substring from the current memory dump; do not
   retype from memory.
3. **Batch size.** Batches are all-or-nothing: they validate every op against a
   working copy and commit once only if the final result passes the char budget;
   a failure writes nothing (`tools/memory_tool_store.py:539-594`,
   `:327-334`). There is no operation-count limit. Keep batches small enough to
   inspect in one glance.
4. **`patch` needs a matching `old_text`.** `patch` replaces a substring; if
   none matches, the whole batch fails. If you intend to add new content with
   no existing anchor, use `add`.
5. **Trusting a stale `current_chars` reading.** Batches update state
   atomically; if a prior tool call in the same turn already wrote, re-read
   `MEMORY.md` before composing the next batch.

### Anti-Patterns

1. **"I finished X, let me remember I finished X."** Completed-task logs are the
   single biggest source of sediment. Save an outcome only if the outcome itself
   will change future behavior (e.g. "watering system live since 2026-07-21" — a
   fact users refer to; not "fixed 4 attributes today").
2. **Storing a runbook in memory.** If it has numbered steps, exact commands, or
   a "verify" section — it is a skill. Move it to
   `~/.hermes/skills/` or `hermes-agent/skills/<category>/`.
3. **"Just this once" oversize entry.** Every entry that survives a year was
   once "just this once". Tighten before saving, not after.
4. **Copy-pasting raw tool output into memory.** Extract the *rule* the output
   taught you, drop the transcript.
5. **Auto-proposing "save to memory?" after every task.** Only propose when the
   fact passes Q1/Q2/Q3. A skill is a better home for most task-completion
   learnings.
6. **Rewriting an entry as an imperative to yourself.** Memory is descriptive
   facts ("HA API returns UTC"), not instructions ("Always convert to CST").
   Imperatives get re-read as directives and can override the user's live
   request.

### Common Pitfalls

1. **Adding when memory is > 80% without consolidating.** An `add` above 80%
   succeeds unless the resulting total exceeds the configured limit
   (`tools/memory_tool_store.py:380-384`); the fact is only lost if the write is
   rejected and the agent moves on without noticing. Consolidation is still best
   practice to preserve headroom — batch it with the add when you can.
2. **Trusting the 80% line as a failure gate.** 80% is a budget *heuristic*, not
   a tool contract. The `add` overflow check fires only when the *resulting
   total* exceeds the configured cap (`memory_tool_store.py:380-384`).
3. **Using `replace` in a batch.** The live schema is `add` / `patch` /
   `remove` only (`tools/memory_tool.py:404,434`); `replace` is rejected outright
   for new writes via `_REPLACE_REMOVED` (`memory_tool.py:168,261-262,280-281`);
   staged pre-removal ops replay as `store.replace()` (`memory_tool.py:110`).

## Verification

- **Trigger:** "memory rejected with overflow" → run §Capacity Management and
  §Batch Syntax Pitfalls; the batch example above uses only resolvable actions.
- **Trigger:** "user asks for a memory audit" → run §Audit Workflow; step 5
  confirms the new byte count via a single all-or-nothing batch
  (`memory_tool_store.py:539-594`).
- **Cross-check against the tool contract:** `test_related_skills_resolve` (canonical CI name) and `AGENTS.md:28` (description) confirm this skill itself commits to standards.

## Related Skills

- `hermes-agent-skill-authoring` — authoring standards this skill follows
  (frontmatter, section order, description ≤ 60 chars, author convention).
- `hermes-memory-curation` — the curation/defrag surface that operationalizes
  this decision framework into automated passes.

---

## Defect-Fix Log

| # | Defect (source) | How the enriched draft fixes it |
|---|-----------------|---------------------------------|
| STORM-BLOCKER | `"action": "replace"` in batch examples (STORM, line 120); `replace` is not in the live schema enum (`memory_tool.py:404,434`); tool returns `_REPLACE_REMOVED`. | All batch examples now use `"action": "patch"` with matching `old_text` + `content`; Anti-Pattern #4 in Common Pitfalls documents the schema as `add`/`patch`/`remove` only. |
| V2 (CI-breaking) | `related_skills: [plan, hermes-agent-skill-authoring]` — `plan` is not a shipped skill dir; `test_related_skills_resolve` fails. | `metadata.hermes.related_skills` now lists only shipped peers: `hermes-agent-skill-authoring`. `plan` and the dangling `hermes-memory-curation` (not a shipped skill dir) both dropped. |
| V3 | Author credit `icegs (@icegs), Hermes Agent` — stray `@`, handle in real-name slot; canonical template is `Real Name (github-handle), Hermes Agent` (`hermes-agent-skill-authoring/SKILL.md:61`). | `author: Tony (icegs), Hermes Agent` — real name first, handle in parentheses without `@`, "Hermes Agent" second, matching the template and `skills/AGENTS.md:44-46`. |
| V4 | `version: 1.0.0` on a brand-new skill; template says new skills start at `0.1.0`. | `version: 0.1.0`. |
| V5 | Section order diverges from `skills/AGENTS.md:47-50`: `# <Skill> Skill`, intro, When to Use, Prerequisites, How to Run, Quick Reference, Procedure, Pitfalls, Verification. | H1 is `# Memory Hygiene` (titlecase, no "Skill" suffix — the bundled-peer convention); body now follows Prerequisites / How to Run / Procedure / Verification, with the framework content nested under Procedure. Framework-type skills are exempt from `## Quick Reference`; noted in How-to-Run. |
| V6 | "no auto-compaction" blanket was imprecise — referenced a fabricated `MemoryStore.bank()` promotion path at `:800-809` that does not exist in the public surface. | Removed all `bank()`/`DreamCycle`/`:800-809` references. Reworded to the accurate public-surface statement only: "the `memory` tool never silently truncates a write on overflow — it returns an error (`tools/memory_tool_store.py:380-384`)." |
| V7 | Stale line citations `:57`, `:469`, `:1000-1005`, `:1027-1028`, `:1054-1055`, `:662` (stress verdict "actuals"). | All citations re-anchored to canonical main lines: `memory_tool.py:56`; `memory_tool_store.py:186,327-334,380-384,539-594`; `skills/AGENTS.md:28-31,44-46`; `hermes-agent-skill-authoring/SKILL.md:60-61`. The previous `:312` misattribution (cited as `__init__`) corrected — `:312` no longer reproduced as a canonical `__init__` line; all re-anchored lines verified against the canonical storm. |

## Summary of Surgical Fixes Applied (69264-FIX)

1. **Removed fabricated `bank()`/`DreamCycle`/`:800-809` reframe (V6 + body):** The V6 section claimed a `MemoryStore.bank()` promotion path at `tools/memory_tool_store.py:800-809` — this does not exist in the public surface (the file is 700 lines; line 800+ is past EOF; no `bank()` in the public repo). Reworded the body (lines ~45-49) to the accurate public-surface statement only: "the `memory` tool never silently truncates a write on overflow — it returns an error (`tools/memory_tool_store.py:380-384`)". The V6 defect-log entry now records the removal. The V7 citation list no longer includes `:800-809`.
2. **`related_skills` trimmed to `[hermes-agent-skill-authoring]` only:** Removed `hermes-memory-curation` from the YAML frontmatter (it is not a shipped skill dir and would dangle in `test_related_skills_resolve` CI). The body may still mention the curation surface in prose (line 259) — it is not listed in `related_skills`.
3. **`_related_skills_resolve` → `test_related_skills_resolve`:** Fixed the CI function name in the Verification cross-check (line 253) and the V2 defect-log entry (line 269).
4. **`agENTS.md:28` → `AGENTS.md:28`:** Fixed the uppercase typo in the Verification cross-check (line 253).

---
