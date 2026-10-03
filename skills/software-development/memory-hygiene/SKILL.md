---
name: memory-hygiene
description: "Use when about to save/audit MEMORY.md or USER.md, when memory is above 80% capacity, when a memory write is rejected for overflow, or when deciding whether a fact belongs in memory, in a skill, or in session_search. Applies the predictive-value framework and 3-tier routing (memory / skill / session_search)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [memory, curation, hygiene, decision-framework, routing, self-management]
    related_skills: [plan, hermes-agent-skill-authoring]
---

# Memory Hygiene

## Overview

Hermes injects `MEMORY.md` (2200 chars) and `USER.md` (1375 chars) into the system prompt as a frozen snapshot at session start. The write budget is small on purpose — every byte is paid on every turn. There is **no auto-compaction**: when a write exceeds the limit the memory tool returns an error instead of silently dropping entries.

Because the budget is fixed and the loss is silent (a rejected write is easy to miss), the agent needs a stable framework for deciding **what enters memory**, **what belongs elsewhere**, and **when to consolidate**. Without a framework, memory fills with completed-task diaries and stale TODOs, and the truly high-value facts get evicted or never make it in.

**Core principle:** memory is for facts the agent needs *predictively* — before it has been told what to do. Everything else routes to a skill (executable procedure) or to `session_search` (historical recall).

## When to Use

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

## The Predictive-Value Framework

Before adding, replacing, or defending any memory entry, run these three questions in order. Stop at the first "no" — that entry does not belong in memory.

### Q1. Do I need to *predict* this fact before being asked?

**Yes** if a colloquial user phrase must trigger the fact without further clarification. Examples:
- User says "check the weather station" → I must already know it's a Modbus device at a specific IP with specific register addresses.
- User says "reply in the HA topic" → I must already know which topic ID maps to HA.
- User says "call the HA API" → I must already know the base URL and where the token lives.

**No** if the user always names the tool/target explicitly (e.g. "upgrade the HA container", "edit the heweather integration"). Those trigger `skill_view` on a matching skill; the details live inside the skill, not in memory.

### Q2. What happens if this fact is missing?

- **I would have to ask the user** → high memory value; asking wastes their time.
- **I would run a tool/search** → depends on cost. A 20-token session_search wins over a 400-token permanent memory slot for anything queried less than weekly.
- **Silent failure** (wrong timezone, wrong register, dropped attachment) → high memory value; the user will not know I made the wrong choice.
- **Nothing** → do not save.

### Q3. Can `session_search` recover this in one query?

- **Yes, with 2–3 obvious keywords** → session_search is enough; do not spend memory budget.
- **No, because the trigger phrase would not obviously match past turns** (e.g. a numeric ID, a one-shot design decision) → memory value.

Only a fact that passes Q1 (predictive), Q2 (real cost if missing), **and** Q3 (not recoverable by search) earns a memory slot.

## The 3-Tier Routing Table

Once a fact is judged, route it to exactly one home. Do not duplicate across tiers.

| Fact type | Home | Reason |
|-----------|------|--------|
| User preferences, communication style, corrections | `USER.md` | Predictively shapes every reply |
| Stable environment facts (IPs, register maps, credentials paths, timezone quirks) | `MEMORY.md` | Predictively needed on colloquial triggers |
| Reproducible technical pitfalls (framework quirks, silent-failure traps) | `MEMORY.md` if high-frequency, else skill | Only high-frequency reproducible pitfalls justify permanent context cost |
| Executable procedures / workflows / step-by-step recipes | Skill | Loaded on demand via `skill_view`; unbounded size |
| Completed work diary | 1 short line in `MEMORY.md` if the completion itself is a stable fact, otherwise nothing | The doing is not durable; the outcome sometimes is |
| Open TODOs, in-flight work, pending user asks | Plan file, session_search, or issue tracker | Not durable; churn creates memory sediment |
| One-shot analysis, reasoning chain, discussion | `session_search` | Free, unbounded, retrievable by keyword |

## Capacity Management

The memory tool reports current chars vs. limit on every call. Match your action to the budget zone:

- **< 60% (comfortable):** direct `add` for a passing entry.
- **60–80% (watchful):** before `add`, scan for entries that could be tightened, merged, or split into a skill. Prefer `replace` over `add` when a related shorter entry already exists.
- **≥ 80% (consolidate before adding):** issue a single **batch** call. Combine `remove` / `replace` (freeing space) with `add` (new entry) in one `operations` array. The overflow check runs on the final state, so a batch that removes enough stale entries can admit a new one that would fail as a standalone `add`.
- **≥ 95% (audit first):** stop adding. Run a full audit (§ Audit Workflow) before any new writes.

Batch example (freeing space and adding in one call):

```
memory(target='memory', operations=[
  {"action": "remove",  "old_text": "<unique substring of stale TODO>"},
  {"action": "replace", "old_text": "<substring of oversized entry>",
                        "content": "<tightened rewrite>"},
  {"action": "add",     "content": "<new high-value fact>"}
])
```

## Audit Workflow

Run this when memory is ≥ 95% or when the user requests a cleanup.

1. **List entries with their sizes.** Read `MEMORY.md` / `USER.md` and note the byte count of each entry. *Done when:* every entry has a rough size in front of you.
2. **Score each entry against Q1/Q2/Q3.** Mark: keep, tighten, split-to-skill, or drop. *Done when:* every entry has one of those four labels.
3. **Draft the tightened / split versions before touching the file.** For a tighten, write the shorter replacement text. For a split, note which skill will absorb the content. *Done when:* every non-keep entry has a concrete replacement plan.
4. **Ask the user before dropping or splitting.** Removing an entry silently is worse than an overflowing memory — the user will not notice until they hit the missing behavior. *Done when:* the user has approved each drop/split, or chosen to keep it.
5. **Execute as one batch.** Apply all approved changes in a single `memory` call with an `operations` array. *Done when:* the tool response confirms the write and reports the new byte count.
6. **Record the new baseline.** Note the post-audit percentage (e.g. "MEMORY 84% after audit"). *Done when:* the baseline is stated to the user so future writes know their starting point.

## Batch Syntax Pitfalls

1. **`old_text` matches too broadly.** Use the shortest **unique** substring, not the full entry. Whole-entry matches silently miss when the entry has been re-flowed.
2. **Half-width vs. full-width punctuation mismatch.** `(` and `（` are different bytes. Copy the exact substring from the current memory dump; do not retype from memory.
3. **More than ~6 ops in one batch.** Batches occasionally fail mid-apply and roll back; keep changes small enough to inspect in one glance.
4. **`replace` used where `add` is meant.** `replace` needs a matching `old_text`; if none matches, the whole batch fails.
5. **Trusting a stale `current_chars` reading.** Batches update state atomically; if a prior tool call in the same turn already wrote, re-read `MEMORY.md` before composing the next batch.

## Anti-Patterns

1. **"I finished X, let me remember I finished X."** Completed-task logs are the single biggest source of sediment. Save an outcome only if the outcome itself will change future behavior (e.g. "watering system live since 2026-07-21" — a fact users refer to; not "fixed 4 attributes today").
2. **Storing a runbook in memory.** If it has numbered steps, exact commands, or a "verify" section — it is a skill. Move it to `~/.hermes/skills/` or `hermes-agent/skills/<category>/`.
3. **"Just this once" oversize entry.** Every entry that survives a year was once "just this once". Tighten before saving, not after.
4. **Copy-pasting raw tool output into memory.** Extract the *rule* the output taught you, drop the transcript.
5. **Auto-proposing "save to memory?" after every task.** Only propose when the fact passes Q1/Q2/Q3. A skill is a better home for most task-completion learnings.
6. **Rewriting an entry as an imperative to yourself.** Memory is descriptive facts ("HA API returns UTC"), not instructions ("Always convert to CST"). Imperatives get re-read as directives and can override the user's live request.

## Common Pitfalls

1. **Adding when memory is > 80% without consolidating.** The next `add` will fail; the fact will be lost silently if the agent moves on without noticing. Always batch consolidation with the add.
2. **Cascading `remove` failures.** If `old_text` does not match, `remove` fails silently in some client versions. Verify by reading `MEMORY.md` back after the batch.
3. **Splitting a fact between memory and skill.** A predictive fact (e.g. a JWT host + kid) split across the two homes means neither is complete. Keep the credential/identifier tuple in memory; keep the *patch procedure* in the skill.
4. **Growing `USER.md` for user preferences that only apply to one project.** Project-scoped preferences belong in the project's plan/config, not in the global user profile.
5. **Treating memory as append-only.** Memory is a garden. If an entry is stale, remove or rewrite it — do not layer contradicting entries.
6. **Forgetting that memory is a frozen snapshot per session.** A write in this session takes effect starting next session. Do not expect the current turn to see the new entry.

## Verification Checklist

Before finalizing any memory write:

- [ ] Entry passes Q1 (predictive), Q2 (real cost if missing), Q3 (not session_search-recoverable)
- [ ] Entry is descriptive, not imperative
- [ ] Entry is in the shortest form that preserves meaning
- [ ] No duplication with an existing entry (searched `MEMORY.md` and `USER.md` for overlap)
- [ ] If capacity ≥ 80%, the write is part of a batch that also frees space
- [ ] Post-write byte count reported to the user so future writes know the budget
- [ ] Nothing that belongs in a skill has leaked into memory
- [ ] Nothing that is only a completed-task log has leaked into memory
- [ ] No TODO or in-flight state written to memory

## One-Shot Recipes

### Recipe A — User says "记住 X"

1. Score X against Q1/Q2/Q3.
2. If all three pass → route via the 3-tier table; if memory is the home, check capacity zone and write (batched if ≥ 80%).
3. If any question fails → tell the user *why* it does not belong in memory and propose the correct home (skill or session_search).

### Recipe B — Memory tool returned "over limit"

1. Read current `MEMORY.md`.
2. Identify the 1–2 largest entries that would still pass Q1/Q2/Q3 in a tightened form.
3. Draft tightened text.
4. Re-issue the write as one batch: `replace` (tighten) + `add` (new).

### Recipe C — User asks for a memory audit

1. Follow the **Audit Workflow** section end to end.
2. Present the score table to the user before executing.
3. Only apply the batch after user approval on drops/splits.
