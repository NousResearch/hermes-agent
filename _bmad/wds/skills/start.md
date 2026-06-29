# /start — Session Resume Skill

**Invocation:** `/start` (also called automatically from agent activation files)
**Works for:** any agent (saga, freya, mimir)

---

## Purpose

Loads project state and session context. Always reads the project index first — this gives the agent a complete picture of what exists before doing anything else.

---

## Behavior When Invoked

### 1. Load Project Index

**Always read `progress/project-index.md` first**, regardless of whether a session state exists.

If found: parse Phase Status and Artifacts sections. Hold this as project context — it informs everything below.
If not found: proceed silently. The index will be built on first wrap.

### 2. Detect Session State

Read `~/.claude/wds/tools/memory/SKILL.md` and follow the `load` operation for the current agent_id.

**Fallback chain:** state found → show resume prompt → fresh start

### 3. If State Found

Parse the state file for:
- Context section
- Next section — extract MODEL prefix if present
- Plan / Milestones section

**Display:**

```
⏸ Previous session found ([date from Wrapped field])

Project:  [N artifacts — current phase from project index, or "no index yet"]
Left off: [content from Context section]
Next:     [Next — strip MODEL prefix, show as plain task]
Model:    [Sonnet | Opus — from MODEL prefix, or inferred]

[If milestones present:]
── Session Plan ──────────────────────────────
[DONE] Milestone 1 — description
[CURRENT] Milestone 2 — description (~N sessions)
[ ] Milestone 3 — description (~N sessions)
──────────────────────────────────────────────

Resume where we left off, or start fresh?
```

Wait for the user's response.

**Model inference (if no MODEL prefix in Next):**
- Any code, build, deploy, implement → Opus
- High-stakes work (production, financial, compliance) → Opus
- Long or complex multi-step tasks → Opus
- Moderate complexity: strategy, spec, dialog, UX, config, analysis → Sonnet
- Simple, low-stakes, short → Haiku
- Default to lightest model that fits.

**If resume:**
- Read the full state file
- Jump straight to the Next Action — no scanning, no re-introduction
- Treat context as already established

**If fresh:**
- Proceed with the normal activation sequence
- Do not delete the state file

### 4. If No State Found

Proceed with the normal activation sequence.

If the user describes a multi-session task at the start of a fresh session, offer to map milestones:

```
This looks like multi-session work. Want me to map it into milestones first?
(Adds ~2 min upfront, saves context thrashing later.)
```

If yes: produce milestone plan before starting work.
If no: proceed directly.

If work appears single-session: proceed directly without asking.

Do not mention /start or the absence of a state file.

---

## Notes

- Always read `progress/project-index.md` — never skip it. It is the project's memory.
- The state file lives at `progress/[agent].md` relative to the project root.
- On resume, get back to work quickly. The user knows the context.
