# 2026-05-28: Android Auto Kanban Bypass Incident

## Sequence

1. Sahil told Misa-Misa via voice: build an Android Auto app for KENSEI dashboard
2. Misa captured → wrote structured intake `intake-2026-05-28-dashboard-voice-mode.md` → pinged KENSEI
3. KENSEI received the intake
4. **Bypass:** Instead of creating a kanban task, wrote `PLAN-ANDROID-AUTO.md` directly to the dashboard repo filesystem
5. No task ever created → no dispatch → work sat dead from 14:33

## Root Cause

The SOUL.md protocol said "Decision — create task or push back" — a **choice** with an implicit third option ("do work directly"). The "or direct message" escape hatch in `kensei-intake` skill reinforced the bypass.

## Fix Applied

### SOUL.md (`~/.hermes/SOUL.md`)
- Replaced "Decision — create task or push back" with a **HARD GATE**
- Mandatory kanban task in same turn, **before any output**
- Vague requests still get a task ID (marked "needs clarification") — no more "ask first, create later"
- Explicit anti-bypass rule: "If reaching for write_file before kanban_create, stop"

### `kensei-intake` skill (SKILL.md)
- Removed "or direct message" from step 4
- Added Kanban Routing Rule section with hard gate
- Restored three valid boundaries (don't make architectural decisions, don't execute, don't load full system context) that were dropped during patch
- Added Misa-Misa section referencing the SOUL.md protocol

### `kanban-orchestrator` skill (SKILL.md)
- Replaced the stale "triage is a dead end — no automatic promotion" section (from 12/05/26) with the current reality: `auto_decompose: true` means the dispatcher handles triage → decomposition → ready promotion
- The triage processor now classifies+notifies only, not promotes

### Kanban task created
- `t_ff4e0575` on research board (triage) — Android Auto companion app
- Plan committed to `github.com/Sahil-SS9/kensei-dashboard` on `master`

## Prevention Rule Going Forward

Every Misa-Misa handoff produces a kanban task ID **in the same message as the acknowledgement**. No plans, no specs, no write_file before `kanban_create --triage`. If the current session cannot make tool calls, the handoff is blocked with "deferred — no tool access" so Sahil sees it needs action.

## Meta

This was a **process failure**, not a tool/environment failure. The bypass existed because the protocol had a soft choice ("do X or do Y") instead of a hard sequence ("do X, then Y, nothing else").
