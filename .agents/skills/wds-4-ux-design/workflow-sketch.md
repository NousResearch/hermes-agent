---
name: 'workflow-sketch'
description: 'Analyze user-provided sketches (photos, screenshots, wireframes) and translate them into structured page specifications.'
---

# [K] Share Sketches — Interpret User Sketches

**Goal:** Analyze user-provided sketches (photos, screenshots, wireframes) and translate them into structured page specifications.

**When to use:** When the user has sketched something on paper, in a tool, or wants to share visual references for the agent to interpret.

---

## INITIALIZATION

Read design log at `{output_folder}/_progress/00-design-log.md` before starting.

## Entry

User provides sketch (image file, photo, or description of sketch).

## Steps

Execute steps in `./steps-k/`:

| Step | File | Purpose |
|------|------|---------|
| 01 | step-01-sketch-analysis.md | Analyze and interpret the sketch |

**Reference data:**
- `./data/guides/SKETCH-TEXT-ANALYSIS-GUIDE.md` — sketch analysis methodology
- `./data/guides/SKETCH-TEXT-QUICK-REFERENCE.md` — quick reference
- `./data/object-types/` — component identification

---

## AFTER COMPLETION

After sketch analysis, the page returns to step-01-exploration.md's flow. The sketch analysis feeds into the wireframe/spec sync step — the calling step handles design log updates and transition options.
