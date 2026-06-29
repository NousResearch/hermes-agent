---
name: 'workflow-discuss'
description: 'Creative dialog for page design — discuss what each page needs, then visualize and specify.'
---

# [C] Discuss — Creative Dialog for Page Design

**Goal:** Lead a focused creative dialog for each page — what does it need, can we simplify it, then visualize and specify.

**When to use:** The default design activity. Start here for any page that needs design thinking before building.

---

## INITIALIZATION

Read design log at `{output_folder}/_progress/00-design-log.md` before starting.

## Entry

Load scenario context (Trigger Map, scenario overview) from `{output_folder}/C-UX-Scenarios/`.

## Steps

Execute steps in `./steps-c/`:

| Step | File | Purpose |
|------|------|---------|
| 01 | step-01-exploration.md | Open-ended design exploration |

**Reference data:**
- `./data/guides/DESIGN-LOOP-GUIDE.md` — the 8-step design loop (discuss → wireframe → iterate → spec sync → implement → refine)
- `./data/scenario-init/` — scenario initialization guides
- `./data/page-creation-flows/` — page creation flow options

---

## AFTER COMPLETION

Step 01's two-option transitions handle all navigation. The design log is updated at every transition within the step itself. There is no separate "after completion" — the step loops through pages until the user stops or all pages are designed.
