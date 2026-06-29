---
name: 'workflow-visual'
description: 'Create visual representations of page designs using external tools and integrate results back into specifications.'
---

# [W] Visual Design — Work with Visual Tools

**Goal:** Create visual representations of page designs using external tools and integrate results back into specifications.

**When to use:** When the user wants to create or review visual mockups, prototypes, or design artifacts using tools like Figma, Nano Banana, Stitch, or Pencil.io.

---

## INITIALIZATION

Read design log at `{output_folder}/_progress/00-design-log.md` before starting.

## Entry

Load page specification from `{output_folder}/C-UX-Scenarios/`.

## Steps

Execute steps in `./steps-w/`:

| Step | File | Purpose |
|------|------|---------|
| 01 | step-01-visual-approach.md | Choose visual tool and approach |
| 02 | step-02-generate-visual.md | Create visual representation |
| 03 | step-03-review-integrate.md | Review result and integrate into spec |

**Supported tools:**
- **Nano Banana** — AI image generation for mockups
- **Figma** — Professional design tool integration
- **Stitch** — Component-based design
- **Pencil.io** — Quick wireframing
- **HTML prototype** — Code-based visual design

**Reference data:**
- `./data/guides/HTML-VS-VISUAL-STYLES.md` — choosing between approaches
- `./data/guides/NANO-BANANA-PROMPT-GUIDE.md` — prompt composition for AI image generation

---

## AFTER COMPLETION

1. Append a progress entry to `{output_folder}/_progress/00-design-log.md` under `## Progress`:
   `### [date] — Visual Design: [what was generated]`
2. Suggest next action based on the adaptive dashboard (read Design Loop Status to find what needs attention next)
