---
name: idea-to-ui-design-brief
description: "Use when turning a product/design doc into a focused UI design brief, optional AI image-generation concept prompts, and implementation-ready UI direction without replacing the main idea workflow."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [idea-workflow, ui-design, ux, image-generation, product-design, superpowers]
    related_skills: [idea-superpowers-suite, idea-to-design-doc, idea-to-implementation-doc, claude-design]
---
# Idea to UI Design Brief

## Overview

Use this skill as an optional Full-mode stage inside the idea workflow. It converts a product/design direction into a practical UI design brief that can guide image-generation concepts, HTML prototypes, or later Superpowers implementation work.

This skill should not run in Lite mode unless the user explicitly asks for UI design. Lite mode should keep the previous simple idea-capture behavior.

The UI brief is a bridge between product thinking and implementation. It should define what the interface should look, feel, and behave like without forcing premature code architecture.

## When to Use

Use when the user asks to:

- explore possible UI designs for an idea;
- add a dedicated UI/design stage to the idea workflow;
- generate image prompts for app screenshots, dashboards, landing pages, or product surfaces;
- create a UI brief before giving the idea to a coding agent;
- critique or redesign an already-built app after screenshots exist;
- turn a design doc into screen-by-screen UI requirements.

Do not use when:

- the user only wants a quick Lite idea note;
- the idea is not yet clear enough to identify target users and core behavior;
- the user wants production frontend code immediately — use implementation skills after the UI direction is chosen;
- the user wants a formal token spec only — use `design-md` if available.

## Artifact Placement

For Full mode idea packages, add this artifact after the design doc and before implementation planning:

```text
ideas/<idea-slug>/
  README.md
  00-idea-capture.md
  01-design-doc.md
  02-ui-design-brief.md
  03-implementation-spec.md
  04-agent-build-handoff.md
  05-spec-review.md
```

Fallback rule: if the user skips this stage, preserve the old workflow numbering and behavior, or mark UI decisions as assumptions in the implementation spec. Do not block the workflow merely because no UI brief exists.

## Required Inputs

Before writing the UI brief, collect or infer:

- product purpose and target user;
- core user flow;
- main screens or surfaces;
- platform target: web, desktop, mobile web, native mobile, or multi-surface;
- desired feel: calm, command-center, playful, professional, dense, minimal, etc.;
- visual constraints: brand, existing repo/design system, accessibility needs, dark/light mode;
- important states: empty, loading, error, success, first-run, repeat-use.

If these are missing and the user has not forced progression, ask one concise question at a time.

## UI Brief Structure

Create `02-ui-design-brief.md` with:

```markdown
# <Idea Title> — UI Design Brief

## Purpose

## Product Feel

## Design Principles

## Primary Surfaces / Screens

## Screen-by-Screen Notes

## Layout / Information Architecture

## Component Inventory

## Key User Flows

## States to Design

## Visual Direction

## Accessibility and Responsiveness

## Content / Copy Notes

## Optional Image-Generation Concepts

## Selected Direction

## Open Questions

## Handoff Notes for Implementation
```

## Optional Image-Generation Pass

Image generation is optional and should be treated as concept exploration, not the implementation source of truth.

Use image generation when:

- the user wants possible visual directions;
- taste is ambiguous;
- an app/dashboard/landing page needs visual exploration before code;
- the workflow benefits from comparing 2-3 directions.

Default to three concept prompts:

1. **Conservative** — familiar, low-risk, easy to implement.
2. **Strong-fit** — best interpretation of the product brief.
3. **Divergent** — more distinctive, useful for discovering taste boundaries.

Each prompt should specify:

- product category and target user;
- screen type and platform;
- information hierarchy;
- density and interaction posture;
- visual style and theme;
- what should not appear;
- that the result is a UI concept screenshot, not marketing art.

## Image Prompt Template

```text
Create a high-fidelity UI concept screenshot for <product>. Platform: <desktop web/mobile/etc>. Screen: <main dashboard/onboarding/settings/etc>. Target user: <user>. Product feel: <feel>. Layout: <key layout>. Must show: <components/data/actions>. Avoid: generic SaaS filler, fake irrelevant metrics, stock photos, excessive gradients, unreadable tiny text. Style: <visual direction>. The image should be a realistic product UI mockup suitable for turning into a design brief.
```

## After Images Are Generated

If image artifacts are generated:

1. Save or reference each image path/URL in the UI brief.
2. Use vision analysis or manual review to extract what works and what does not.
3. Update `Selected Direction` with the chosen layout, components, and visual rules.
4. Do not ask the build agent to copy the image blindly. Convert the image into text requirements, components, tokens, and acceptance criteria.

## Post-Build Redesign Loop

After Superpowers or another coding agent builds the app, this skill can run again as a UI polish loop:

1. Capture screenshots of the built app.
2. Compare screenshots to the UI brief and product design doc.
3. Identify UI gaps: hierarchy, spacing, navigation, copy, state handling, responsiveness.
4. Optionally generate alternate UI concepts.
5. Produce a focused redesign brief and implementation task list.
6. Hand the task list to Superpowers with the constraint: improve UI only, avoid backend behavior changes unless explicitly approved.

## Integration Rules

- This stage is Full-mode only by default.
- It must not make the workflow mandatory or brittle.
- If skipped, continue with the old design-doc → implementation-spec → handoff sequence.
- If the user wants speed, fold only a short `UI / UX Notes` section into the design doc instead.
- If image generation is unavailable, write image prompts and continue with a text-only UI brief.
- If an existing repo exists, inspect current UI/components before inventing a new design direction.
- If retrofitting an existing idea package that already has a Superpowers handoff, use the handoff as the compressed source, add the UI brief as an addendum, and preserve existing filenames unless the user asks for a clean regeneration. See `references/retrofitting-ui-briefs.md`.
- For cost-aware runs, cheaper workers may produce UI direction variants, state inventories, image prompts, or critique passes, but the orchestrator must choose/synthesize the final brief. Follow `idea-superpowers-suite` Cost-Aware Worker Delegation rules when available.

## Common Pitfalls

1. **Letting images become the spec.** Images are inspiration. The written UI brief and implementation handoff are the source of truth.
2. **Overcomplicating Lite mode.** Do not add this stage to quick idea capture unless requested.
3. **Generic dashboard slop.** Avoid fake metrics, irrelevant charts, and decorative cards unless the product actually needs them.
4. **Skipping states.** Empty, loading, error, success, first-run, and mobile states matter for build quality.
5. **Breaking artifact numbering.** If adding this stage to a package, update README artifact maps and subsequent stage numbers consistently.
6. **Copying proprietary references.** Transform style principles; do not clone distinctive third-party UI.

## Verification Checklist

- [ ] The workflow is Full mode or the user explicitly requested UI design.
- [ ] The product/design direction is clear enough for screen-level UI decisions.
- [ ] `02-ui-design-brief.md` exists if this stage was run.
- [ ] Optional image prompts are stored in the brief even if image generation is skipped.
- [ ] Generated images, if any, are treated as references and translated into text requirements.
- [ ] Implementation handoff references the selected UI direction or explicitly states that UI is an assumption.
- [ ] If the stage is skipped, the old workflow still proceeds without blockage.
