---
name: design-critic
description: Rank screen critiques by observable user impact.
version: 0.1.0
author: Mark (unsupportedpastels), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [design, accessibility, critique, screenshots]
    category: bots
    related_skills: [dogfood, popular-web-designs]
---

# Design Critic Workflow Skill

Turn one screenshot, URL capture, or text-described flow into a ranked critique. Visual inspection uses `vision_analyze` only when configured; otherwise the text-described fallback is useful but explicitly limited.

## When to Use

- The user provides a screen, short flow, or clear text description.
- The user wants ranked hierarchy, copy, interaction, or accessibility findings.
- Do not use to invent hidden states, exact measurements, or implementation details.

## Prerequisites

The first useful task can use a pasted textual description and user goal. For images, use `vision_analyze` only if configured. For a public URL, `browser_navigate` can inspect observable states if available. Installed skills do not guarantee either capability.

## How to Run

1. Copy `templates/critique.md` to a collision-safe path in the owning profile workspace.
2. Try the synthetic text description in `references/sample-input.md` or inspect supplied evidence.
3. Apply `references/expected-output-rubric.md` before delivery.

## Quick Reference

| Severity | Meaning |
|---|---|
| Blocker | Prevents or seriously misleads the primary task |
| Worth fixing | Material friction, comprehension, or accessibility cost |
| Minor | Polish with limited task impact |

A periodic design review can be recommended for a stable release cadence. Do not automatically create a scheduled job; explicit activation must approve targets, cadence, evidence capture, and delivery.

## Procedure

### 1. Establish task and evidence mode

Record intended user, primary task, device/context, and whether evidence is image, live page, or description. In text-described fallback mode, state that visual hierarchy, contrast, spacing, and geometry cannot be directly verified. **Done when** critique scope and limitations are explicit.

### 2. Observe without diagnosing code

List visible elements, hierarchy, labels, states, and task path. Do not invent color values, dimensions, hover/focus states, or technical causes. **Done when** observations are separated from interpretations.

### 3. Evaluate consistent criteria

Check hierarchy, spacing, typography, color cues, copy, interaction states, accessibility, and error recovery only where observable. **Done when** each applicable criterion has evidence or a not-observable marker.

### 4. Rank concrete findings

For every finding, name the affected element, severity, user cost, evidence, and a bounded fix. Avoid redesigning the whole product. **Done when** the highest-impact issues appear first and no duplicate findings remain.

### 5. Preserve and compare

Write to a new copy of `templates/critique.md`. For revisions, check old findings before adding new ones and keep prior evidence intact. **Done when** `references/expected-output-rubric.md` passes and unresolved limitations remain visible.

## Pitfalls

- Text descriptions cannot validate visual contrast, spacing, or touch target size.
- A screenshot cannot reveal keyboard flow, animation, or hidden error states.
- Do not infer implementation technology from appearance.
- Do not file tickets, edit designs, or schedule reviews without approval.

## Verification

- [ ] Every finding names evidence, user cost, severity, and fix.
- [ ] Unobservable criteria are marked rather than guessed.
- [ ] Revision checks address prior findings first.
- [ ] Source images/descriptions are preserved unchanged.
