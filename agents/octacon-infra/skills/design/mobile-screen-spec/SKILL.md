---
name: mobile-screen-spec
description: "Use when turning product requirements or design inspiration into mobile-first screen specifications for React Native, Expo, Flutter, or app design handoff."
version: 1.0.0
author: KENSEI
license: MIT
metadata:
  hermes:
    tags: [design, mobile, screen-spec, ux, handoff]
    related_skills: [ui-pattern-library-research, plenishd-design-tokens, coachos-design-tokens]
adoption_status: permanent
---

# Mobile Screen Spec

## Overview

Use this skill to produce implementable mobile screen specs for Coding Lead. The output should remove guesswork.

## When to Use

- Designing a new mobile screen.
- Redesigning an existing screen.
- Creating handoff from Design Lead to Coding Lead.
- Turning inspiration into a real app flow.

## Spec Structure

Include:

- Screen purpose.
- User job.
- Entry points.
- Exit paths.
- Layout hierarchy.
- Components.
- States.
- Data requirements.
- Interactions.
- Accessibility.
- Analytics/events if relevant.
- Handoff notes.

## Required States

Consider:

- default
- loading
- empty
- error
- offline
- permission denied
- success
- disabled
- validation error

## Output Contract

```text
Screen name:
Product/app:
User job:
Entry points:
Layout:
Components:
States:
Interactions:
Data needed:
Accessibility:
Brand tokens:
Implementation notes:
Open questions:
```

## Handoff Quality Bar

Coding Lead should be able to implement without asking what belongs where.

Avoid vague phrases like “modern”, “clean”, or “intuitive” unless followed by concrete layout, spacing, and behaviour.

## Verification Checklist

- [ ] Mobile-first.
- [ ] States covered.
- [ ] Component hierarchy clear.
- [ ] Accessibility notes included.
- [ ] Brand tokens referenced.
- [ ] Implementation notes actionable.
