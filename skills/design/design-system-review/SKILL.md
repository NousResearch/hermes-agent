---
name: design-system-review
description: "Use when reviewing an app or design spec for design-system consistency, token usage, accessibility, responsive behaviour, and implementation handoff quality."
version: 1.0.0
author: KENSEI
license: MIT
metadata:
  hermes:
    tags: [design, design-system, review, accessibility, tokens]
    related_skills: [mobile-screen-spec, plenishd-design-tokens, coachos-design-tokens]
adoption_status: permanent
---

# Design System Review

## Overview

Use this skill to review whether a design or implemented UI follows the intended design system and is ready for Coding Lead or QA Lead.

## When to Use

- Reviewing UI specs before implementation.
- Reviewing screenshots or app flows.
- Checking token consistency.
- Checking accessibility design.
- Preparing handoff to Coding Lead or QA Lead.

## Review Lenses

### Brand fit

Does it match the app's design language and voice?

### Token consistency

Colours, spacing, typography, radii, shadows, and component styles should use known tokens or justify exceptions.

### State coverage

Default, loading, empty, error, disabled, success, and offline states should be defined where relevant.

### Accessibility

Check contrast, touch targets, font scaling, labels, focus order, and reduced-motion implications.

### Mobile behaviour

Check safe areas, keyboard avoidance, scroll behaviour, hit areas, and platform conventions.

### Handoff clarity

Coding Lead should know exactly what to build.

## Verdicts

Use:

- `APPROVED`
- `BLOCKED`
- `CONDITIONAL`

## Output Contract

```text
Verdict:
Scope reviewed:
Brand fit:
Token consistency:
State coverage:
Accessibility:
Mobile behaviour:
Handoff quality:
Required fixes:
```

## Common Pitfalls

- Calling something “clean” without checking consistency.
- Ignoring empty/error states.
- Designing web interactions for mobile unchanged.
- Missing accessibility until QA.
- Adding one-off styles instead of tokens.

## Verification Checklist

- [ ] Brand fit checked.
- [ ] Tokens checked.
- [ ] States checked.
- [ ] Accessibility checked.
- [ ] Mobile behaviour checked.
- [ ] Handoff clarity checked.
