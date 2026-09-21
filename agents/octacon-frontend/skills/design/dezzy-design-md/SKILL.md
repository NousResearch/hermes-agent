---
name: dezzy-design-md
description: "Author, validate, and export DESIGN.md token specification files — Google's format for documenting design tokens, component specs, and design system documentation. Outputs structured markdown with token tables, component API specs, and usage guidelines."
version: 1.0.0
author: KENSEI
adoption_status: provisional
metadata:
  hermes:
    tags: [design, tokens, specs, documentation, design-system]
    category: design
    related_skills: [design-system-review, plenishd-design-tokens, coachos-design-tokens, mobile-screen-spec]
    created_for_profile: dezzy
---

# Dezzy Design MD

Author DESIGN.md token specification files — the canonical documentation format for KENSEI's brand design systems. Each spec captures colours, typography, spacing, elevation, component APIs, and usage guidelines.

## When This Skill Activates

- "Write the DESIGN.md for Plenishd"
- "Document the design system"
- "Create token specs for the button component"
- "Update the design system documentation"
- "/dezzy-design-md"

## DESIGN.md Structure

### Required Sections

1. **Overview** — what this design system covers, which product/brand
2. **Colour System** — token name, hex value, usage, light/dark variants
3. **Typography** — token name, font family, weight, size, line-height, letter-spacing
4. **Spacing Scale** — token name, rem/px value, intended use (gap, padding, margin)
5. **Elevation** — token name, shadow value, usage (card, modal, dialog)
6. **Border Radii** — token name, value, where it applies
7. **Iconography** — style, sizes, naming convention
8. **Component Tokens** — per-component: default and variant styles

### Optional Sections

9. **Motion** — duration tokens, easing curves
10. **Breakpoints** — responsive grid breakpoints
11. **Accessibility** — contrast ratios, focus styles, touch targets

## Token Naming Convention

KENSEI tokens follow: `{category}-{property}-{variant}-{state}`

```
color-bg-primary
color-text-secondary
spacing-padding-lg
typography-heading-h1
elevation-shadow-card
radius-border-sm
motion-duration-fast
```

## Verification

- All hex values reference the correct brand palette
- Token names follow naming convention
- No orphaned tokens (every token has a usage)
- Component section has at least: default, hover, active, disabled states
- Accessibility notes include minimum contrast ratios

## Pitfalls

- Don't reference external CDN fonts unless the brand explicitly uses them
- Don't include tokens that don't have a defined usage — they add bloat
- Validate against the produced codebase — tokens should match actual component implementations
- When updating, version the DESIGN.md with a changelog