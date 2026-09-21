---
name: dezzy-html-prototype
description: "Generate throwaway HTML mockups for rapid design exploration — produce 2-3 visual variants of a screen or component for side-by-side comparison before committing to final design direction."
version: 1.0.0
author: KENSEI
adoption_status: provisional
metadata:
  hermes:
    tags: [design, prototype, html, mockup, rapid]
    category: design
    related_skills: [mobile-screen-spec, dezzy-design-systems]
    created_for_profile: dezzy
---

# Dezzy HTML Prototype

Rapid HTML mockups for design exploration. Generate 2-3 visual variants of a UI concept so Dezzy and Sahil can compare approaches before committing to final design direction.

## When This Skill Activates

- "Mock up this screen"
- "Show me two ways this could look"
- "Create a prototype of the dashboard"
- "Rapid prototype the onboarding flow"
- "/dezzy-html-prototype"

## Workflow

1. **Understand the scope** — single screen, multi-step flow, or component variant?
2. **Load relevant design context** — load plenishd-design-tokens, coachos-design-tokens, or popular-web-designs SKILL.md for the brand's visual vocabulary
3. **Generate 2-3 variants** — each as a standalone HTML file with inline CSS
4. **Each variant must differ meaningfully** — layout approach, visual hierarchy, interaction model, not just colours
5. **Save each** — `prototypes/<screen-name>/variant-<1|2|3>.html` with an index page linking all three

## KENSEI Design System Integration

When prototyping for a known brand (Plenishd, CoachOS, MatchdayMaestro), load the relevant design-tokens skill first and reference:
- Primary, secondary, accent colour values
- Typography stack (font family, weights, sizes)
- Spacing scale (4px grid or 8px grid)
- Border radius conventions
- Shadow/ elevation tokens

## Variant Strategies

| Strategy | When |
|----------|------|
| Sidebar vs top-nav | Navigation layout exploration |
| Card vs list | Content density comparison |
| Dark vs light | Theme direction |
| Modal vs inline | Interaction pattern choice |
| Wizard vs single-page | Multi-step flow preference |

## Verification

- Each variant opens in browser without errors
- Design tokens are applied consistently within each variant
- Index page links all variants for side-by-side comparison
- Files are under `prototypes/` not mixed with production code

## Pitfalls

- Don't polish one variant more than others — they should be equal fidelity
- Default to our design system tokens unless exploring a new direction
- Keep styles inline — no build step, no external dependencies
- Label the variants clearly in the filename and page title