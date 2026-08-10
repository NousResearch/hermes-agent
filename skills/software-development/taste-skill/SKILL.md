---
name: taste-skill
description: "Anti-slop frontend design skill for landing pages, portfolios, and redesigns. Reads the brief, infers the right design direction, and ships interfaces that don't look templated."
version: 1.0.0
author: Hermes Agent (adapted from tasteskill)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [design, frontend, UI, landing-page, portfolio, anti-slop]
    related_skills: [hallmark, design-md-library, baseline-ui]
---

# Taste Skill — Anti-Slop Frontend Design

Landing pages, portfolios, and redesigns. Not dashboards, not data tables, not multi-step product UI.

## 0. Brief Inference (Read the Room First)

Before touching code, infer what the user actually wants:

1. **Page kind** — landing (SaaS/consumer/agency/event), portfolio (dev/designer/studio), redesign, editorial/blog
2. **Vibe words** — "minimalist", "Linear-style", "Apple-y", "playful", "serious B2B", "editorial", "glassy", "dark tech"
3. **Reference signals** — URLs they linked, screenshots, brands they're competing with
4. **Audience** — B2B procurement vs. design-conscious consumer vs. recruiter
5. **Brand assets** — logo, color, type, photography that already exist
6. **Quiet constraints** — accessibility-first, public-sector, regulated, trust-first

### Output a one-line "Design Read" before generating

_"Reading this as: <page kind> for <audience>, with a <vibe> language, leaning toward <design system>."_

### Anti-Default Discipline

Do not default to: AI-purple gradients, centered hero over dark mesh, three equal feature cards, generic glassmorphism, infinite-loop micro-animations, Inter + slate-900. Reach past them deliberately.

## 1. The Three Dials

| Dial                 | 1                | 10                  |
| -------------------- | ---------------- | ------------------- |
| **DESIGN_VARIANCE**  | Perfect Symmetry | Artsy Chaos         |
| **MOTION_INTENSITY** | Static           | Cinematic/Physics   |
| **VISUAL_DENSITY**   | Art Gallery/Airy | Cockpit/Packed Data |

**Baseline:** 8 / 6 / 4. Override based on design read.

## 2. Typography System

- **2+1 font discipline**: display + body, mono only for code
- Pair fonts by purpose (serif display + sans body, or sans display + sans body in different weights)
- Use Google Fonts or `next/font` for self-hosted
- Anchor sizes: display `clamp(2.5rem, 5vw, 5rem)`, body `1rem` / `1.25rem`

## 3. Color System

- **Dominant + accent + neutral hierarchy**, not 50-50 split
- Accent is a spice, not a blanket — 5-15% of visible color area
- OKLCH for perceptual uniformity
- Dark mode as a sibling, not a negative

## 4. Spacing & Layout

- 4-pt or 8-pt spacing scale
- CSS Grid for page-level, Flexbox for component-level
- Generous whitespace: 120-160px section padding
- `max-width: 1280px` content wrapper with `margin: 0 auto`

## 5. Motion

- Purpose over decoration: entrance, emphasis, state change, scroll reveal
- `prefers-reduced-motion` respected always
- Duration: 150-400ms; easing: `cubic-bezier(0.16, 1, 0.3, 1)` (ease-out-expo feel)

## Hermes Integration

- Use `read_file` to scan existing project for brand assets
- Use `design-md-library` skill for brand-specific design tokens
- Use `hallmark` skill for full-page design workflow
- Use `browser_navigate` + `browser_snapshot` to study reference sites
- Use `write_file` to create design artifacts
