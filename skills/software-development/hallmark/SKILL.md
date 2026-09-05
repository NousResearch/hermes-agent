---
name: hallmark
description: 'Anti-AI-slop design skill for greenfield pages, audits, redesigns, and design extraction. Use when building new apps, landing pages, redesigning, or invoking hallmark by name.'
version: 1.1.0
author: Hermes Agent (adapted from hallmark)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [design, UI, frontend, landing-page, anti-slop, redesign, audit]
    related_skills: [design-md-library, popular-web-designs, baseline-ui, taste-skill]
---

# Hallmark

A design skill for AI coding assistants. Makes the UIs they generate look made, not generated.

Hallmark is opinionated, short, and boring on purpose. It encodes a tight set of rules drawn from the consensus of the anti-AI-slop design field and refuses to let the model fall back to the defaults every LLM was trained on.

The differentiator: Hallmark insists on **structural variety**, not just visual variety. Two pages by Hallmark for two different briefs should not share the same hero → 3-feature → CTA → footer rhythm. They should feel like different sites, not different colour-swaps of the same template.

## How to use this skill

Hallmark has one default behaviour and three explicit verbs.

| Invocation                           | What it does                                                                                                                                                                                                             |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| _(default)_                          | The user asked you to design or build something new. Follow the **Design flow** below.                                                                                                                                   |
| `hallmark audit <target>`            | Read the target, score it against the anti-pattern list, return a ranked punch list. **Do not edit.**                                                                                                                    |
| `hallmark redesign <target>`         | Take the target's content and intent, then redesign the visual structure inside the existing implementation boundaries. Preserve existing routes, component ownership, copy intent, brand, and information architecture. |
| `hallmark study <screenshot or URL>` | Extract the DNA — macrostructure, archetypes, type-pairing, colour anchor — and produce a diagnosis report.                                                                                                              |

## Six Disciplines (apply across all verbs)

1. **Pre-emit self-critique.** Before handing back any output, score it 1–5 on six axes — Philosophy, Hierarchy, Execution, Specificity, Restraint, Variety. Anything < 3 triggers a revision pass.

2. **Honest copy — no fabricated content.** If the user did not supply a metric, do not invent one. "+47% conversion", "trusted by 50,000+ teams", and "10× faster" are slop the moment they're invented.

3. **Locked tokens — no mid-render improvisation.** Once a theme is selected, every colour and every `font-family` declaration must reference a named token (`var(--color-accent)`, `font-family: var(--font-display)`). Inline values are not allowed.

4. **Re-drawn chrome forbidden.** Hallmark must not hand-build fake browser bars, fake phone frames, fake code-block windows, or fake IDE chrome. Use real screenshots wrapped in a `<figure>`, or omit the chrome.

5. **Mobile responsiveness — verified at 320/375/414/768px.** No horizontal scroll. No two-line clickable text. Image grids use `minmax(0, 1fr)`. Display headers wrap via `overflow-wrap: anywhere`.

6. **Typography purity — no italic headers.** Headings and display type are always roman (`font-style: normal`). Carry emphasis with weight, accent colour, or a drawn underline.

## Design flow (default)

### 0. Pre-flight scan

If the project already has code, read it before asking the user anything. Scan for:

- **Font stack** — `package.json` for font deps, Tailwind config, CSS imports
- **Palette** — OKLCH/HSL/hex in `:root`, Tailwind `theme.extend.colors`, design tokens
- **Microinteraction stance** — `framer-motion`, `gsap`, `motion` in deps = "motion-on"
- **Spacing scale** — Tailwind `theme.extend.spacing`, CSS `--space-*` pattern
- **Framework** — Next.js, Astro, Vue, Svelte, Remix, or vanilla HTML

Emit findings before proceeding. Preserve existing tokens; introduce macrostructure and microinteraction discipline.

### 1. Genre detection

Classify the brief into one of four genres:

- **Editorial** — content-first, strong typography, minimal chrome
- **Modern-minimal** — clean, spacious, geometric
- **Atmospheric** — moody, immersive, background-driven
- **Playful** — bold colours, rounded corners, animated

### 2. Macrostructure pick

Choose a section rhythm that fits the brief. Avoid the default hero → 3-feature → CTA → footer template. Rotate through structural archetypes.

### 3. Theme selection

Pick from the catalog of 20 named themes or construct a custom OKLCH palette. Default to catalog rotation.

### 4. Enrichment

Add hero polish, demo visuals, or abstract backgrounds as the genre demands.

### 5. Multi-section preview

Emit the full page with all sections, verified against the slop test.

## Hermes Integration

- Use `read_file` to scan existing project code (pre-flight)
- Use `write_file` to create new UI files
- Use `browser_navigate` + `browser_snapshot` to study reference sites
- Use `design-md-library` skill for brand-specific design tokens
- Use `popular-web-designs` skill for additional design references
- Use `terminal` for build commands to verify output
