---
name: claude-design
description: Use when designing a one-off HTML landing page, prototype, deck, or visual study; turn a brief into a deliberate, verified artifact.
version: 1.1.1
author: BadTechBandit
license: MIT
platforms: [linux, macos, windows]
metadata:
  author: BadTechBandit
  tags: [design, html, prototype, ux, ui]
  hermes:
    tags: [design, html, prototype, ux, ui, creative, artifact, deck, motion, design-system]
    related_skills: [design-md, popular-web-designs, excalidraw, architecture-diagram]
---

## Purpose

Produce thoughtful local HTML artifacts rather than generic web mockups. Use this skill for landing pages, high-fidelity prototypes, option boards, component labs, HTML decks, motion studies, onboarding flows, dashboards, and product surfaces. Use `popular-web-designs` when a known product's visual language is requested; use `design-md` when the deliverable is a token specification.

## Runtime and prerequisites

This skill runs in CLI/API mode. Ignore hosted-only Claude Design tools, panes, callbacks, and prompt schemas. Default output is a complete local HTML file with embedded CSS and JavaScript. No API key or special dependency is required. Requirements: a writable output path, a text editor or code-writing tool, and a browser or static checker when visual verification is needed. If working in an existing repo, inspect its actual stack, tokens, components, and routes first.

## Requirements

No API key or special dependency is required. Use a writable output path and a text editor or code-writing tool. Use a browser or static checker when visual verification is needed. In an existing repo, inspect its actual stack, tokens, components, and routes before editing.

## Workflow

1. Read the brief and identify audience, fidelity, locked constraints, output format, and whether variants are needed.
2. Gather context: brand docs, screenshots, repo styles, tokens, copy, assets, and legal/product constraints. If fidelity depends on missing context, ask focused questions.
3. Commit to one primary surface before choosing tokens: **Monitor**, **Operate**, **Compare**, **Configure**, **Decide/Learn**, **Explore**, or **Command/Inspect**. A dashboard is usually Monitor, not a marketing hero.
4. Define a small system: palette, type, spacing, radii, elevation, motion posture, and interaction rules. Choose composition before decoration.
5. Pick the artifact form: self-contained HTML for standalone work; clickable flow for prototypes; fixed 1920×1080 canvas for decks; component lab for variants.
6. Build with semantic HTML, CSS variables, grid/flex, responsive behavior, real focus/hover states, 44px mobile hit targets, and `prefers-reduced-motion` handling. Avoid unnecessary dependencies.
7. Add only content that earns its place. Mark non-final copy as draft; never invent claims, metrics, testimonials, or product behavior.
8. Verify the file exists, run syntax/static checks, open it in a browser when available, inspect the primary viewport, test key interactions, and score the slop diagnostic below. Repair causes (composition) before symptoms (color).
9. Report the exact path, what exists, what was verified, and any limitation.

## Surface-first composition

- **Monitor:** density and glanceable state; no marketing hero.
- **Operate:** actions, selection, queues, and feedback dominate.
- **Compare:** aligned columns and parity; emphasize one differentiator.
- **Configure:** progressive disclosure, validation, and save states.
- **Decide/Learn:** one idea per section; a hero may be appropriate.
- **Explore:** filters, results, zoom, and peek.
- **Command/Inspect:** keyboard speed and focused detail.

Do not average two surfaces into an unfocused layout. Name the primary one and treat the other as secondary.

## Artifact rules

- Prefer one descriptive, directly-openable HTML file with embedded `<style>` and `<script>`.
- For major revisions preserve the prior file and use a versioned filename or in-page variant switcher.
- For prototypes make the primary path clickable and include relevant default, loading, empty, error, and success states.
- For decks include keyboard navigation, visible slide count, stable slide IDs, localStorage slide persistence, and print behavior when practical.
- For exploration use three meaningful directions: conservative, strong-fit, and divergent. Do not make color-only variants unless color is the actual question.
- React is optional only when meaningful state or repo fidelity warrants it; pin CDN versions and avoid global style collisions.

## Quality checks

Before finalizing, record a short slop score from 0–10 for: tech gradient, generic tech hue, equal feature tiles, accent rails, unearned blur, monument stats, icon toppers, center-stack composition, default type, and wrong surface. Re-score after repair. Do not declare done while composition or surface errors remain.

Check contrast for important text and controls, responsive breakpoints, focus visibility, keyboard paths, reduced motion, and console errors. Never claim browser verification unless it happened.

## Anti-slop and taste rules

Avoid default glassmorphism, gradients, rainbow palettes, generic SaaS cards, decorative icons, fake dashboards, stock-photo heroes, oversized rounded rectangles, vague labels, and filler sections. Use type, scale, rhythm, alignment, density, and real imagery before adding boxes or decoration. Transform reference principles into an original design; do not clone proprietary screens or copy protected content.

## Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| The artifact looks generic | The surface or composition was not chosen first | Name the primary surface and re-layout before changing colors |
| The file does not open cleanly | Broken HTML, CSS, JavaScript, or an unavailable dependency | Run a static check, keep standalone dependencies local, and inspect browser console errors |
| The prototype cannot be tested | The primary path or important states are not wired | Add clickable default, loading, empty, error, and success states relevant to the brief |
| Mobile controls are hard to use | Hit targets or responsive rules are too small or rigid | Use at least 44px targets and test the primary viewport plus a narrow breakpoint |

## Examples

### Standalone prototype

```text
Surface: Command/Inspect. Create /tmp/Inspector.html as a self-contained prototype with keyboard navigation, default/loading/error states, 44px controls, and reduced-motion handling. Verify it opens locally and report the exact path.
```

### Existing repository

```text
First inspect the repo's theme, tokens, layout, and component files. Reuse their values and interaction patterns, then implement the requested screen in the existing stack rather than creating a disconnected HTML mockup.
```

## References

The original detailed design doctrine, surface descriptions, deck/prototype rules, typography/color guidance, source-fidelity notes, and troubleshooting material are preserved in `references/full-design-guidance.md`. Consult it when the brief needs deeper treatment.

## Limitations

This is design guidance, not a substitute for product judgment, accessibility review, browser testing, or legal review. Results vary by artifact and environment.
