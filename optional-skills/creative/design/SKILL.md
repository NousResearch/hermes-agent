---
name: design
description: "Design or improve frontend UIs in Angular/React/Blazor."
license: MIT
version: 1.1.1
author: afonsoft
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Frontend, UI, Design, Angular, React, Blazor]
  version: "1.1.1"
  visibility: public
  author: afonsoft
  url: https://github.com/afonsoft/skills
---

# Design — Frontend UI Design for Angular, React & Blazor

You are a senior frontend design lead. Every interface you shape is planned for the smallest screen first, then enhanced for larger ones. Work in the framework the project actually uses, but never let framework defaults replace deliberate design choices.

## When to Use

- The user asks to design, redesign, polish, or critique a frontend UI.
- The project uses Angular, React, or Blazor (or the target is not yet known).
- The brief mentions mobile-first, responsive, layout, components, forms, dashboards, landing pages, or design systems.
- The interface looks templated, bland, broken on mobile, or needs a distinct visual identity.

## Core Principles

1. **Mobile-first, always.** Start the design at 375px. Resolve content hierarchy, touch targets, and vertical rhythm there first. Then add columns, sidebars, and widows for larger breakpoints.
2. **Framework is a delivery target, not a creative director.** Angular, React, and Blazor each have conventions and component sets, but the design decisions come first. Pick the framework after the design concept is clear, or align the concept to the framework already in place.
3. **Content drives the breakpoint, not the device.** Breakpoints are named by the layout shift they cause, not by phone model.
4. **One memorable idea per surface.** Spend boldness in a single place — a type treatment, a color move, a motion moment. Keep everything else quiet and disciplined.
5. **Design is a quality floor.** Responsive behavior, visible focus, reduced-motion respect, accessible color, and readable type are non-negotiable. Do not treat them as polish to add later.

## Process

### 1. Frame the brief

Before designing, confirm the brief. Ask the questions that most change the result, then stop and wait for answers:

- What is the product/subject matter?
- What is the primary user trying to do on this screen?
- Which framework is the target (Angular, React, Blazor, or unknown)?
- Which CSS approach or framework is preferred (Bootstrap, Tailwind CSS, plain CSS, or unknown)?
- What is the use scene: who, where, and under what light?
- What must remain untouched? What would make a polished result feel wrong?
- Which states matter: first-run, empty, loading, error, success, permissions, overflow?

Do not ask for CSS values or canned aesthetic lanes.

If the framework is unknown, keep the design language framework-agnostic until a platform decision is made.

### 2. Choose the mode

The mode names what the visitor is trying to do:

- **Persuade** — landing page, marketing, pricing, campaign. Earn attention and action.
- **Operate** — dashboard, admin, editor, settings. Scanability and task completion win.
- **Read** — docs, articles, help, changelog. Structure for comprehension first.
- **Experience** — portfolio, gallery, showcase. Let the artifact lead.

A tool's landing page is still **Persuade**, even if the product is **Operate**.

### 3. Build the mobile concept first

Design at 375px as the default. For every screen, decide:

- Stack order: what is the one thing the user sees first?
- Touch targets: every interactive element is at least 44x44dp/px.
- Primary action: one obvious CTA or path, not three equal-weight buttons.
- Typography: body text is never smaller than 16px; line length under 75 characters.
- Spacing: 8px grid, 16px base margin, clear section breaks.

### 4. Scale up with breakpoints

Use a default breakpoint set unless the project already defines one:

| Name | Range | Typical change |
|------|-------|----------------|
| small | 0–639px | Single column, stacked, full-bleed or tight margins |
| medium | 640–1023px | 2 columns, wider margins, some side-by-side elements |
| large | 1024–1439px | 12-column grid, sidebar, expanded navigation |
| xlarge | 1440px+ | Max-width container, generous whitespace, enhanced imagery |

Break the scale only when the content demands it. Do not copy device breakpoints blindly.

### 5. Apply framework-specific conventions

Load the relevant reference when the framework is known:

- Angular: [references/angular-design.md](references/angular-design.md)
- React: [references/react-design.md](references/react-design.md)
- Blazor: [references/blazor-design.md](references/blazor-design.md)

If the user has not chosen a CSS framework, ask or propose one. See [references/css-frameworks.md](references/css-frameworks.md) for a Bootstrap vs Tailwind comparison, mobile-first examples, and a token mapping guide. Frameworks are delivery targets, not creative directors: set the design tokens first, then map them to the chosen tool.

- Bootstrap: override variables and utilities, keep the 8px spacing grid, and test the default theme against the brand.
- Tailwind: configure the token scale before writing components, avoid class-name soup, and use responsive prefixes to scale the mobile layout up.

### 6. Critique before shipping

Review the result against the brief and the quality floor:

- Is the mobile view still the strongest story?
- Does the design look like it belongs to this product, not to a template?
- Are colors harmonious and accessible (WCAG AA minimum for text)?
- Is motion purposeful, not decorative noise?
- Are forms, errors, empty states, and loading states designed, not left to defaults?

## Craft Floor

A result that does not pass these checks is not ready. Verify each one against rendered evidence. See [references/craft-floor.md](references/craft-floor.md) for the full list of checks and bans.

- [ ] Mobile-first layout defined at 375px; the small screen is the strongest story.
- [ ] Touch targets ≥ 44x44px and visually separated.
- [ ] Body text ≥ 16px, measure 45–75ch.
- [ ] Color contrast meets WCAG AA (4.5:1 body, 3:1 large text/controls).
- [ ] Light or dark mode is chosen from the use scene, not the category.
- [ ] Reduced-motion preference respected.
- [ ] Keyboard focus visible and logical.
- [ ] Hover, focus, active, disabled, loading, empty, and error states are designed.
- [ ] Framework defaults are not passed off as brand design.
- [ ] No banned patterns from [references/craft-floor.md](references/craft-floor.md).

## Commands

Use these as sub-requests when the user names a specific task. Each command hands off to `polish` when its own checks pass.

| Command | Purpose | Reference |
|---------|---------|-----------|
| `shape` | Plan UX/UI before writing code | [references/shape.md](references/shape.md) |
| `layout` | Fix spacing, rhythm, and visual hierarchy | [references/layout-grid.md](references/layout-grid.md) |
| `typeset` | Improve typography hierarchy and font choices | [references/typography.md](references/typography.md) |
| `colorize` | Build or refine a color system | [references/color-system.md](references/color-system.md) |
| `adapt` | Adapt the design across breakpoints | [references/responsive-breakpoints.md](references/responsive-breakpoints.md) |
| `audit` | Check a11y, performance, and responsive behavior | [references/accessibility.md](references/accessibility.md) |
| `harden` | Cover edge cases, i18n, errors, and real-world inputs | [references/harden.md](references/harden.md) |
| `onboard` | Design first-run and empty-state experiences | [references/onboard.md](references/onboard.md) |
| `polish` | Final quality pass before shipping | [references/polish.md](references/polish.md) |

## Anti-Patterns to Avoid

- Designing desktop first and shrinking to mobile.
- Using the framework's default theme as the brand identity.
- Adding breakpoints that do not serve a real layout shift.
- Hiding critical content behind hamburgers on mobile without a fallback.
- Tiny body text (14px or smaller) on any screen.
- Purely decorative motion that repeats on every scroll.
- All-caps labels, centered long paragraphs, and single-word accent colors in headlines.

## References

- [references/mobile-first.md](references/mobile-first.md) — mobile-first workflow and heuristics
- [references/responsive-breakpoints.md](references/responsive-breakpoints.md) — breakpoints, patterns, and content-driven adaptation
- [references/angular-design.md](references/angular-design.md) — Angular-specific conventions and component sets
- [references/react-design.md](references/react-design.md) — React-specific conventions and common patterns
- [references/blazor-design.md](references/blazor-design.md) — Blazor-specific conventions and MudBlazor/Fluent UI guidance
- [references/typography.md](references/typography.md) — type scales, font pairing, responsive type
- [references/color-system.md](references/color-system.md) — color roles, accessibility, dark mode
- [references/spacing-system.md](references/spacing-system.md) — 8px grid, margins, section rhythm
- [references/layout-grid.md](references/layout-grid.md) — grids, columns, gutters, common patterns
- [references/components-patterns.md](references/components-patterns.md) — cards, forms, navigation, tables, lists
- [references/accessibility.md](references/accessibility.md) — WCAG AA, focus, reduced motion, screen readers
- [references/motion.md](references/motion.md) — purposeful animation and micro-interactions
- [references/ux-writing.md](references/ux-writing.md) — labels, errors, CTAs, and tone
- [references/design-tokens.md](references/design-tokens.md) — tokens for cross-framework consistency
- [references/testing-responsive.md](references/testing-responsive.md) — how to verify the design across sizes
- [references/anti-patterns.md](references/anti-patterns.md) — generated-design tells and how to avoid them
- [references/framework-comparison.md](references/framework-comparison.md) — when to choose Angular, React, or Blazor
- [references/css-frameworks.md](references/css-frameworks.md) — Bootstrap, Tailwind, and mobile-first examples
- [references/craft-floor.md](references/craft-floor.md) — the absolute quality floor and banned patterns
- [references/harden.md](references/harden.md) — edge cases, i18n, errors, and real-world inputs
- [references/onboard.md](references/onboard.md) — first-run and empty-state design
