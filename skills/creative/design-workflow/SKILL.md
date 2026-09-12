---
name: design-workflow
description: "Use when designing, redesigning, critiquing, polishing, or auditing landing pages, product websites, web apps, dashboards, frontend UI flows, and native iOS/Android product screens. Routes work through product clarity, visual direction, style dials, implementation guidance, visual verification, and UX/accessibility audit."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [design, frontend, mobile, ios, android, ui, ux, landing-page, dashboard, audit]
    related_skills: [claude-design, sketch, popular-web-designs, design-md, hermes-agent-skill-authoring]
---

# Design Workflow

## Overview

`design-workflow` is a product UI design director skill. Use it to move from vague product intent to a clear, usable, polished interface across Landing Pages, product websites, Web Apps, dashboards, and native iOS/Android product screens.

The goal is not to make every screen flashy. The goal is to make the product understandable, credible, usable, emotionally coherent, and shippable. The workflow combines four ideas:

1. **Frontend design direction** — decide the aesthetic and product point of view before coding or drawing.
2. **Impeccable-style process** — shape, craft, critique, polish, harden, adapt, and audit in explicit stages.
3. **Taste dials** — tune variance, motion, and density instead of relying on vague words like "高级" or "more modern".
4. **Final quality gate** — check accessibility, responsiveness, platform conventions, states, copy, and performance before calling work done.

This skill is intentionally an umbrella. Do not load every reference for every request. Route the task first, then use only the relevant references.

## When to Use

Use this skill when the task includes any of these:

- Creating or redesigning a Landing Page, product website, SaaS homepage, app screen, dashboard, onboarding flow, pricing page, settings page, or marketing/product UI.
- Improving a UI that feels generic, AI-generated, cluttered, low-trust, visually weak, or unclear.
- Exploring multiple visual directions before implementation.
- Turning product requirements into page/screen structure, hierarchy, copy, and interaction guidance.
- Auditing UI/UX before shipping.
- Translating a reference image, screenshot, or competitor page into a practical design direction.
- Designing native iOS/Android screens while respecting platform conventions rather than forcing web patterns.

## When Not to Use

Do not use this as the primary skill for:

- Pure backend work, infrastructure, data pipelines, or API-only tasks.
- Brand identity systems where the user wants a full VI/logo/brand-strategy package rather than product UI.
- Graphic illustration, comics, generative art, diagrams, or slides unless they are part of a product UI task.
- Native mobile implementation details that require platform-specific build/debug workflows; use this for design/product/UX guidance, then pair with platform-specific development workflows if needed.

## Core Principles

1. **Understand before beautifying.** A beautiful screen that users cannot understand is a failed screen.
2. **One primary job per screen/section.** Every view needs a clear main action, main message, and main hierarchy.
3. **Design is persuasion order, not decoration.** Arrange information as: understand → relevance → trust → action.
4. **Visuals serve hierarchy.** Color, spacing, type, motion, and imagery should clarify importance and state.
5. **Real states beat ideal screenshots.** Check loading, empty, error, long text, disabled, focus, hover, active, success, warning, and extreme data.
6. **Respect platform and project constraints.** Existing design systems, brand tokens, accessibility rules, and iOS/Android conventions outrank generic style preferences.
7. **No heavy dependencies by default.** Do not introduce Framer Motion, GSAP, Three.js, Lottie, Rive, Lenis, or similar libraries unless the project already uses them, the user explicitly asks, or the interaction is core to the product and the trade-off is explained.
8. **Avoid AI-template defaults.** Do not unconsciously ship blue-purple gradients, centered hero + three cards, vague "AI-powered" copy, generic icon grids, excessive glassmorphism, or placeholder-only content.

## First Move: Route the Task

Before making or editing a UI, classify the user's request:

| User intent | Route | Primary references | Typical output |
|---|---|---|---|
| New page/screen | Understand → Shape → Direction → Craft → Verify → Polish → Audit | `routing.md`, `principles.md`, `frontend-direction.md`, `style-dials.md` | design brief, structure, implementation guidance, verification notes |
| Existing UI feels weak/AI/generic | Critique → Distill → Repair → Style Tune → Verify → Audit | `anti-ai-patterns.md`, `impeccable-flow.md`, `visual-verification.md` | critique report, prioritized fixes, redesign plan |
| Style exploration | Style Brief → 2-3 Concepts → Compare → Pick → Craft | `frontend-direction.md`, `style-dials.md`; optionally `sketch`, `claude-design`, `popular-web-designs` | visual directions with trade-offs |
| Final review before shipping | Audit → Fix P0/P1 → Recheck | `audit-checklist.md`, `visual-verification.md` | P0/P1/P2 report with file/screen references |
| Native iOS/Android screen | Platform Context → Shape → Native Direction → Interaction/State Review → Audit | `principles.md`, `style-dials.md`, `audit-checklist.md` | screen brief, platform-convention notes, state checklist |

If the intent is ambiguous but obviously design-related, choose the safest route and state the assumption briefly. Ask a question only if the answer would materially change the route or platform.

## Default Workflow

### 1. Understand

Capture enough product context to avoid generic design:

- Who is the user?
- What job are they trying to accomplish?
- What is the screen/page goal?
- What is the primary action?
- What must be trusted before the user acts?
- What platform is this for: web, responsive web, iOS, Android, or cross-platform?
- What design system, component library, brand, or technical constraints already exist?

For small tasks, infer what you can and proceed. For larger ambiguous tasks, produce a compact `Design Brief` first.

### 2. Shape

Define information architecture before visuals:

- Main message or screen title.
- Primary CTA/action.
- Secondary actions.
- Section/order or screen zones.
- Required states and edge cases.
- Data density and interaction complexity.

### 3. Direction

Make an explicit aesthetic decision:

- Mood: calm, premium, editorial, industrial, playful, trustworthy, technical, warm, utilitarian, etc.
- Reference family: not copied, but useful as design vocabulary.
- Type direction, spacing rhythm, color strategy, component sharpness/roundness, imagery style.
- Dials: variance, motion, density.

### 4. Craft

Implement or specify the UI according to the project context:

- Use existing tokens/components first.
- Prefer semantic HTML and accessible components on web.
- Prefer native platform controls and conventions on iOS/Android unless there is a strong product reason not to.
- Write concrete copy; avoid placeholder marketing language.
- Include realistic data and states.

### 5. Verify Visually and Functionally

When tools are available, inspect the actual result rather than trusting code:

- Desktop and mobile/responsive web screenshots for web work.
- Key breakpoints, overflow, focus, hover/touch, and reduced-motion behavior when relevant.
- For AI-backed upload/analyze flows, verify a real fixture through the provider/API path in addition to mocks; screenshots alone do not prove the core workflow works.
- For native mobile designs, check compact width, thumb reach, safe areas, status/navigation bars, keyboard behavior, modal sheets, and platform back behavior.

### 6. Polish

Use targeted passes rather than random tweaking:

- `layout` — spacing, alignment, grouping, rhythm.
- `typeset` — type scale, measure, hierarchy, truncation, localization.
- `colorize` — semantic color, contrast, brand fit, dark/light modes.
- `clarify` — UX copy, labels, CTAs, error text.
- `distill` — remove anything not serving the screen's job.
- `bolder` / `quieter` — adjust visual intensity deliberately.
- `harden` — real states, data extremes, interaction edge cases.
- `adapt` — platform, screen size, and input method adaptation.

### 7. Audit

Before finalizing, produce a prioritized quality report:

- **P0** — blocks usability, accessibility, comprehension, core conversion, or platform correctness.
- **P1** — materially hurts experience or trust.
- **P2** — polish and quality improvements.
- **Pass** — important checks that are already OK.
- **Not checked** — anything that could not be verified.

## Style Dials

Use numbers to make taste controllable:

- **Design Variance 1-10** — from strict, conventional, highly systematic to asymmetric, distinctive, surprising.
- **Motion Intensity 1-10** — from static/instant to expressive/kinetic. Default lower for dashboards and accessibility-heavy contexts.
- **Visual Density 1-10** — from spacious, editorial, focused to dense, data-rich, operational.

Default starting points:

| Product type | Variance | Motion | Density |
|---|---:|---:|---:|
| B2B SaaS landing | 5 | 3 | 5 |
| Developer tool | 4 | 2 | 7 |
| Data dashboard | 2 | 1 | 8 |
| Premium consumer landing | 7 | 4 | 3 |
| iOS productivity app | 4 | 3 | 6 |
| Android utility app | 3 | 2 | 7 |
| Editorial/product story | 7 | 4 | 4 |

Treat these as defaults, not laws.

## Native iOS/Android Compatibility

This skill supports native mobile design, but it must not flatten platform differences.

For iOS:

- Respect safe areas, Dynamic Type, SF Symbols conventions, navigation bars, tab bars, sheets, swipe/back gestures, haptics, permission prompts, and keyboard behavior.
- Prefer platform-native interaction patterns unless the product has a strong brand reason to differ.
- Check light/dark mode, large text, VoiceOver labels, touch targets, and one-handed reach.

For Android:

- Respect Material conventions where relevant, system back, navigation bars/gesture nav, status bar contrast, edge-to-edge layout, bottom sheets, snackbars, permission flows, and TalkBack labels.
- Check density buckets, dynamic color when appropriate, text scaling, touch targets, and hardware/software back behavior.

For cross-platform apps:

- Share product hierarchy and design intent, not every pixel.
- Let platform primitives differ where users expect them to differ.

## Output Formats

### Design Brief

```markdown
## Design Brief
- User:
- Situation:
- Screen/page goal:
- Primary action:
- Trust proof / risk reducer:
- Platform(s):
- Existing constraints:
- Aesthetic direction:
- Style dials: variance / motion / density
- Key sections or screen zones:
- Required states:
- Risks:
```

### Critique Report

```markdown
## Design Critique
### Summary
### What works
### P0
- Screen/file: issue → why it matters → fix direction
### P1
### P2
### Recommended fix order
```

### Audit Report

```markdown
## UI/UX Audit
### P0 Must Fix
### P1 Should Fix
### P2 Polish
### Passed
### Not Checked
### Verification performed
```

## Related Local Skills

Reference: `references/screenshot-to-design-system-case.md` captures a worked example for products that turn screenshots/websites into extracted design systems and restoration prompts.
Reference: `references/ai-vision-upload-flows.md` captures implementation and verification lessons for AI-backed screenshot/image upload analysis flows, including tall-screenshot compression and retry/fallback handling.

- Use `sketch` when 2-3 quick HTML design variants are useful before committing to one direction.
- Use `claude-design` for one-off polished HTML artifacts, prototypes, decks, or visual concept pages.
- Use `popular-web-designs` when the user asks for a recognizable design-system flavor or benchmark family.
- Use `design-md` when the project needs durable design tokens or a formal `DESIGN.md` spec.

## Common Pitfalls

1. **Jumping straight to code.** Always establish the screen job, hierarchy, and direction first unless the user explicitly asks for a tiny visual fix.
2. **Over-designing dashboards.** Operational UIs often need density, predictability, and fast scanning more than novelty.
3. **Forcing web aesthetics onto native apps.** Native mobile users expect platform navigation, gestures, typography behavior, safe areas, and accessibility semantics.
4. **Using taste as a substitute for strategy.** A style preset cannot fix unclear value proposition or broken information architecture.
5. **Ignoring project constraints.** Existing tokens, components, and brand rules are inputs, not obstacles.
6. **Adding animation because it looks good.** Motion must explain state, preserve orientation, or add product meaning. Respect reduced-motion settings.
7. **Audit without verification.** If you did not open the UI, inspect a screenshot, or read the relevant code, mark items as not checked rather than pretending.

## Verification Checklist

Before considering a design task done:

- [ ] Task route was chosen and stated or implicitly followed.
- [ ] User, goal, primary action, and platform are clear.
- [ ] Aesthetic direction and style dials are explicit when visual direction matters.
- [ ] Existing design system/platform conventions were respected.
- [ ] Real states and edge cases were considered.
- [ ] Visual verification was performed when implementation or screenshots were available.
- [ ] Accessibility and responsive/platform checks were included.
- [ ] Final output includes either implementation, prioritized fixes, or a clear next action.
