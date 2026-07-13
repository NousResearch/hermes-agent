---
name: artdirector
description: Use when designing, rebuilding, auditing, or shipping any user-facing product UI. Enforces a real component design system, reference-grounded art direction, anti-AI-slop constraints, screenshot QA, and repository-level prevention rules before completion.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [ui, ux, art-direction, design-system, anti-slop, visual-qa]
    related_skills: [money-project-uiux-expert, lazyweb-design, codex-ai-slop-cleaner]
---

# Art Director

## Overview

Art Director is the strict design authority for user-facing product work. It prevents a functioning interface from being reported as complete when it still looks generic, template-derived, visually inconsistent, or AI-generated.

The standard is not “clean.” The standard is a coherent product someone could trust, use, and pay for. A markdown design document alone is not a design system. A component library alone is not art direction. Passing build and overflow checks alone is not visual QA.

## When to Use

Use this skill before and during:

- new web or mobile product screens;
- redesigns and UI-quality improvements;
- B2C, fashion, commerce, media, creator, and consumer-service interfaces;
- design-system adoption or migration;
- any task where the user says quality, taste, consistency, premium, strict, or AI slop;
- final approval before deployment.

## Core Operating Contract

1. Inspect the actual product, repository, screenshots, references, and current component stack before proposing a direction.
2. Preserve product behavior with existing tests or targeted regression coverage.
3. Define or repair the design system before polishing individual screens.
4. Implement through shared primitives and semantic tokens.
5. Capture exact target-viewport screenshots.
6. Perform strict visual review; do not infer quality from tests.
7. Encode newly discovered strict rules in the repository through docs and automated checks whenever possible.
8. Do not report completion or deploy while a hard visual failure remains.

## Real Design-System Gate

A design system exists only when all of the following are true:

- a real primitive/component layer is used, such as shadcn-style components, Radix/Base UI primitives, or a deliberate equivalent;
- semantic tokens drive color, typography, spacing, radius, shadow, and control sizes;
- components expose finite, typed variants rather than per-screen visual overrides;
- hover, focus, active, selected, loading, error, and disabled states are coherent;
- route CSS owns layout only and does not silently redefine primitive identity;
- automated checks prevent raw-value drift where practical.

A `DESIGN.md`, loose CSS variables, Tailwind utilities, or shadcn defaults alone do not satisfy this gate.

## Anti-AI-Slop Brand System

### Color

- Start with one neutral canvas family, one ink family, and exactly one brand accent family.
- Add another family only when product semantics require it, not for decoration.
- Define color roles and usage boundaries before assigning hex values.
- Do not mix fashionable-sounding clay, oxblood, moss, sky, sand, stone, purple, or gradient accents without a semantic reason.
- Photographs may retain native color; UI chrome must remain disciplined.
- Avoid beige editorial styling as an automatic shortcut to “premium.”

### Typography

- Load the chosen font for real; a font-family name without a valid asset is a failure.
- Prefer repository-pinned/local font assets over fragile runtime CDN dependencies.
- Define Display, Title, Body, Label, and Caption roles with explicit size, line-height, and allowed weights.
- Use a small weight set such as 400/500/600/700.
- Forbid arbitrary weights such as 760, 780, 820, 850, or 880.
- Validate Korean wrapping with `word-break: keep-all` and screenshots; overflow checks do not catch ugly line breaks.

### Spacing and Size

- Use a 4px-based spacing scale.
- Define a constrained control-height scale; mobile targets are at least 48px.
- Do not accumulate unexplained one-off heights such as 58/54/52/34 unless they map to documented tokens.
- Use no more than three radius values and avoid pills by default.
- Use no more than two shadow recipes; prefer hierarchy, border, and tonal contrast.

### Composition

- Remove card-within-card framing, double mattes, decorative labels, unnecessary eyebrows, badge noise, and dead whitespace.
- Do not add gradients, blobs, glows, floating pills, or excessive shadow to simulate polish.
- Hero imagery must occupy the intended stage confidently; do not place a small image inside a second decorative frame unless the reference system requires it.
- Every large empty area must have a compositional purpose.
- CTA hierarchy must read instantly; confirmed/saved states must not look disabled.
- Desktop and mobile may have independent IA but must visibly belong to one brand.

## B2C and Fashion-Specific Rules

- Use real or intentionally art-directed people and garments when human styling is the product value; generic silhouette SVGs are not an acceptable final substitute unless illustration is the explicit brand strategy.
- Verify head-to-foot or complete garment silhouette in the actual crop, not merely source orientation or metadata.
- Visually inspect each stock image; search-agent descriptions are not evidence.
- Keep imagery stylistically consistent across lighting, pose, crop, and commerce/editorial context.
- Store production imagery locally when stability matters and record source, photographer, license, and non-endorsement language.
- Do not imply pictured people endorse the service.

## Button and Control Rules

- All user-facing buttons derive from one primitive family or documented composed controls.
- Variants must be semantic and finite: primary, secondary, outline, ghost, selected, destructive only if needed.
- Do not create screen-specific button colors, radii, font weights, or shadows.
- Avoid default shadcn appearance; primitives must use product tokens.
- Selected text and secondary copy must meet readable contrast.
- A saved/confirmed CTA must read as intentional confirmation, not a disabled control.
- Verify keyboard focus, hover, active, selected, and disabled states.

## Reference-First Workflow

1. Collect platform-specific references at the real target viewport.
2. Save actual screenshot files, not URL lists.
3. Reject blocked, blank, popup-obscured, or structurally irrelevant captures.
4. Build a manifest with provenance and why each reference matters.
5. Extract reusable visual DNA: hierarchy, density, typography, image treatment, controls, and spacing.
6. Produce materially distinct concepts before implementation when redesign scope is broad.
7. Select by reference fidelity, product usefulness, fashion/brand credibility, and hierarchy—not novelty.
8. Do not preserve a rejected visual direction accidentally through old tokens or components.

## Repository Reinforcement Rule

Whenever the user says a rule must be applied strictly:

1. Add it to the product or Art Director contract.
2. Convert it into a lint, static check, test, or QA assertion when objectively detectable.
3. Add a screenshot/visual checklist item when it requires taste judgment.
4. Make the normal QA command run the new check.
5. Treat violations as completion blockers.

Recommended automated checks:

- raw hex values outside token definitions;
- arbitrary font sizes and numeric weights in route CSS;
- forbidden radius/shadow values;
- native `<button>` usage outside the primitive layer;
- console/page errors;
- missing local images or fonts;
- horizontal overflow;
- target viewport and persistence behavior.

Automated token compliance is necessary but not sufficient. A token-compliant screen may still be ugly.

## Required Visual QA Loop

1. Run lint, typecheck/build, regression tests, and design-system checks.
2. Render exact mobile and desktop target viewports.
3. Inspect screenshots for:
   - hierarchy and first action;
   - color restraint;
   - typography rhythm and Korean wrapping;
   - dead space and density;
   - image crop and consistency;
   - button quality and state clarity;
   - borders, pills, shadows, labels, and nested cards;
   - mobile/desktop brand continuity;
   - clipping and overflow.
4. Name hard failures explicitly.
5. Fix and recapture.
6. Repeat until no hard failure remains.
7. After deployment, verify hashed assets, local media/font responses, and remote QA—not only the hosting provider's success message.

## Hard Failures

Do not approve or deploy when any applies:

- generic admin/SaaS/template appearance for a consumer product;
- AI-slop palette soup or decorative premium styling;
- font declared but not actually loaded;
- arbitrary typography or control sizes without a scale;
- ad-hoc button styling outside the primitive system;
- selected/confirmed state with poor contrast or disabled appearance;
- meaningless large whitespace;
- image inside an accidental double frame;
- incomplete or misleading fashion crop;
- inconsistent stock-photo art direction;
- desktop and mobile look unrelated;
- screenshot QA missing;
- console errors, clipping, overflow, broken CTA, failed build, or stale production assets.

## Common Pitfalls

1. **Equating tokens with quality.** Tokens can formalize a bad visual direction. Always inspect screenshots.
2. **Equating shadcn with a design system.** Product tokens and finite variants are still required.
3. **Trusting source metadata for full-body imagery.** Verify pixels in the final crop.
4. **Using remote fonts without browser QA.** CDN 403s and fallback fonts can silently change composition.
5. **Adding more tokens to solve inconsistency.** Reduce families and recipes first.
6. **Passing overflow checks while typography looks broken.** Inspect Korean wrapping visually.
7. **Calling a saved CTA disabled-looking “subtle.”** State meaning must remain clear.
8. **Fixing one product only.** Promote repeated strict lessons into this Art Director skill and repository checks.

## Case References

- `references/artbilder-design-system-case-study.md` — condensed failure-and-correction pattern covering false-positive design-system claims, palette/size AI slop, local font verification, stock-image crop validation, confirmed CTA semantics, and the required static-check → screenshot-QA → production-verification loop.
- `references/ima2-kinetics-patterns.md` — combines ima2-gen's campaign/editorial image scale with Kinetics' palette, component, responsive, and state discipline; includes patterns to apply, motifs not to copy, and enforcement additions.

## Completion Report

Report:

- design-system stack actually used;
- color, typography, spacing, radius, shadow, and control scales;
- components migrated;
- automated prevention checks added;
- exact screenshots inspected;
- hard failures found and fixed;
- lint/build/test/design-check evidence;
- deployment and production-staleness evidence;
- remaining risks.

## Verification Checklist

- [ ] Real component primitive system is present
- [ ] Semantic tokens cover color/type/space/radius/shadow/controls
- [ ] Font asset loads successfully in browser
- [ ] No arbitrary route-level visual values outside documented exceptions
- [ ] Buttons use shared variants and all states are legible
- [ ] Mobile targets are at least 48px
- [ ] Palette has one intentional accent family
- [ ] Screens contain no accidental nested cards/double frames/dead voids
- [ ] Fashion imagery is complete, consistent, licensed, and locally stable
- [ ] Mobile and desktop share brand DNA
- [ ] Automated design-system check passes
- [ ] Lint/build/regression QA passes
- [ ] Exact viewport screenshots were inspected
- [ ] No hard visual failure remains
- [ ] Production serves the latest assets, images, and fonts
