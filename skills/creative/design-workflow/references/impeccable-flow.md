# Impeccable-Style Flow

Use these named passes to make design work systematic. Pick the pass that matches the current problem instead of tweaking randomly.

## Core Passes

### teach

Collect project/product context. Use when the design would otherwise be generic.

Output:
- user
- job-to-be-done
- product promise
- constraints
- examples and anti-examples

### shape

Plan hierarchy before implementation.

Output:
- screen/page goal
- section order or screen zones
- CTA/action structure
- required states
- content/data requirements

### craft

Implement or specify the UI end-to-end with realistic content and states.

Rules:
- use existing components first
- do not introduce heavy libraries by default
- include responsive/native state thinking

### critique

Review the current UI honestly.

Look for:
- unclear value
- weak hierarchy
- generic AI patterns
- poor spacing/type/color
- missing states
- accessibility gaps
- platform-convention violations

### distill

Remove what does not support the primary screen job.

Common targets:
- duplicate feature cards
- generic copy
- decorative sections
- unnecessary secondary CTAs
- low-signal metrics

### polish

Final visual and interaction refinement after structure is right.

Do not polish before shape is correct.

## Targeted Repair Passes

### layout

Fix spacing, grouping, alignment, visual rhythm, responsive behavior, safe areas.

### typeset

Fix type scale, line length, hierarchy, label clarity, truncation, localization, Dynamic Type/text scaling.

### colorize

Fix semantic color, contrast, brand fit, dark/light behavior, state colors.

### clarify

Fix UX copy, labels, empty/error text, CTA wording, onboarding hints.

### bolder

Increase memorability: stronger headline, more distinctive layout, sharper visual metaphor, less generic composition.

Use carefully for dashboards and enterprise workflows.

### quieter

Reduce noise: fewer accents, calmer motion, lower contrast decorations, simpler grouping.

Use when UI feels salesy, chaotic, or fatiguing.

### harden

Make the UI survive reality:
- loading
- empty
- error
- long content
- permissions
- slow network
- offline/retry
- extreme values
- keyboard open on mobile

### adapt

Adapt across:
- desktop/tablet/mobile web
- iOS/Android conventions
- light/dark mode
- touch/mouse/keyboard
- localization

### optimize

Improve performance and implementation quality:
- image sizing
- bundle impact
- animation cost
- layout shift
- hydration risk
- unnecessary dependencies

## Stop Conditions

A pass is complete when it produces either:
- concrete edits,
- a prioritized fix list,
- or a clearly stated reason no change is needed.
