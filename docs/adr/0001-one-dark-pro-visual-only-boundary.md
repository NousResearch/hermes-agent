# ADR 0001: One Dark Pro is a visual-only skin — the skin/behavior boundary

- Status: Accepted
- Date: 2026-11-26
- Scope: Hermes desktop themes (apps/desktop/src/themes), contribution policy for theme PRs

## Context

The One Dark Pro desktop theme (preset in `apps/desktop/src/themes/one-dark.ts`,
scoped styles in `one-dark.css`, plus the integration glue in `skin.ts`,
`persist-accent.ts`, `user-themes.ts` and `context.tsx`) was visually approved by
the Maintainer: background `#282c34`, accent `#61afef`, borders `#181a1f`.

An earlier iteration of the contribution (superseded PR #159253, closed
2026-11-23) also carried fixes that changed renderer *behavior*: terminal zoom
unscale corrections, terminal output anchoring, scrollback preservation on
transport retry, and boot-gap stripping. Those are quality fixes, but they are
not theme work. Bundling them into a theme PR makes the PR impossible to review
as a skin, couples cosmetic adoption to functional risk, and forces maintainers
to re-validate terminal behavior when they only asked for a palette.

## Decision

One Dark Pro is exclusively a **visual layer**. The boundary between the skin
and upstream Hermes behavior is:

### The theme owns (customizable)

- Palette: backgrounds, foregrounds, borders/separators, contrast levels.
- Colors of text, selection, accent, and visual states (hover/focus/active
  *colors*, not their mechanics).
- Appearance-only surface styling expressed through the existing theme token
  system (`oneDarkProperties`, scoped CSS under
  `:root[data-hermes-palette='one-dark']`) or the terminal palette.

### Upstream Hermes owns (must be preserved unchanged by a theme)

- Component structure and composition.
- Navigation, menus, sidebar and its toggle, and overall layout.
- Terminal behavior: scrollback, retries, output anchoring, zoom handling.
- Functional accessibility and the original UX (hit areas, keyboard
  interaction, focus order, motion mechanics).

### The boundary rule

A skin may change **what the app looks like**; it may not change **what the app
does** or **how it is arranged**. If a change requires touching component
structure, element geometry, event handling, or terminal/zoom logic to achieve
its look, it is out of scope for a theme PR — even when the motivation is
visual.

## Criteria

### Accepting a visual improvement in a theme PR

- It expresses itself through palette values, theme tokens, or scoped CSS.
- It preserves upstream geometry, positioning, structure and interaction
  mechanics (hover/focus/active behavior included).
- It degrades to the upstream look when the palette marker is removed.
- It is testable as a color/token assertion, not as a behavior assertion.

### Rejecting a functional change inside a theme PR

- It changes what a component renders structurally or how it behaves.
- It modifies terminal session handling, zoom math, or layout logic.
- Its tests assert behavior instead of tokens/colors.
- It would need a behavior-level review independent of the palette.

Functional improvements discovered while theming are welcome — as **separate,
independent contributions** against upstream, with their own justification and
review. The zoom/terminal corrections removed from this contribution were split
out for exactly that purpose.

## Consequences

- Theme PRs review as diffs of colors and tokens; functional review is not
  required and upstream behavior diffs should be treated as red flags.
- Upstream may evolve component structure, zoom handling, and terminal behavior
  freely; the theme rebases as a thin palette layer with no semantic conflicts.
- The One Dark Pro preset stays compatible with future theme-system work
  (persist-accent, retint, user-themes) because it only feeds the existing
  token contract.
- Future visual improvements to One Dark Pro remain allowed and expected, as
  long as they respect this separation.
