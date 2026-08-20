# Spacing & Adaptivity

Space between controls, margins against the viewport, hinting at off-screen content, and layouts that survive resizing and translation.

## Breathing Room Between Targets

Controls placed too close together get mis-tapped and read as one unit. When the project has no established density scale, use these starting points:

| Between | Starting point |
| --- | --- |
| Adjacent bordered/filled controls (buttons, inputs) | `12px` |
| Around borderless controls (text buttons, icon buttons) | `24px` |
| Unrelated control groups | `24px`+ (2× the intra-group gap) |

Borderless controls usually need more clearance because nothing marks where one target ends and the next begins; the space itself is the boundary. Compact professional tools may use less when the hit areas remain distinct and do not overlap. Preserve an established, usable density instead of expanding controls solely to match these values.

```html
<!-- Good: bordered buttons at 12px, icon buttons given room -->
<div class="flex gap-3">
  <button class="rounded-lg border px-4 py-2">Cancel</button>
  <button class="rounded-lg bg-blue-600 px-4 py-2 text-white">Save</button>
</div>

<!-- Bad: three borderless icon buttons packed at 4px -->
<div class="flex gap-1">
  <button><TrashIcon /></button>
  <button><ArchiveIcon /></button>
  <button><ShareIcon /></button>
</div>
```

WCAG target-size requirements, larger usability targets, and pseudo-element expansion are covered by the `better-accessibility` skill; these clearances are in addition, so expanded hit areas never overlap.

## Inset Buttons from the Edges

In content layouts, buttons pressed accidentally against the viewport can look like system chrome and clip against curved corners or gesture zones. Keep them inside the layout margins. Edge-to-edge actions remain valid when they intentionally are application/platform chrome and account for safe areas:

```css
/* Good: inset action bar */
.action-bar {
  padding-inline: 16px;
  padding-bottom: calc(16px + env(safe-area-inset-bottom));
}
.action-bar button { width: 100%; border-radius: 12px; }

/* Bad: button glued to all three edges */
.action-bar button {
  width: 100vw;
  border-radius: 0;
  position: fixed;
  bottom: 0;
}
```

Start near `16px` inline margin on mobile when the project has no layout token; the button can still span the full content width inside those margins.

## Progressive Disclosure Needs an Affordance

Hiding complexity is good; hiding it without a cue is a trap. Every piece of off-screen or collapsed content needs a visible hint that it exists. Preserve the product's established scroll indicator or disclosure pattern; use the recipes below when no clear cue exists:

- **Peeking items.** In a horizontal scroller or carousel, size items so the next one peeks `16–32px` past the container edge. A row of cards that ends exactly at the edge looks complete, and nobody scrolls it.
- **Disclosure controls.** Collapsed sections get a chevron or "Show more" control; the label states what's hidden ("Show 12 more results"), not just "More".
- **Truncation cues.** Clamped text shows an ellipsis and a way to expand; see `better-typography` for truncation mechanics.

The peeking-scroller recipe: the container's padding creates the peek, and snap points stay on the content edge.

```css
.scroller {
  display: flex;
  gap: 12px;
  overflow-x: auto;
  padding-inline: 24px;
  scroll-padding-inline: 24px;
  scroll-snap-type: x mandatory;
}
.scroller > * {
  flex: 0 0 calc(100% - 48px - 24px); /* container minus margins minus peek */
  scroll-snap-align: start;
}
```

```html
<!-- Tailwind: the 80% width keeps the next card's leading 16-32px visible -->
<div class="flex gap-3 overflow-x-auto px-6 [scroll-padding-inline:1.5rem] snap-x snap-mandatory">
  <div class="w-[80%] shrink-0 snap-start">…</div>
  <div class="w-[80%] shrink-0 snap-start">…</div>
</div>
```

## Content Bleeds, Controls Float

The two layers behave differently at the edges:

- **Content layer**: backgrounds, hero media, and scrollable lists extend to the viewport edges.
- **Control layer**: text and controls stay inside the layout margins and safe areas, floating above the content.

```css
/* Good: full-bleed media inside a constrained article */
.article {
  display: grid;
  grid-template-columns: 1fr min(65ch, calc(100% - 48px)) 1fr;
}
.article > * { grid-column: 2; }
.article > .full-bleed { grid-column: 1 / -1; }
```

Sticky headers and floating action buttons account for safe areas:

```css
.fab {
  position: fixed;
  inset-inline-end: calc(16px + env(safe-area-inset-right));
  bottom: calc(16px + env(safe-area-inset-bottom));
}
```

## Hold Structure Until It Breaks

Breakpoints belong to the content, not the device catalog:

- Break where the layout actually stops fitting (when the sidebar squeezes the content below its minimum measure, when the card grid drops below a usable column width), not at `768px` because a preset says so.
- Collapse late. A layout that keeps its expanded structure as long as it genuinely fits stays stable and familiar; premature collapsing throws away space users paid for.
- Prefer **container queries** for components: a card should adapt to the column it's in, not to the viewport.

```css
/* Good: component adapts to its container */
.card-list { container-type: inline-size; }
@container (max-width: 400px) {
  .card { grid-template-columns: 1fr; }
}

/* Bad: viewport media query breaks the card inside a narrow sidebar */
@media (max-width: 768px) {
  .card { grid-template-columns: 1fr; }
}
```

Test order: the smallest supported size and the largest first (those break first), then the sizes in between.

## Plan for Growth and Clipping

Layouts fail in two directions: content grows, and viewports shrink.

**String expansion varies substantially by language and source-string length.** Do not rely on one universal percentage. Rules:

- No fixed widths sized to English labels; use `max-width` plus wrapping.
- No fixed heights on text containers; use `min-height` if a floor is needed.
- Buttons size themselves from their label (`padding-inline`), never a hardcoded width.
- Test with pseudo-localization or a long-string locale before shipping.

```css
/* Good: label defines the size */
.button { padding-inline: 16px; white-space: nowrap; }

/* Bad: German will overflow or truncate */
.button { width: 96px; overflow: hidden; }
```

**Clipping:** never park critical actions where they can be cut off: the bottom edge of a resizable pane, below the fold of a fixed-height modal, behind an expanding keyboard. Keep primary actions in stable chrome: a sticky footer with safe-area padding, or the top of the view. If a modal's content scrolls, its action row doesn't.
