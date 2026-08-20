# Grouping & Alignment

How spacing, shapes, shared edges, and ordering communicate what belongs together and what matters most.

## Group with Space, Not Lines

Three tools create grouping, in order of preference:

1. **Negative space**: the default. Related items sit close; unrelated items sit far apart.
2. **Background shapes**: a card or filled container, when a group needs to read as one unit (a selectable row, a draggable card).
3. **Separator lines**: last resort, for dense data where space would cost too much (tables, long settings lists).

The structural rule: the gap between groups must be at least 2× the gap within a group. If items inside a group are `8px` apart, groups need `16px`+ between them, otherwise the eye can't tell where one group ends.

```css
/* Good: spacing alone communicates the grouping */
.field-group { display: flex; flex-direction: column; gap: 8px; }
.form { display: flex; flex-direction: column; gap: 24px; }

/* Bad: uniform spacing plus lines to compensate */
.form > * { margin-bottom: 12px; border-bottom: 1px solid var(--separator); }
```

```html
<!-- Good: Tailwind -->
<div class="space-y-6">
  <div class="space-y-2">…field group…</div>
  <div class="space-y-2">…field group…</div>
</div>
```

When a separator is genuinely needed, keep it quiet: hairline width, low contrast, and never combined with a large gap (the gap already did the job).

## Keep Controls Distinct from Content

Interactive elements need a visual signal that they're interactive: a background, a border, an underline, or placement in a consistent control zone (toolbar, footer row). A control styled identically to static text is invisible.

```html
<!-- Bad: action looks exactly like the description text next to it -->
<p class="text-zinc-600">Your trial ends soon. Upgrade now</p>

<!-- Good: the action reads as an action -->
<p class="text-zinc-600">Your trial ends soon.</p>
<button class="font-medium text-blue-600">Upgrade now</button>
```

The inverse also holds: don't give static elements control styling. A non-clickable badge shaped exactly like the buttons beside it collects dead clicks.

## Align to Shared Edges

Pick a small set of alignment edges and put everything on them; the eye tracks straight edges to scan content.

- Every stray edge (an icon 2px off the text edge, a card padded differently from its neighbor) reads as noise even when nobody can name the problem.
- Use one consistent project spacing step to express hierarchy; `16px` is a useful default when no scale exists, and deeper nesting repeats the same step.
- Numbers in tables right-align to the trailing edge (see `better-typography` for tabular figures); text left-aligns to the leading edge.

```css
/* Good: one shared leading edge, one indent step */
.section { padding-inline: 24px; }
.section .child { margin-inline-start: 16px; }

/* Bad: three unrelated leading edges in one column */
.header { padding-inline-start: 20px; }
.list-item { padding-inline-start: 14px; }
.footer { padding-inline-start: 24px; }
```

## Logical Properties, Not Physical

Express direction-dependent horizontal position as leading/trailing so the layout mirrors automatically under `dir="rtl"`:

| Physical (avoid) | Logical (use) |
| --- | --- |
| `margin-left` | `margin-inline-start` |
| `padding-right` | `padding-inline-end` |
| `left: 0` | `inset-inline-start: 0` |
| `text-align: left` | `text-align: start` |
| `border-right` | `border-inline-end` |

```html
<!-- Good: Tailwind logical utilities -->
<div class="ms-4 pe-6 text-start">…</div>

<!-- Bad: breaks in RTL -->
<div class="ml-4 pr-6 text-left">…</div>
```

Reserve physical properties for things that genuinely refer to physical screen sides regardless of language, e.g. positioning relative to a device notch, or an element that must match a physical gesture direction.

When the arrangement of elements encodes progression (star ratings, step indicators, progress bars), the sequence mirrors in RTL: stars fill from the trailing side. Flexbox and grid with logical properties mirror automatically; hand-positioned elements don't. Digit order inside numbers never reverses; that and other bidi text rules live in the `better-typography` skill.

## Order by Importance

Readers scan top-to-bottom and leading-to-trailing. Place content accordingly:

- The most important information sits near the top and the leading edge; the further down and trailing something sits, the less attention it gets.
- Give essential information room. Don't bury the one number the user came for under rows of secondary detail; push secondary content into collapsed sections, tabs, or detail views.
- Within a row, the identifying content (name, title) leads; metadata and actions trail.

```html
<!-- Good: primary fact first, detail demoted -->
<div>
  <p class="text-2xl font-semibold">$4,320.00</p>
  <p class="text-sm text-zinc-500">Available balance</p>
</div>

<!-- Bad: the key fact is buried below the fold of the card -->
<div>
  <p class="text-sm">Account 4402 · Opened 2019 · Standard tier</p>
  <p class="text-sm">Last statement: June 30</p>
  <p class="text-sm">Balance: $4,320.00</p>
</div>
```

Think in **leading/trailing**, not left/right: combined with logical properties, the same hierarchy mirrors correctly in RTL locales.

## Don't Overload the Entry Point

The first screenful is a table of contents, not the whole book. If everything is prominent, nothing is:

- One primary action per view (see `better-colors` for how color enforces this).
- Group secondary actions behind a menu once they exceed two or three.
- Prefer a short view that links deeper over a long view that shows everything at level one.
