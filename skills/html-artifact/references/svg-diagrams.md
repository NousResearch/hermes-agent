# SVG Diagrams

All diagrams are hand-authored inline `<svg>` — no Mermaid, no D3, no images. This
gives full control and keeps the file self-contained. Coordinates are computed by
hand, which makes the **visual verify loop mandatory** (see `fidelity-and-verify.md`):
the #1 failure is arrows landing in whitespace or boxes overlapping after an edit.

For light/educational diagrams use the 9-ramp design system in
`concept-archetypes.md`. For cloud/infra/system architecture use the dark token
variant in `dark-tech.md`. Both share the structural techniques below.

## Arrow markers

Define once in `<defs>`. Use `context-stroke` so the arrowhead inherits its line's
color (one marker serves every edge color):

```xml
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1 L8 5 L2 9" fill="none" stroke="context-stroke"
          stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>
```

Apply with `marker-end="url(#arrow)"`. When you need fixed per-semantic colors
(happy/fail/success) instead of inheritance, define matched markers `#arrow`,
`#arrow-rust`, `#arrow-olive` with hard-coded `fill`.

## Node groups

A node is a `<g>` wrapping a `<rect>` and centered `<text>`. Style via CSS classes,
not inline attributes — states live in the stylesheet:

```xml
<g class="node">
  <rect x="100" y="20" width="180" height="44" rx="8"/>
  <text class="th" x="190" y="42" text-anchor="middle" dominant-baseline="central">Service</text>
</g>
```

```css
.node rect { fill: var(--white); stroke: var(--gray-300); stroke-width: 1.5; }
.node.hot rect { fill: rgba(217,119,87,0.10); stroke: var(--clay); }   /* focus */
.node.ok  rect { fill: rgba(120,140,93,0.12); stroke: var(--olive); }  /* success */
.node.bad rect { fill: rgba(176,74,63,0.10);  stroke: var(--rust); }   /* error */
text { pointer-events: none; }   /* so clicks hit the node group, not the label */
```

Two-line node: add a second `<text class="ts">` for a subtitle, 18px below the title
baseline; make the rect 56px tall.

## Decision diamonds

Gates are a `<path>` diamond, not a rect:

```xml
<path class="gate" d="M310 262 L352 294 L310 326 L268 294 Z"/>
<text x="310" y="294" text-anchor="middle" dominant-baseline="central">valid?</text>
```

## Edges and semantics

Straight edges are `<line>`; branching/failure edges are Bézier `<path>` with
`fill="none"` (SVG paths default to `fill:black`). Encode meaning in style:

```css
.edge      { stroke: var(--gray-500); stroke-width: 1.5; fill: none; marker-end: url(#arrow); }
.edge.yes  { stroke: var(--olive); }                       /* happy path */
.edge.no   { stroke: var(--rust); stroke-dasharray: 4 4; } /* failure / dashed */
```

Label edges with a small mono `<text class="lbl">` near the midpoint ("pass",
"fail → 503", "retry").

## Coordinate-grid discipline

Hand-placed coordinates drift on edits. Keep them sane:

- **ViewBox**: `viewBox="0 0 W H"` where W is fixed (680 for educational, ~720–960
  for infra) and H = bottom of the last element + 40px buffer. Recompute H whenever
  you add rows.
- **Lanes / ranks**: put nodes on a regular grid. Pick a column x for each lane and a
  fixed row pitch (e.g. rows every 90px). Reuse the same x for every node in a lane so
  vertical edges are straight.
- **Gaps**: ≥60px between boxes; 10px between an arrowhead and the box it points at.
- **Wrap in scroll**: `.diagram { overflow-x: auto; } .diagram svg { min-width: 760px; }`
  so wide diagrams don't squish on mobile.
- **Width check**: a box must fit its text — `box_width >= chars * px_per_char + 48`.
  At 14px/weight-500 ≈ 8px/char; at 12px/weight-400 ≈ 6.5px/char.

## Interactive diagrams (optional)

To make a flowchart clickable with a synced detail panel, key each node with a
`data-k` attribute and look it up in a small JS dictionary. Always set a default-active
node on load so the panel is never empty, and keep the chart fully readable with JS off:

```js
const DETAIL = { ingest: { title: "Ingest", body: "…", code: "…" }, /* … */ };
document.querySelectorAll('.node[data-k]').forEach(n => {
  n.addEventListener('click', () => {
    document.querySelectorAll('.node.active').forEach(a => a.classList.remove('active'));
    n.classList.add('active');
    const d = DETAIL[n.dataset.k];
    panel.querySelector('.t').textContent = d.title;
    panel.querySelector('.b').innerHTML = d.body;
  });
});
document.querySelector('.node[data-k="ingest"]').click();  // default-active
```

## Exportable standalone SVG (optional)

If the user wants the SVG as its own downloadable file, the SVG must carry its own
`<defs><style>`, its own `<marker>`, a background `<rect fill="#FAF9F5">`, and
hard-coded hex (not `var()`, which won't resolve outside the host page). Then:

```js
const blob = new Blob([new XMLSerializer().serializeToString(svg)], {type:'image/svg+xml'});
const a = Object.assign(document.createElement('a'), {href: URL.createObjectURL(blob), download:'diagram.svg'});
a.click(); URL.revokeObjectURL(a.href);
```
