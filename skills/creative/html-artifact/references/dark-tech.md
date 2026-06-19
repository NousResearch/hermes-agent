# Dark-Tech Diagram Variant

The dark "infra" aesthetic for cloud / software / system architecture diagrams —
slate-950 background, a faint grid, neon-ish category strokes. Carried over from the
former `architecture-diagram` skill (based on Cocoon AI's generator, MIT). Use this
when the subject is infrastructure or a software system; use the light 9-ramp system
in `concept-archetypes.md` for educational/physical subjects.

Read `svg-diagrams.md` for the shared structural techniques (markers, node groups,
coordinate discipline).

> **Self-contained adaptation:** the original loaded JetBrains Mono from Google Fonts.
> This skill forbids external fonts — use the OS-native `--mono` stack instead. The
> dark look is otherwise unchanged.

## Background

Slate-950 page with a subtle 40px grid:

```css
body { background: #020617; color: #e2e8f0; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
.diagram-card { background: #0b1220; border: 1px solid #1e293b; border-radius: 14px; padding: 20px; }
```

```xml
<defs>
  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" stroke-width="0.5"/>
  </pattern>
</defs>
<rect width="100%" height="100%" fill="url(#grid)"/>
```

## Semantic component palette

Fill is a translucent tint; stroke is the saturated category color:

| Component type | Fill (rgba) | Stroke (hex) |
|---|---|---|
| Frontend | `rgba(8,51,68,0.4)` | `#22d3ee` cyan |
| Backend | `rgba(6,78,59,0.4)` | `#34d399` emerald |
| Database | `rgba(76,29,149,0.4)` | `#a78bfa` violet |
| AWS / Cloud | `rgba(120,53,15,0.3)` | `#fbbf24` amber |
| Security | `rgba(136,19,55,0.4)` | `#fb7185` rose |
| Message bus | `rgba(251,146,60,0.3)` | `#fb923c` orange |
| External | `rgba(30,41,59,0.5)` | `#94a3b8` slate |

Type sizes: 12px names, 9px sublabels, 8px annotations, 7px tiny labels.

## Component rendering — double-rect mask

Semi-transparent fills let arrows show through. Mask each component with an opaque
backing rect, then the styled rect on top:

```xml
<rect x="100" y="80" width="160" height="60" rx="6" fill="#0f172a"/>                       <!-- opaque backing -->
<rect x="100" y="80" width="160" height="60" rx="6" fill="rgba(6,78,59,0.4)" stroke="#34d399" stroke-width="1.5"/>
<text x="180" y="114" text-anchor="middle" fill="#e2e8f0" font-size="12">API server</text>
```

Components are `rx="6"`, 1.5px strokes. Standard service height 60px; large components
80–120px; ≥40px vertical gap.

## Connections & boundaries

- **Z-order**: draw arrows *early* (right after the grid) so component boxes render on
  top of them.
- **Security flows**: dashed rose lines (`stroke-dasharray="4 4"`, `#fb7185`).
- **Security group boundary**: dashed `4 4`, rose, `rx="8"`.
- **Region boundary**: large dash `8 4`, amber, `rx="12"`.
- **Message buses** go *in the gap* between services, never overlapping them.
- **Legend** (critical): place it *outside* every boundary box — compute the lowest
  boundary Y and put the legend ≥20px below it.

## Document structure

Four parts: (1) header with a pulsing dot + subtitle, (2) the SVG in a rounded border
card, (3) a grid of summary info-cards below, (4) minimal footer. Pulsing dot is pure
CSS (`@keyframes`), no JS.

Info-card pattern:

```html
<div class="card">
  <div class="card-header"><span class="card-dot cyan"></span><h3>Title</h3></div>
  <ul><li>Item one</li><li>Item two</li></ul>
</div>
```

Pure CSS for any animation (pulsing dots) — no JavaScript. The dual-mode
`templates/diagram.html` includes this dark CSS alongside the light educational CSS;
add `class="dark"` (or use the dark `<style>` block) for infra diagrams.
