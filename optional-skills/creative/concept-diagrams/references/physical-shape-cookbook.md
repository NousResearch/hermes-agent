# Physical Shape Cookbook

Guidance for drawing physical objects (vehicles, buildings, hardware, mechanical systems, anatomy) — when rectangles aren't enough.

## Shape selection

| Physical form | SVG element | Example use |
|---------------|-------------|-------------|
| Curved bodies | `<path>` with Q/C curves | Fuselage, tanks, pipes |
| Tapered/angular shapes | `<polygon>` | Wings, fins, wedges |
| Cylindrical/round | `<ellipse>`, `<circle>` | Engines, wheels, buttons |
| Linear structures | `<line>` | Struts, beams, connections |
| Internal sections | `<rect>` inside parent | Compartments, rooms |
| Dashed boundaries | `stroke-dasharray` | Hidden parts, fuel tanks |

## Layering approach

1. Draw outer structure first (fuselage, frame, hull)
2. Add internal sections on top (cabins, compartments)
3. Add detail elements (engines, wheels, controls)
4. Add leader lines with labels

## Semantic CSS classes (instead of c-* ramps)

For physical diagrams, define component-specific classes directly rather than applying `c-*` color classes. This makes each part self-documenting and lets you keep a restrained palette:

```css
.fuselage { fill: #F1EFE8; stroke: #5F5E5A; stroke-width: 1; }
.wing     { fill: #E6F1FB; stroke: #185FA5; stroke-width: 1; }
.engine   { fill: #FAECE7; stroke: #993C1D; stroke-width: 1; }
```

Add these to a local `<style>` inside the SVG (or extend the host page's `<style>` block). The light-mode/dark-mode pattern still works — use the CSS variables from the template (`var(--bg-secondary)`, `var(--border)`, `var(--text-primary)`) if you want dark-mode awareness.

## Reference examples

Look at these example files for working physical-diagram patterns:

- `examples/commercial-aircraft-structure.md` — fuselage curves + tapered wings + ellipse engines
- `examples/wind-turbine-structure.md` — underground foundation, tubular tower, nacelle cutaway
- `examples/smartphone-layer-anatomy.md` — exploded-view stack with alternating labels
- `examples/apartment-floor-plan-conversion.md` — walls, doors, windows, proposed changes
