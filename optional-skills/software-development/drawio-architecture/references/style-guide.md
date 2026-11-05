# Style Guide — palette, typography, effects, legend

Apply these when no user style preset is active. When a diagram uses 3+ semantic colors, auto-generate a legend (procedure at the bottom).

## Color palette (coordinated fill/stroke pairs)

| Role | fillColor | strokeColor |
|------|-----------|-------------|
| Service / client | `#dae8fc` | `#6c8ebf` |
| Success / database | `#d5e8d4` | `#82b366` |
| Queue / decision | `#fff2cc` | `#d6b656` |
| Gateway / API | `#ffe6cc` | `#d79b00` |
| Error / alert | `#f8cecc` | `#b85450` |
| External / neutral | `#f5f5f5` | `#666666` |
| Security / auth | `#e1d5e5` | `#9673a6` |

Stick to **2–3 soft colors** per diagram; leave 20–40px padding inside containers.

## Typography

| Property | Values | Example |
|----------|--------|---------|
| `fontSize` | number | `fontSize=12` |
| `fontStyle` | 0 normal, 1 bold, 2 italic, 3 bold+italic, 4 underline (bitwise OR) | `fontStyle=1` |
| `align` | left, center, right | `align=center` |
| `verticalAlign` | top, middle, bottom | `verticalAlign=middle` |

## Borders & effects

| Property | Values | Example |
|----------|--------|---------|
| `rounded` | 0 or 1 | `rounded=1` |
| `strokeWidth` | number | `strokeWidth=2` |
| `dashed` | 0 or 1 | `dashed=1` |
| `dashPattern` | pattern | `dashPattern=8 8` |
| `shadow` | 0 or 1 | `shadow=1` |

Consistent semantics: pick **one** meaning for `dashed=` (e.g. optional / async / inferred) and keep it across the diagram; add a legend entry if mixing styles.

## Shape size defaults

| Shape | Size |
|-------|------|
| Rectangle | `140×60` |
| Diamond | `140×80` |
| Circle | `60×60` |
| Document | `120×80` |
| Cylinder | `100×70` |

## HTML labels

Always include `html=1`. Use `&#xa;` for line breaks (works with or without `html=1`) or `&lt;br&gt;` (needs `html=1`). Escape `<` `>` `&` `"` in values. Example:
```xml
<mxCell id="t" value="&lt;b&gt;Title&lt;/b&gt;&lt;br&gt;Description" style="rounded=1;whiteSpace=wrap;html=1;" vertex="1" parent="1">
  <mxGeometry x="100" y="100" width="160" height="60" as="geometry"/>
</mxCell>
```

## Auto-generating a legend

When 3+ roles appear, generate a legend container mechanically from the roles actually present — never invent entries that aren't in the diagram.

```xml
<mxCell id="legend" value="Legend" style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#666666;verticalAlign=top;fontStyle=1;" vertex="1" parent="1">
  <mxGeometry x="40" y="720" width="200" height="30" as="geometry"/>
</mxCell>
<mxCell id="leg1" value="" style="rounded=0;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="legend">
  <mxGeometry x="10" y="30" width="30" height="16" as="geometry"/>
</mxCell>
<mxCell id="leg1t" value="Service" style="text;html=1;align=left;verticalAlign=middle;" vertex="1" parent="legend">
  <mxGeometry x="50" y="28" width="140" height="20" as="geometry"/>
</mxCell>
```

Rules:
- Swatch fill/stroke from the active palette; label = the **role name** (Service, Database, Queue…).
- One swatch+label pair per used role; 24px row pitch; swatches/labels are children of `legend` (relative coords).
- Container height = `30 + 24 × rows`.
- Skip the legend entirely for single-color diagrams.

## Dark mode

Set `adaptiveColors="auto"` on `<mxGraphModel>`. `strokeColor/fillColor/fontColor="default"` auto-adapt (black→white). Explicit colors are inverted automatically; use `light-dark(light,dark)` only when the inverse is wrong (e.g. `fontColor=light-dark(#7EA6E0,#FF0000)`).
