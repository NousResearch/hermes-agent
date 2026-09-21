# dark-cyberpunk-hud

Dev-tool system monitor / terminal HUD aesthetic. Reference: the four "ChatGPT
Image" samples in ~/Reference Images/ — dark cyberpunk poster with structured
technical data (CPU/RAM/UPTIME stats, git commands, system monitors).

## Design Aesthetic

Dense technical HUD on near-black canvas. Multiple structured panels, monospace
data labels, neon accent glow. Feels like a developer tool's live dashboard or
a security operations centre monitor. The image communicates a system running
with measurable state, not a marketing graphic.

## Background

- Color: Near-Black (#0A0A0F) or Deep Navy-Black (#0B0E1A)
- Texture: Subtle scanline or grid overlay, very low contrast

## Color Palette

| Role | Color | Hex | Usage |
|------|-------|-----|-------|
| Background | Near-Black | #0A0A0F | Primary background |
| Panel | Charcoal Navy | #13161F | Panel/card backgrounds |
| Border | Cool Gray | #2A2D3A | Panel borders, dividers |
| Primary Text | Off-White | #E5E7EB | Main labels |
| Secondary Text | Cool Gray | #94A3B8 | Captions, units |
| Accent 1 (Primary) | Cyan | #22D3EE | Highlighted data, callouts |
| Accent 2 | Electric Magenta | #F472B6 | Secondary data, alerts |
| Accent 3 | Lime | #A3E635 | Positive metrics, success |
| Accent 4 | Amber | #FBBF24 | Warnings, attention |
| Mono Code | Soft Cyan | #67E8F9 | Code snippets, commands |

## Visual Elements

- Monospace font for all data labels
- Rectangular panel/box structure with thin borders
- Key-value pairs left-aligned, value right-aligned
- Status indicators (dots, bars) on key metrics
- Section headers in small caps with bracket markers: `[ CPU ]` `[ RAM ]`
- Arrow markers for flows: `>` `>>` `->`
- Numbered list markers for steps
- Subtle grid or scanline texture in background
- Low-saturation except for accent data

## Style Rules

### Do

- Use monospace font for all numeric and code data
- Group related data into labelled panels
- Use neon accent sparingly to highlight key values only
- Show measurable state (numbers, percentages, status)
- Keep layout structured, never organic or flowing

### Don't

- Use photographic imagery or organic textures
- Add any decorative imagery unrelated to the technical content
- Use more than 4 accent colours
- Render full sentences — prefer short labels and key-value pairs
- Add watermarks, branding text, or caption boxes

## Text Discipline

- Labels: 1-3 words max (`CPU`, `UPTIME`, `STATUS`, `PIPELINE`)
- Values: numbers, percentages, or short code (`99.9%`, `1024`, `> 50K+ LINES`)
- Headers: short phrases in brackets or all-caps (`[ SYSTEM MONITOR ]`)
- Avoid full sentences, paragraphs, or marketing copy

## Best For

Build-in-public posts, dev-tool launches, system architecture reveals,
performance metrics posts, security/ops dashboards, command-line tool
announcements, technical concept explainers with measurable state.
