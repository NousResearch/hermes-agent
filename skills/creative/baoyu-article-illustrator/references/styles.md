# Style Reference

## Core Styles

Simplified style tier for quick selection:

| Core Style | Maps To | Best For |
|------------|---------|----------|
| `vector` | vector-illustration | Knowledge articles, tutorials, tech content |
| `minimal-flat` | notion | General, knowledge sharing, SaaS |
| `sci-fi` | blueprint | AI, frontier tech, system design |
| `hand-drawn` | sketch/warm | Relaxed, reflective, casual content |
| `editorial` | editorial | Processes, data, journalism |
| `scene` | warm/watercolor | Narratives, emotional, lifestyle |
| `poster` | screen-print | Opinion, editorial, cultural, cinematic |

Use Core Styles for most cases. See full Style Gallery below for granular control.

---

## Style Gallery

| Style | Description | Best For |
|-------|-------------|----------|
| `vector-illustration` | Clean flat vector art with bold shapes | Knowledge articles, tutorials, tech content |
| `notion` | Minimalist hand-drawn line art | Knowledge sharing, SaaS, productivity |
| `elegant` | Refined, sophisticated | Business, thought leadership |
| `warm` | Friendly, approachable | Personal growth, lifestyle, education |
| `minimal` | Ultra-clean, zen-like | Philosophy, minimalism, core concepts |
| `blueprint` | Technical schematics | Architecture, system design, engineering |
| `watercolor` | Soft artistic with natural warmth | Lifestyle, travel, creative |
| `editorial` | Magazine-style infographic | Tech explainers, journalism |
| `scientific` | Academic precise diagrams | Biology, chemistry, technical research |
| `chalkboard` | Classroom chalk drawing style | Education, teaching, explanations |
| `fantasy-animation` | Ghibli/Disney-inspired hand-drawn | Storybook, magical, emotional |
| `flat` | Modern bold geometric shapes | Modern digital, contemporary |
| `flat-doodle` | Cute flat with bold outlines | Cute, friendly, approachable |
| `intuition-machine` | Technical briefing with aged paper | Technical briefings, academic |
| `nature` | Organic earthy illustration | Environmental, wellness |
| `pixel-art` | Retro 8-bit gaming aesthetic | Gaming, retro tech |
| `playful` | Whimsical pastel doodles | Fun, casual, educational |
| `retro` | 80s/90s neon geometric | 80s/90s nostalgic, bold |
| `sketch` | Raw pencil notebook style | Brainstorming, creative exploration |
| `screen-print` | Bold poster art, halftone textures, limited colors | Opinion, editorial, cultural, cinematic |
| `sketch-notes` | Soft hand-drawn warm notes | Educational, warm notes |
| `ink-notes` | Black ink on pure white, sparse semantic accents, hand-lettered (à la Mike Rohde's sketchnoting) | Before/After essays, tech manifestos, framework analogies |
| `vintage` | Aged parchment historical | Historical, heritage |

Full specifications: `references/styles/<style>.md`

## Type × Style Compatibility Matrix

| | vector-illustration | notion | warm | minimal | blueprint | watercolor | elegant | editorial | scientific | screen-print |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| infographic | ✓✓ | ✓✓ | ✓ | ✓✓ | ✓✓ | ✓ | ✓✓ | ✓✓ | ✓✓ | ✓ |
| scene | ✓ | ✓ | ✓✓ | ✓ | ✗ | ✓✓ | ✓ | ✓ | ✗ | ✓✓ |
| flowchart | ✓✓ | ✓✓ | ✓ | ✓ | ✓✓ | ✗ | ✓ | ✓✓ | ✓ | ✗ |
| comparison | ✓✓ | ✓✓ | ✓ | ✓✓ | ✓ | ✓ | ✓✓ | ✓✓ | ✓ | ✓ |
| framework | ✓✓ | ✓✓ | ✓ | ✓✓ | ✓✓ | ✗ | ✓✓ | ✓ | ✓✓ | ✓ |
| timeline | ✓ | ✓✓ | ✓ | ✓ | ✓ | ✓✓ | ✓✓ | ✓✓ | ✓ | ✓ |

✓✓ = highly recommended | ✓ = compatible | ✗ = not recommended

## Auto Selection by Type

| Type | Primary Style | Secondary Styles |
|------|---------------|------------------|
| infographic | vector-illustration | notion, blueprint, editorial |
| scene | warm | watercolor, elegant |
| flowchart | vector-illustration | notion, blueprint |
| comparison | vector-illustration | notion, elegant |
| framework | blueprint | vector-illustration, notion |
| timeline | elegant | warm, editorial |

## Auto Selection by Content Signals

| Content Signals | Recommended Type | Recommended Style |
|-----------------|------------------|-------------------|
| API, metrics, data, comparison, numbers | infographic | blueprint, vector-illustration |
| Knowledge, concept, tutorial, learning, guide | infographic | vector-illustration, notion |
| Tech, AI, programming, development, code | infographic | vector-illustration, blueprint |
| How-to, steps, workflow, process, tutorial | flowchart | vector-illustration, notion |
| Framework, model, architecture, principles | framework | blueprint, vector-illustration |
| vs, pros/cons, before/after, alternatives | comparison | vector-illustration, notion |
| Manifesto, mindset shift, workforce, OS, whiteboard, professional visual note | comparison / framework | ink-notes |
| Story, emotion, journey, experience, personal | scene | warm, watercolor |
| History, timeline, progress, evolution | timeline | elegant, warm |
| Productivity, SaaS, tool, app, software | infographic | notion, vector-illustration |
| Business, professional, strategy, corporate | framework | elegant |
| Opinion, editorial, culture, philosophy, cinematic, dramatic, poster | scene | screen-print |
| Biology, chemistry, medical, scientific | infographic | scientific |
| Explainer, journalism, magazine, investigation | infographic | editorial |

## Style Characteristics by Type

### infographic + vector-illustration
- Clean flat vector shapes, bold geometric forms
- Vibrant but harmonious color palette
- Clear visual hierarchy with icons and labels
- Modern, professional, highly readable
- Perfect for knowledge articles and tutorials

### flowchart + vector-illustration
- Bold arrows and connectors
- Distinct step containers with icons
- Clean progression flow
- High contrast for readability

### comparison + vector-illustration
- Split layout with clear visual separation
- Bold iconography for each side
- Color-coded distinctions
- Easy at-a-glance comparison

### framework + vector-illustration
- Geometric node representations
- Clear hierarchical structure
- Bold connecting lines
- Modern system diagram aesthetic

### infographic + blueprint
- Technical precision, schematic lines
- Grid-based layout, clear zones
- Monospace labels, data-focused
- Blue/white color scheme

### infographic + notion
- Hand-drawn feel, approachable
- Soft icons, rounded elements
- Neutral palette, clean backgrounds
- Perfect for SaaS/productivity

### scene + warm
- Golden hour lighting, cozy atmosphere
- Soft gradients, natural textures
- Inviting, personal feeling
- Great for storytelling

### scene + watercolor
- Artistic, painterly effect
- Soft edges, color bleeding
- Dreamy, creative mood
- Best for lifestyle/travel

### flowchart + notion
- Clear step indicators
- Simple arrow connections
- Minimal decoration
- Focus on process clarity

### flowchart + blueprint
- Technical precision
- Detailed connection points
- Engineering aesthetic
- For complex systems

### comparison + elegant
- Refined dividers
- Balanced typography
- Professional appearance
- Business comparisons

### framework + blueprint
- Precise node connections
- Hierarchical clarity
- System architecture feel
- Technical frameworks

### timeline + elegant
- Sophisticated markers
- Refined typography
- Historical gravitas
- Professional presentations

### timeline + warm
- Friendly progression
- Organic flow
- Personal journey feel
- Growth narratives

### scene + screen-print
- Bold silhouettes, symbolic compositions
- 2-5 flat colors with halftone textures
- Figure-ground inversion (negative space tells secondary story)
- Vintage poster aesthetic, conceptual not literal
- Great for opinion pieces and cultural commentary

### comparison + screen-print
- Split duotone composition (one color per side)
- Bold geometric dividers
- Symbolic icons over detailed rendering
- High contrast, immediate visual impact

### framework + screen-print
- Geometric node representations with stencil-cut edges
- Limited color coding (one color per concept level)
- Clean silhouette-based iconography
- Poster-style hierarchy with bold typography

---

## Palette Gallery

Palettes override a style's default colors. Combine any style with any palette: `--style vector-illustration --palette macaron`.

| Palette | Description | Best For |
|---------|-------------|----------|
| `macaron` | Soft pastel blocks (blue, mint, lavender, peach) on warm cream | Educational, knowledge, tutorials |
| `warm` | Warm earth tones (orange, terracotta, gold) on soft peach, no cool colors | Brand, product, lifestyle |
| `neon` | Vibrant neon (pink, cyan, yellow) on dark purple | Gaming, retro, pop culture |
| `mono-ink` | Black ink on pure white with sparse semantic accents (coral red, muted teal, dusty lavender) | Professional visual notes, Before/After, manifestos |

Full specifications: `references/palettes/<palette>.md`

When no palette is specified, the style's built-in Color Palette is used.

## Palette Override Rules

1. Read style file → rendering rules (Visual Elements, Style Rules)
2. Read palette file → Colors + Background
3. Palette colors **replace** style's default Color Palette
4. Palette Background **replaces** style's default Background color
5. Style's texture description is preserved

