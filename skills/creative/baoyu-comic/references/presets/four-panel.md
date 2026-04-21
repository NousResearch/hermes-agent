# four-panel

四格漫画预设 - Minimalist four-panel business allegory comics

## Base Configuration

| Dimension | Value |
|-----------|-------|
| Art Style | minimalist |
| Tone | neutral |
| Layout | four-panel (default) |
| Aspect | 4:3 (landscape) |

Equivalent to: `--art minimalist --tone neutral --layout four-panel --aspect 4:3`

## Unique Rules

This preset includes special rules beyond the art+tone combination. When `--style four-panel` is used, ALL rules below must be applied.

### 起承转合 Narrative Structure (CRITICAL)

Every comic MUST follow the four-panel 起承转合 structure:

| Panel | Role | Requirements |
|-------|------|-------------|
| 1 (起 Setup) | Introduce the situation | Show character(s) in a recognizable context. Establish the "normal" state or problem |
| 2 (承 Development) | Build on the setup | Add complication, show an attempt, or introduce the concept. Stakes become clearer |
| 3 (转 Turn) | The twist or key insight | **Most important panel.** Show the unexpected reversal, contrast, or "aha" moment that makes the allegory work |
| 4 (合 Conclusion) | Resolution and takeaway | Show the result, consequence, or lesson learned. Can be a visual punchline or summary |

**CRITICAL**: Do NOT deviate from exactly 4 panels. No 5th panel, no title panel, no footer panel within the image.

### Single-Page Story Rule (CRITICAL)

- The entire story is told in ONE page with exactly 4 panels
- Page count: always 1 (plus optional cover)
- No multi-page four-panel stories — if content requires more, create multiple separate four-panel comics
- Storyboard structure: Cover (optional) + 1 page

### Accent Color System

- The image is primarily black-and-white line art
- Use exactly 1-2 spot colors per strip (default: orange `#FF6B35`)
- Rules:
  - Key concept label or object: filled with accent color or outlined in accent
  - Panel 3 (转 Turn) should have the strongest color emphasis
  - Characters remain B&W — color is for concepts/objects/labels only
  - Consistent accent color across all 4 panels (do not switch colors between panels)

### Character Design Rules

- Simplified stick-figure-like characters
- Distinguish characters through simple props: ties, glasses, hats, briefcases, aprons
- No detailed faces — dot eyes, line mouth at most
- Characters should be generic enough to represent archetypes (the manager, the employee, the customer)
- Maximum 2-3 characters per strip

### Text in Panels

- Chinese text for dialogue and labels (or match source language)
- Keep text minimal — 1-2 short lines per panel maximum
- Key concept terms can be highlighted with accent color background
- No narrator boxes — dialogue and labels only
- Speech bubbles: simple rectangles or ovals, thin black outline

### Optional Title & Caption

- A brief descriptive title above the 4 panels
- An optional one-line caption/moral below the panels
- These are part of the page composition, not separate panels

### Character Archetypes (Flexible)

Create simple stick-figure characters based on content. No fixed defaults:

| Role | Archetype | Visual Cues |
|------|-----------|------------|
| Protagonist | Worker/employee facing a situation | Simple figure, minimal distinguishing feature (glasses, tie) |
| Authority | Boss/manager/expert | Slightly larger figure, or prop like pointer/clipboard |
| Object | The concept itself | Labeled object, icon, or highlighted text with accent color |

### Prompt Template

When generating image prompts for four-panel comics, include these keywords:

> A minimalist, clean line art digital comic strip in a four-panel grid layout (2×2). The style is simplified cartoon illustration with clear black outlines and a minimal color palette of black, white, and specific spot [accent color] for key concepts.

Each panel description should specify:
- Panel position (Top Left / Top Right / Bottom Left / Bottom Right)
- Character poses and gestures (simple, stick-figure style)
- Dialogue text in Chinese (hand-drawn style)
- Any accent-colored elements (concept labels, key objects)

## Quality Markers

- ✓ Exactly 4 panels in strict 2×2 grid
- ✓ 起承转合 narrative arc clearly present
- ✓ 90%+ black-and-white with strategic spot color
- ✓ Simplified stick-figure characters
- ✓ Key concept visually highlighted with accent color
- ✓ Text is minimal and in Chinese (or source language)
- ✓ Single complete story in one page
- ✓ Panel 3 delivers a clear "turn" or insight

## Best For

Business allegory, management fables, short insights, workplace parables, concept contrasts, social media educational content, quick-read comics
