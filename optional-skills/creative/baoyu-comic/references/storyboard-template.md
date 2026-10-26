# Storyboard Template

## Storyboard Document Format

```markdown
---
title: "[Comic Title]"
topic: "[topic description]"
time_span: "[e.g., 1912-1954]"
narrative_approach: "[chronological/thematic/character-focused]"
recommended_style: "[style name]"
recommended_layout: "[layout name or varies]"
aspect_ratio: "3:4"    # 3:4 (portrait), 4:3 (landscape), 16:9 (widescreen)
language: "[zh/en/ja/etc.]"
page_count: [N]
generated: "YYYY-MM-DD HH:mm"
---

# [Comic Title] - Knowledge Comic Storyboard

**Character Reference**: characters/characters.png

---

## Cover

**Filename**: 00-cover-[slug].png
**Core Message**: [one-liner]

**Visual Design**:
- Title typography style
- Main visual composition
- Color scheme
- Subtitle / time span notation

**Visual Prompt**:
[Detailed image generation prompt]

---

## Page 1 / N

**Filename**: 01-page-[slug].png
**Layout**: [standard/cinematic/dense/splash/mixed]
**Narrative Layer**: [Main narrative / Narrator layer / Mixed]
**Core Message**: [What this page conveys]

### Panel Layout

**Panel Count**: X
**Layout Type**: [grid/irregular/splash]

#### Panel 1 (Size: 1/3 page, Position: Top)

**Scene**: [Time, location]
**Image Description**:
- Camera angle: [bird's eye / low angle / eye level / close-up / wide shot]
- Characters: [pose, expression, action]
- Environment: [scene details, period markers]
- Lighting: [atmosphere description]
- Color tone: [palette reference]

**Text Elements**:
- Dialogue bubble (oval): "Character line"
- Narrator box (rectangular): 「Narrator commentary」
- Caption bar: [Background info text]

#### Panel 2...

**Page Hook**: [Cliffhanger or transition at page end]

**Visual Prompt**:
[Full page image generation prompt]

---

## Page 2 / N
...
```

## Cover Design Principles

- Academic gravitas with visual appeal
- Title typography reflecting knowledge/science theme
- Composition hinting at core theme (character silhouette, iconic symbol, concept diagram)
- Subtitle or time span for epic scope

## Panel Composition Guidelines

| Panel Type | Recommended Count | Usage |
|-----------|-------------------|-------|
| Main narrative | 3-5 per page | Story progression |
| Concept diagram | 1-2 per page | Visualize abstractions |
| Narrator panel | 0-1 per page | Commentary, transition |
| Splash (full/half) | Occasional | Major moments |

## Panel Size Reference

- **Full page (Splash)**: Major moments, key breakthroughs
- **Half page**: Important scenes, turning points
- **1/3 page**: Standard narrative panels
- **1/4 or smaller**: Quick progression, sequential action

## Concept Visualization Techniques

Transform abstract concepts into concrete visuals:

| Abstract Concept | Visual Approach |
|-----------------|-----------------|
| Neural network | Glowing nodes with connecting lines |
| Gradient descent | Ball rolling down valley terrain |
| Data flow | Luminous particles flowing through pipes |
| Algorithm iteration | Ascending spiral staircase |
| Breakthrough moment | Shattering barrier, piercing light |
| Logical proof | Building blocks assembling |
| Uncertainty | Forking paths, fog, multiple shadows |

## Text Element Design

| Text Type | Style | Usage |
|-----------|-------|-------|
| Character dialogue | Oval speech bubble | Main narrative speech |
| Narrator commentary | Rectangular box | Explanation, commentary |
| Caption bar | Edge-mounted rectangle | Time, location info |
| Thought bubble | Cloud shape | Character inner monologue |
| Term label | Bold / special color | First appearance of technical terms |

## Prompt Structure for Consistency

Each page prompt should include character reference:

```
[CHARACTER REFERENCE]
(Key details from characters.md for characters in this page)

[PAGE CONTENT]
(Specific scene, panel layout, and visual elements)

[CONSISTENCY REMINDER]
Maintain exact character appearances as defined in character reference.
- [Character A]: [key identifying features]
- [Character B]: [key identifying features]
```
