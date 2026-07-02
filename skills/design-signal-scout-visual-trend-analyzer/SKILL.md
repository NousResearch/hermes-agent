---
name: visual-trend-analyzer
description: >-
  Analyze images, video frames, or a set of visual references to identify form
  language, materials, color, composition, and repeated design motifs, then
  judge whether the pattern is isolated, emerging, established, or declining.
  Use this whenever the user shares images or reference sets and asks "what's
  the visual language here," "is this a trend," "break down the aesthetic of
  X," or wants a comparison of visual direction across multiple sources.
  Trigger proactively any time a scouting cycle surfaces recurring imagery in
  footwear, industrial/product design, 3D rendering, or architecture that
  needs a structured visual read rather than a one-line description.
---

# Visual Trend Analyzer

Read [../../references/operating-model.md](../../references/operating-model.md) first — this skill's confidence levels and application ideas feed into the shared signal record and `trend-clusterer`.

## Objective
Analyze images and video frames to identify visual patterns, form language, materials, color, composition, and repeated design motifs.

## Inputs
- image
- image set
- video frames
- source metadata
- associated caption or transcript
- comparison period
- related projects

## Analyze
Evaluate:

### Form
- silhouette
- proportion
- massing
- geometry
- curvature
- symmetry
- modularity
- repetition
- softness versus hardness
- organic versus technical character

### Surface and Material
- transparency
- translucency
- gloss
- roughness
- metallic behavior
- soft-touch appearance
- textiles
- mesh
- foam
- molded plastic
- rubber
- composite materials
- weathering
- texture density

### Color
- dominant palette
- accent palette
- saturation
- contrast
- temperature
- monochrome versus multicolor
- gradient use
- color blocking

### Graphics
- typography
- iconography
- pattern
- overlays
- decals
- branding density
- information hierarchy

### Composition
- framing
- camera angle
- depth
- negative space
- lighting
- focal hierarchy
- motion
- visual rhythm

### Production Language
- handcrafted
- mass-produced
- parametric
- generative
- machined
- molded
- inflated
- folded
- layered
- assembled
- printed
- scanned
- simulated

## Process
1. Describe what is visibly present.
2. Avoid guessing intent too early.
3. Extract visual attributes.
4. Compare against known references.
5. Identify repeated motifs.
6. Check whether repetition occurs across unrelated sources.
7. Estimate whether the pattern is emerging, established, declining, or isolated.
8. Suggest relevance to current projects.

## Trend Confidence
Use:

- `isolated`: one example
- `weak`: two related examples
- `emerging`: three or more independent examples
- `established`: repeated across domains
- `saturated`: widespread and heavily imitated
- `declining`: losing novelty or engagement

## Output
```markdown
# Visual Analysis: [Theme]

## Observed Characteristics
- ...

## Repeated Motifs
- ...

## Source Diversity
- ...

## Trend Stage
Isolated / Weak / Emerging / Established / Saturated / Declining

## Why It Matters
- ...

## Application Ideas
- footwear
- product design
- rendering
- content
- client work
- experimental study

## Confidence
Low / Medium / High
```

## Rules
- Separate visual observation from cultural interpretation.
- Never infer material from appearance without marking uncertainty.
- Do not claim a trend without source diversity.
- Do not reduce analysis to color palette extraction.
- Highlight contradictions and outliers.
