---
name: signal-scout
description: >-
  Run a scouting cycle to discover high-value design and technology signals
  across footwear, industrial/product design, 3D and visualization, CAD,
  Blender, architecture, game development, and AI-assisted creative tooling.
  Use this whenever the user asks to scout, search for, or round up what's
  happening in design/tech right now; wants emerging visual language, new
  tools, or workflow shifts; or is starting a scouting cycle for content or
  product opportunities. Trigger proactively for asks like "what's new in
  [domain]," "find me signals on X," or "scan for anything interesting in
  footwear/3D/Blender/creative AI this week" — this is the entry point most
  other skills in this plugin (visual-trend-analyzer, technology-radar,
  pain-point-miner) feed into.
---

# Design Signal Scout

Read [../../references/operating-model.md](../../references/operating-model.md) first for shared signal categories, the required signal record, and scoring dimensions.

## Objective
Discover high-value design and technology signals relevant to footwear, product design, 3D, visualization, architecture, games, and creative tools.

## Trigger
Use this skill when:
- running a scheduled scouting cycle
- researching a new design direction
- looking for emerging visual language
- tracking tools or workflows
- searching for content or product opportunities

## Search Themes
Monitor:

### Design
- footwear design
- industrial design
- product design
- CMF
- materials
- manufacturing
- design systems
- form language
- speculative design
- digital fashion
- architecture
- interiors
- transportation
- packaging

### 3D and Visualization
- Blender
- procedural modeling
- geometry nodes
- CAD
- Plasticity
- Rhino
- Unreal Engine
- Substance
- rendering
- motion design
- photogrammetry
- Gaussian splatting
- neural rendering
- 3D generation

### Technology
- creative AI
- multimodal models
- image generation
- video generation
- agentic workflows
- automation
- design software
- open-source tools
- creative coding
- digital fabrication

## Process
1. Generate broad and narrow searches.
2. Search primary and specialist sources.
3. Collect candidate signals.
4. Deduplicate.
5. Verify material claims.
6. Classify.
7. Score.
8. Route to another skill if deeper analysis is needed — `visual-trend-analyzer`
   for imagery/motifs, `technology-radar` for tools, `pain-point-miner` for
   community complaints, `trend-clusterer` once several related signals exist.

## Discovery Queries
Use query patterns such as:

```text
new workflow for [domain]
recent open source [tool category]
designers discussing [pain point]
emerging material in [industry]
new Blender add-on for [task]
site:github.com [workflow]
site:reddit.com [pain point]
"how do I" [design task]
"wish there was" [design tool]
"takes forever" [workflow]
"new release" [software]
```

## Signal Threshold
Surface a candidate only when at least one is true:
- directly relevant to current work
- repeated across multiple independent sources
- unusually novel
- commercially actionable
- likely to become content
- likely to save meaningful time
- likely to create a product or service opportunity

## Scoring
Calculate:

```text
signal_score =
relevance
+ novelty
+ evidence
+ momentum
+ actionability
+ commercial_value
+ creative_value
- duplication_penalty
- hype_penalty
- access_risk
```

Use a 0–40 range.

- `0–12`: archive
- `13–20`: monitor
- `21–28`: include in digest
- `29–34`: recommend action
- `35–40`: immediate alert

## Output
For each selected signal:

```markdown
## [Signal]

**Category:**
**Source:**
**Observed:**
**Interpretation:**
**Why it matters:**
**Evidence strength:**
**Momentum:**
**Potential use:**
**Recommended action:**
**Score:**
```

## Rules
- Do not call something a trend based on one source.
- Do not summarize obvious mainstream news unless it changes strategy.
- Prefer original projects, repositories, papers, demos, and creator posts.
- Surface counter-signals when evidence conflicts.
- Mark speculative conclusions clearly.
