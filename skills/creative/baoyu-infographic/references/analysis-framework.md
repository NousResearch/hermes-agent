# Infographic Content Analysis Framework

Deep analysis framework applying instructional design principles to infographic creation.

## Purpose

Before creating an infographic, thoroughly analyze the source material to:
- Understand the content at a deep level
- Identify clear learning objectives for the viewer
- Structure information for maximum clarity and retention
- Match content to optimal layout×style combinations
- Preserve all source data verbatim

## Instructional Design Mindset

Approach content analysis as a **world-class instructional designer**:

| Principle | Application |
|-----------|-------------|
| **Deep Understanding** | Read the entire document before analyzing any part |
| **Learner-Centered** | Focus on what the viewer needs to understand |
| **Visual Storytelling** | Use visuals to communicate, not just decorate |
| **Cognitive Load** | Simplify complex ideas without losing accuracy |
| **Data Integrity** | Never alter, summarize, or paraphrase source facts |

## Analysis Dimensions

### 1. Content Type Classification

| Type | Characteristics | Best Layout | Best Style |
|------|-----------------|-------------|------------|
| **Timeline/History** | Sequential events, dates, progression | linear-progression | craft-handmade, aged-academia |
| **Process/Tutorial** | Step-by-step instructions, how-to | linear-progression, winding-roadmap | ikea-manual, technical-schematic |
| **Comparison** | A vs B, pros/cons, before-after | binary-comparison, comparison-matrix | corporate-memphis, bold-graphic |
| **Hierarchy** | Levels, priorities, pyramids | hierarchical-layers, tree-branching | craft-handmade, corporate-memphis |
| **Relationships** | Connections, overlaps, influences | venn-diagram, hub-spoke, jigsaw | craft-handmade, subway-map |
| **Data/Metrics** | Statistics, KPIs, measurements | dashboard, periodic-table | corporate-memphis, technical-schematic |
| **Cycle/Loop** | Recurring processes, feedback loops | circular-flow | craft-handmade, technical-schematic |
| **System/Structure** | Components, architecture, anatomy | structural-breakdown, bento-grid | technical-schematic, ikea-manual |
| **Journey/Narrative** | Stories, user flows, milestones | winding-roadmap, story-mountain | storybook-watercolor, comic-strip |
| **Overview/Summary** | Multiple topics, feature highlights | bento-grid, periodic-table, dense-modules | chalkboard, bold-graphic |
| **Product/Buying Guide** | Multi-dimension comparisons, specs, pitfalls | dense-modules | morandi-journal, pop-laboratory, retro-pop-grid |

### 2. Learning Objective Identification

Every infographic should have 1-3 clear learning objectives.

**Good Learning Objectives**:
- Specific and measurable
- Focus on what the viewer will understand, not just see
- Written from the viewer's perspective

**Format**: "After viewing this infographic, the viewer will understand..."

| Content Aspect | Objective Type |
|----------------|----------------|
| Core concept | "...what [topic] is and why it matters" |
| Process | "...how to [accomplish something]" |
| Comparison | "...the key differences between [A] and [B]" |
| Relationships | "...how [elements] connect to each other" |
| Data | "...the significance of [key statistics]" |

### 3. Audience Analysis

| Factor | Questions | Impact |
|--------|-----------|--------|
| **Knowledge Level** | What do they already know? | Determines complexity depth |
| **Context** | Why are they viewing this? | Determines emphasis points |
| **Expectations** | What do they hope to learn? | Determines success criteria |
| **Visual Preferences** | Professional, playful, technical? | Influences style choice |

### 4. Complexity Assessment

| Level | Indicators | Layout Recommendation |
|-------|------------|----------------------|
| **Simple** (3-5 points) | Few main concepts, clear relationships | sparse layouts, single focus |
| **Moderate** (6-8 points) | Multiple concepts, some relationships | balanced layouts, clear sections |
| **Complex** (9+ points) | Many concepts, intricate relationships | dense layouts, multiple sections |

### 5. Visual Opportunity Mapping

Identify what can be shown rather than told:

| Content Element | Visual Treatment |
|-----------------|------------------|
| Numbers/Statistics | Large, highlighted numerals |
| Comparisons | Side-by-side, split screen |
| Processes | Arrows, numbered steps, flow |
| Hierarchies | Pyramids, layers, size differences |
| Relationships | Lines, connections, overlapping shapes |
| Categories | Color coding, grouping, sections |
| Timelines | Horizontal/vertical progression |
| Quotes | Callout boxes, quotation marks |

### 6. Data Verbatim Extraction

**Critical**: All factual information must be preserved exactly as written in the source.

| Data Type | Handling Rule |
|-----------|---------------|
| **Statistics** | Copy exactly: "73%" not "about 70%" |
| **Quotes** | Copy word-for-word with attribution |
| **Names** | Preserve exact spelling |
| **Dates** | Keep original format |
| **Technical Terms** | Do not simplify or substitute |
| **Lists** | Preserve order and wording |

**Never**:
- Round numbers
- Paraphrase quotes
- Substitute simpler words
- Add implied information
- Remove context that affects meaning

## Output Format

Save analysis results to `analysis.md`:

```yaml
---
title: "[Main topic title]"
topic: "[educational/technical/business/creative/etc.]"
data_type: "[timeline/hierarchy/comparison/process/etc.]"
complexity: "[simple/moderate/complex]"
point_count: [number of main points]
source_language: "[detected language]"
user_language: "[user's language]"
---

## Main Topic
[1-2 sentence summary of what this content is about]

## Learning Objectives
After viewing this infographic, the viewer should understand:
1. [Primary objective]
2. [Secondary objective]
3. [Tertiary objective if applicable]

## Target Audience
- **Knowledge Level**: [Beginner/Intermediate/Expert]
- **Context**: [Why they're viewing this]
- **Expectations**: [What they hope to learn]

## Content Type Analysis
- **Data Structure**: [How information relates to itself]
- **Key Relationships**: [What connects to what]
- **Visual Opportunities**: [What can be shown rather than told]

## Key Data Points (Verbatim)
[All statistics, quotes, and critical facts exactly as they appear in source]
- "[Exact data point 1]"
- "[Exact data point 2]"
- "[Exact quote with attribution]"

## Layout × Style Signals
- Content type: [type] → suggests [layout]
- Tone: [tone] → suggests [style]
- Audience: [audience] → suggests [style]
- Complexity: [level] → suggests [layout density]

## Design Instructions (from user input)
[Any style, color, layout, or visual preferences extracted from user's steering prompt]

## Recommended Combinations
1. **[Layout] + [Style]** (Recommended): [Brief rationale]
2. **[Layout] + [Style]**: [Brief rationale]
3. **[Layout] + [Style]**: [Brief rationale]
```

## Analysis Checklist

Before proceeding to structured content generation:

- [ ] Have I read the entire source document?
- [ ] Can I summarize the main topic in 1-2 sentences?
- [ ] Have I identified 1-3 clear learning objectives?
- [ ] Do I understand the target audience?
- [ ] Have I classified the content type correctly?
- [ ] Have I extracted all data points verbatim?
- [ ] Have I identified visual opportunities?
- [ ] Have I extracted design instructions from user input?
- [ ] Have I recommended 3 layout×style combinations?
