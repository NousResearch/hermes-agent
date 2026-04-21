# Comic Content Analysis Framework

Deep analysis framework for transforming source content into effective visual storytelling.

## Purpose

Before creating a comic, thoroughly analyze the source material to:
- Identify the target audience and their needs
- Determine what value the comic will deliver
- Extract narrative potential for visual storytelling
- Plan character arcs and key moments

## Analysis Dimensions

### 1. Core Content (Understanding "What")

**Central Message**
- What is the single most important idea readers should take away?
- Can you express it in one sentence?

**Key Concepts**
- What are the essential concepts readers must understand?
- How should these concepts be visualized?
- Which concepts need simplified explanations?

**Content Structure**
- How is the source material organized?
- What is the natural narrative arc?
- Where are the climax and turning points?

**Evidence & Examples**
- What concrete examples, data, or stories support the main ideas?
- Which examples translate well to visual panels?
- What can be shown rather than told?

### 2. Context & Background (Understanding "Why")

**Source Origin**
- Who created this content? What is their perspective?
- What was the original purpose?
- Is there bias to be aware of?

**Historical/Cultural Context**
- When and where does the story take place?
- What background knowledge do readers need?
- What period-specific visual elements are required?

**Underlying Assumptions**
- What does the source assume readers already know?
- What implicit beliefs or values are present?
- Should the comic challenge or reinforce these?

### 3. Audience Analysis

**Primary Audience**
- Who will read this comic?
- What is their existing knowledge level?
- What are their interests and motivations?

**Secondary Audiences**
- Who else might benefit from this comic?
- How might their needs differ?

**Reader Questions**
- What questions will readers have?
- What misconceptions might they bring?
- What "aha moments" can we create?

### 4. Value Proposition

**Knowledge Value**
- What will readers learn?
- What new perspectives will they gain?
- How will this change their understanding?

**Emotional Value**
- What emotions should readers feel?
- What connections will they make with characters?
- What will make this memorable?

**Practical Value**
- Can readers apply what they learn?
- What actions might this inspire?
- What conversations might it spark?

### 5. Narrative Potential

**Story Arc Candidates**
- What natural narratives exist in the content?
- Where is the conflict or tension?
- What transformations occur?

**Character Potential**
- Who are the key figures?
- What are their motivations and obstacles?
- How do they change throughout?

**Visual Opportunities**
- What scenes have strong visual potential?
- Where can abstract concepts become concrete images?
- What metaphors can be visualized?

**Dramatic Moments**
- What are the breakthrough/revelation moments?
- Where are the emotional peaks?
- What creates tension and release?

### 6. Adaptation Considerations

**What to Keep**
- Essential facts and ideas
- Key quotes or moments
- Core emotional beats

**What to Simplify**
- Complex explanations
- Dense technical details
- Lengthy descriptions

**What to Expand**
- Brief mentions that deserve more attention
- Implied emotions or relationships
- Visual details not in source

**What to Omit**
- Tangential information
- Redundant examples
- Content that doesn't serve the narrative

## Output Format

Analysis results should be saved to `analysis.md` with:

1. **YAML Front Matter**: Metadata (title, topic, time_span, source_language, user_language, aspect_ratio, recommended_page_count, recommended_art, recommended_tone, recommended_layout)
2. **Target Audience**: Primary, secondary, tertiary audiences with their needs
3. **Value Proposition**: What readers will gain (knowledge, emotional, practical)
4. **Core Themes**: Table with theme, narrative potential, visual opportunity
5. **Key Figures & Story Arcs**: Character profiles with arcs, visual identity, key moments
6. **Content Signals**: Style and layout recommendations based on content type
7. **Recommended Approaches**: Narrative approaches ranked by suitability

### YAML Front Matter Example

```yaml
---
title: "Alan Turing: The Father of Computing"
topic: alan-turing-biography
time_span: 1912-1954
source_language: en
user_language: zh  # From EXTEND.md or detected
aspect_ratio: "3:4"
recommended_page_count: 16
recommended_art: ligne-claire  # ligne-claire|manga|realistic|ink-brush|chalk
recommended_tone: neutral      # neutral|warm|dramatic|romantic|energetic|vintage|action
recommended_layout: mixed      # standard|cinematic|dense|splash|mixed|webtoon
---
```

### Language Fields

| Field | Description |
|-------|-------------|
| `source_language` | Detected language of source content |
| `user_language` | Output language for comic (from EXTEND.md > --lang > source_language) |

## Analysis Checklist

Before proceeding to storyboard:

- [ ] Can I state the core message in one sentence?
- [ ] Do I know exactly who will read this comic?
- [ ] Have I identified at least 3 ways this comic provides value?
- [ ] Are there clear protagonists with compelling arcs?
- [ ] Have I found at least 5 visually powerful moments?
- [ ] Do I understand what to keep, simplify, expand, and omit?
- [ ] Have I identified the emotional peaks and valleys?
