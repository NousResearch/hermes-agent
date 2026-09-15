# Structured Content Template

Template for generating structured infographic content that informs the visual designer.

## Purpose

This document bridges content analysis and visual design:
- Transforms source material into designer-ready format
- Organizes learning objectives into visual sections
- Preserves all source data verbatim
- Separates content from design instructions

## Instructional Design Process

### Phase 1: High-Level Outline

1. **Title**: Capture the essence in a compelling headline
2. **Overview**: Brief description (1-2 sentences)
3. **Learning Objectives**: List what the viewer will understand

### Phase 2: Section Development

For each learning objective:

1. **Key Concept**: One-sentence summary of the section
2. **Content**: Points extracted verbatim from source
3. **Visual Element**: What should be shown visually
4. **Text Labels**: Exact text for headlines, subheads, labels

### Phase 3: Data Integrity Check

Verify all source data is:
- Copied exactly (no paraphrasing)
- Attributed correctly (for quotes)
- Formatted consistently

## Critical Rules

| Rule | Requirement | Example |
|------|-------------|---------|
| **Output format** | Markdown only | Use proper headers, lists, code blocks |
| **Tone** | Expert trainer | Knowledgeable, clear, encouraging |
| **No new information** | Only source content | Don't add examples not in source |
| **Verbatim data** | Exact copies | "73% increase" not "significant increase" |

## Structured Content Format

```markdown
# [Infographic Title]

## Overview
[Brief description of what this infographic conveys - 1-2 sentences]

## Learning Objectives
The viewer will understand:
1. [Primary objective]
2. [Secondary objective]
3. [Tertiary objective if applicable]

---

## Section 1: [Section Title]

**Key Concept**: [One-sentence summary of this section]

**Content**:
- [Point 1 - verbatim from source]
- [Point 2 - verbatim from source]
- [Point 3 - verbatim from source]

**Visual Element**: [Description of what to show visually]
- Type: [icon/chart/illustration/diagram/photo]
- Subject: [what it depicts]
- Treatment: [how it should be presented]

**Text Labels**:
- Headline: "[Exact text for headline]"
- Subhead: "[Exact text for subhead]"
- Labels: "[Label 1]", "[Label 2]", "[Label 3]"

---

## Section 2: [Section Title]

**Key Concept**: [One-sentence summary]

**Content**:
- [Point 1]
- [Point 2]

**Visual Element**: [Description]

**Text Labels**:
- Headline: "[text]"
- Labels: "[Label 1]", "[Label 2]"

---

[Continue for each section...]

---

## Data Points (Verbatim)

All statistics, numbers, and quotes exactly as they appear in source:

### Statistics
- "[Exact statistic 1]"
- "[Exact statistic 2]"
- "[Exact statistic 3]"

### Quotes
- "[Exact quote]" — [Attribution]

### Key Terms
- **[Term 1]**: [Definition from source]
- **[Term 2]**: [Definition from source]

---

## Design Instructions

Extracted from user's steering prompt:

### Style Preferences
- [Any color preferences]
- [Any mood/aesthetic preferences]
- [Any artistic style preferences]

### Layout Preferences
- [Any structure preferences]
- [Any organization preferences]

### Other Requirements
- [Any other visual requirements from user]
- [Target platform if specified]
- [Brand guidelines if any]
```

## Section Types by Content

### For Process/Steps

```markdown
## Section N: Step N - [Step Title]

**Key Concept**: [What this step accomplishes]

**Content**:
- Action: [What to do]
- Details: [How to do it]
- Note: [Important consideration]

**Visual Element**:
- Type: numbered step icon
- Subject: [visual representing the action]
- Arrow: leads to next step

**Text Labels**:
- Headline: "Step N: [Title]"
- Action: "[Imperative verb + object]"
```

### For Comparison

```markdown
## Section N: [Item A] vs [Item B]

**Key Concept**: [What distinguishes them]

**Content**:
| Aspect | [Item A] | [Item B] |
|--------|----------|----------|
| [Factor 1] | [Value] | [Value] |
| [Factor 2] | [Value] | [Value] |

**Visual Element**:
- Type: split comparison
- Left: [Item A representation]
- Right: [Item B representation]

**Text Labels**:
- Headline: "[Item A] vs [Item B]"
- Left label: "[Item A name]"
- Right label: "[Item B name]"
```

### For Hierarchy

```markdown
## Section N: [Level Name]

**Key Concept**: [What this level represents]

**Content**:
- Position: [Top/Middle/Bottom]
- Priority: [Importance level]
- Contains: [Elements at this level]

**Visual Element**:
- Type: layer/tier
- Size: [relative to other levels]
- Position: [where in hierarchy]

**Text Labels**:
- Level title: "[Name]"
- Description: "[Brief description]"
```

### For Data/Statistics

```markdown
## Section N: [Metric Name]

**Key Concept**: [What this data shows]

**Content**:
- Value: [Exact number/percentage]
- Context: [What it means]
- Comparison: [Benchmark if any]

**Visual Element**:
- Type: [chart/number highlight/gauge]
- Emphasis: [how to draw attention]

**Text Labels**:
- Main number: "[Exact value]"
- Label: "[Metric name]"
- Context: "[Brief context]"
```

## Quality Checklist

Before finalizing structured content:

- [ ] Title captures the main message
- [ ] Learning objectives are clear and measurable
- [ ] Each section maps to an objective
- [ ] All content is verbatim from source
- [ ] Visual elements are clearly described
- [ ] Text labels are specified exactly
- [ ] Data points are collected and verified
- [ ] Design instructions are separated
- [ ] No new information has been added
