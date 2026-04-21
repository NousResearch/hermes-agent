---
name: first-time-setup
description: First-time setup flow for baoyu-comic preferences
---

# First-Time Setup

## Overview

When no EXTEND.md is found, guide user through preference setup.

**⛔ BLOCKING OPERATION**: This setup MUST complete before ANY other workflow steps. Do NOT:
- Ask about content/source material
- Ask about art style or tone
- Ask about layout preferences
- Proceed to content analysis

ONLY ask the questions in this setup flow, save EXTEND.md, then continue.

## Setup Flow

```
No EXTEND.md found
        │
        ▼
┌─────────────────────┐
│ AskUserQuestion     │
│ (all questions)     │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Create EXTEND.md    │
└─────────────────────┘
        │
        ▼
    Continue to Step 1
```

## Questions

**Language**: Use user's input language or preferred language for all questions. Do not always use English.

Use single AskUserQuestion with multiple questions (AskUserQuestion auto-adds "Other" option):

### Question 1: Watermark

```
header: "Watermark"
question: "Watermark text for generated comic pages? Type your watermark content (e.g., name, @handle)"
options:
  - label: "No watermark (Recommended)"
    description: "No watermark, can enable later in EXTEND.md"
```

Position defaults to bottom-right.

### Question 2: Preferred Art Style

```
header: "Art"
question: "Default art style preference? Or type another style name"
options:
  - label: "Auto-select (Recommended)"
    description: "Auto-select based on content analysis"
  - label: "ligne-claire"
    description: "Uniform lines, flat colors, European comic (Tintin style)"
  - label: "manga"
    description: "Japanese manga style, expressive eyes and emotions"
  - label: "realistic"
    description: "Digital painting, sophisticated and professional"
```

### Question 3: Preferred Tone

```
header: "Tone"
question: "Default tone/mood preference?"
options:
  - label: "Auto-select (Recommended)"
    description: "Auto-select based on content signals"
  - label: "neutral"
    description: "Balanced, rational, educational"
  - label: "warm"
    description: "Nostalgic, personal, comforting"
  - label: "dramatic"
    description: "High contrast, intense, powerful"
```

### Question 4: Language

```
header: "Language"
question: "Output language for comic text?"
options:
  - label: "Auto-detect (Recommended)"
    description: "Match source content language"
  - label: "zh"
    description: "Chinese (中文)"
  - label: "en"
    description: "English"
```

### Question 5: Save Location

```
header: "Save"
question: "Where to save preferences?"
options:
  - label: "Project"
    description: ".baoyu-skills/ (this project only)"
  - label: "User"
    description: "~/.baoyu-skills/ (all projects)"
```

## Save Locations

| Choice | Path | Scope |
|--------|------|-------|
| Project | `.baoyu-skills/baoyu-comic/EXTEND.md` | Current project |
| User | `~/.baoyu-skills/baoyu-comic/EXTEND.md` | All projects |

## After Setup

1. Create directory if needed
2. Write EXTEND.md with frontmatter
3. Confirm: "Preferences saved to [path]"
4. Continue to Step 1

## EXTEND.md Template

```yaml
---
version: 2
watermark:
  enabled: [true/false]
  content: "[user input or empty]"
  position: bottom-right
  opacity: 0.5
preferred_art: [selected art style or null]
preferred_tone: [selected tone or null]
preferred_layout: null
preferred_aspect: null
language: [selected or null]
character_presets: []
---
```

## Modifying Preferences Later

Users can edit EXTEND.md directly or run setup again:
- Delete EXTEND.md to trigger setup
- Edit YAML frontmatter for quick changes
- Full schema: `config/preferences-schema.md`
