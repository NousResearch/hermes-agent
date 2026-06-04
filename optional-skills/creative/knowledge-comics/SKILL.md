---
name: knowledge-comics
description: "Baoyu-based knowledge visualization: knowledge comics (知识漫画) and infographics (信息图)."
version: 1.56.1
author: 宝玉 (JimLiu)
license: MIT
metadata:
  hermes:
    tags: [knowledge-comics, infographic, knowledge-comic, visual-summary, creative, image-generation]
    homepage: https://github.com/JimLiu/baoyu-skills
---

# Knowledge Comics & Infographics

Adapted from [baoyu-skills](https://github.com/JimLiu/baoyu-skills) for Hermes Agent's tool ecosystem.

This umbrella skill consolidates two baoyu-based visual knowledge generation tools:

| Tool | Focus | Best For |
|------|-------|----------|
| **baoyu-comic** | Multi-page sequential narrative with character consistency | Educational comics, biographies, tutorials, "知识漫画" |
| **baoyu-infographic** | Single-image information design with layout × style grids | Visual summaries, data-dense guides, "信息图", "可视化" |

## When to Use

- **Knowledge comic / 知识漫画**: Trigger [baoyu-comic](references/baoyu-comic.md) — multi-page comics with art styles, tones, layouts, character sheets, and storyboarding
- **Infographic / 信息图 / 可视化**: Trigger [baoyu-infographic](references/baoyu-infographic.md) — single-image infographics with 21 layouts × 21 styles
- **Both**: If the request is ambiguous, describe both options and ask the user to choose

## Tool Comparison

| Dimension | baoyu-comic | baoyu-infographic |
|-----------|-------------|-------------------|
| Output | Multi-page comic series | Single infographic image |
| Workflow steps | 8-step (analyze → storyboard → prompts → images) | 7-step (analyze → structure → prompt → image) |
| Character support | Yes (character sheet + embedded descriptions) | No |
| Layout options | 7 layouts | 21 layouts |
| Art/style options | 6 art styles × 7 tones + 5 presets | 21 styles |
| Image generation | `image_generate` (prompt-only, returns URL → download) | `image_generate` (prompt-only, returns URL → download) |
| Review gates | Steps 2, 4, 6 are conditional on user preference | Steps 3, 4 are confirm/clarify steps |
| Output dir | `comic/{topic-slug}/` | `infographic/{topic-slug}/` |

## Quick Reference

### baoyu-comic — Knowledge Comic Creator

Trigger: educational comic, biography comic, tutorial comic, "知识漫画", "教育漫画", Logicomix-style

**Options**: Art (6) × Tone (7) × Layout (7) × Aspect (3) + 5 presets + reference images

**Workflow**: Input → Analyze → [Confirm style] → Storyboard → [Review?] → Prompts → [Review?] → Images (character sheet + pages) → Complete

**Key pitfall**: Always use **absolute paths** for `curl -o`. `image_generate` returns a URL; you must download it. Character consistency comes from text descriptions embedded in page prompts.

**Details**: [references/baoyu-comic.md](references/baoyu-comic.md)

### baoyu-infographic — Infographic Generator

Trigger: infographic, visual summary, "信息图", "可视化", "高密度信息大图"

**Options**: Layout (21) × Style (21) × Aspect (named + custom) + keyword shortcuts

**Workflow**: Input → Analyze → Structured content → [Recommend combos] → Confirm → Prompt → Image → Summary

**Key pitfall**: Preserve source data **verbatim**. Never summarize statistics. Map custom aspect ratios to nearest named option.

**Details**: [references/baoyu-infographic.md](references/baoyu-infographic.md)

## References

- [baoyu-comic](references/baoyu-comic.md) — Full knowledge comic skill
- [baoyu-infographic](references/baoyu-infographic.md) — Full infographic skill
