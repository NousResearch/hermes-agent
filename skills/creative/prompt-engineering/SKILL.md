---
name: prompt-engineering
description: "Use when the user asks for a prompt for any AI generation tool (image, video, music, voice), or says 'I need a prompt'. Covers Nano Banana Pro 2, Veo 3.1, Kling 3.0/2.6/O1, Z Image Turbo, Flux, Krea 1, Elevenlabs TTS, and SUNO music."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [creative, prompts, ai-generation, image, video, music, voice]
    related_skills: [songwriting-and-ai-music, heartmula, image_gen]
---

# Prompt Engineering for AI Generation

## Overview

Generate production-ready prompts for AI creative tools. Use when the user says "I need a prompt" or wants image, video, music, or voice generated. Think like a creative director, not a tag list.

## Tool Selection

| Goal | Tool | Goal | Tool |
|------|------|------|------|
| Best image | Nano Banana Pro 2 | Cinematic video | Veo 3.1 |
| Real-time image | Krea 1 | Character video | Kling 3.0 |
| Text in images | Flux / Z Image Turbo | Beat-synced video | Kling 2.6 |
| Speed | Z Image Turbo | UGC / product | Kling O1 |
| Full songs | SUNO | Expressive TTS | Elevenlabs |

## Universal Principles

1. **Specificity beats keywords** — Write like a creative director, not a tag list
2. **Front-load important elements** — Models weight early content heavily
3. **One change at a time** — Adjust single variables when iterating
4. **Motion needs verbs** — Video prompts require action, not static scenes
5. **Lock what stays** — Explicitly state preserved elements when editing

## Cross-Tool Pipelines

**Nano Banana → Kling (character video):** Generate 2-3 angle character sheet in Nano Banana, export at 1080p+, then Kling prompt describes **only motion** — never redescribe the subject. See [references/cross-tool.md](references/cross-tool.md).

## References

- [references/nano-banana.md](references/nano-banana.md) — Image (reasoning, edits, consistency)
- [references/veo.md](references/veo.md) — Video (cinematic, native audio, multi-shot)
- [references/kling.md](references/kling.md) — Video (3.0/2.6/O1, beat-matched)
- [references/flux.md](references/flux.md) — Image (text rendering, layered)
- [references/z-image.md](references/z-image.md) — Image (speed, bilingual)
- [references/krea.md](references/krea.md) — Image (real-time, artistic)
- [references/elevenlabs.md](references/elevenlabs.md) — Voice (expression brackets)
- [references/suno.md](references/suno.md) — Music (lyrics, style JSON)
- [references/cross-tool.md](references/cross-tool.md) — Pipelines and common pitfalls
