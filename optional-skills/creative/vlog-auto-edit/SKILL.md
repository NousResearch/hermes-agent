---
name: vlog-auto-edit
description: Auto-edit travel vlogs from raw clips, upstream-maintained.
version: 1.1.0
author: nyx研究所 (znyupup)
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [video, video-editing, vlog, ffmpeg, whisper, automation]
    category: creative
    related_skills: [ai-presenter-video, kanban-video-orchestrator]
    upstream:
      repo: znyupup/ai-video-editing-skill
      path: .
---

# Vlog Auto Edit (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [znyupup/ai-video-editing-skill](https://github.com/znyupup/ai-video-editing-skill),
> which keeps its `SKILL.md`, `scripts/`, `templates/` and `examples/` at the
> repo root. `hermes skills install official/creative/vlog-auto-edit` pulls the
> current tree live from that repo (quarantined and scanned like any hub
> install). This directory holds only the catalog metadata, so it never goes
> stale.

Turns a folder of phone-shot travel clips into a finished vlog. The agent
inventories the footage, transcribes and scores each clip (speech, volume,
sampled frames), trims recording cues, repeated takes and shaky starts, then
drafts a three-act edit plan the user reviews in a browser dashboard before
ffmpeg renders the cut with section titles, a highlight intro and optional
BGM. It is written in Chinese and tuned for Chinese-language narration.

## Prerequisites

- `ffmpeg` on `PATH`, and Python 3.9+ with `openai-whisper` (or FunASR) and
  `Pillow`. The skill checks for these before installing anything.
- An OpenAI-compatible vision model for frame analysis (the upstream
  recommends Zhipu GLM-4.6V-Flash, which is free). In Hermes, frames can go
  through `vision_analyze` instead of a separate API key.
- Optional: an AI music provider (MiniMax, Suno) for generated BGM.

Linux and macOS only: the upstream workflow writes intermediates to the system temp directory
and uses platform font paths for title overlays.

Full documentation: https://github.com/znyupup/ai-video-editing-skill#readme
