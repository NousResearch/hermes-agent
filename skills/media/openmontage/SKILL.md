---
name: openmontage
description: "Use when installing, running, or directing OpenMontage: an open-source agentic video production system that turns AI coding assistants into end-to-end video studios with pipelines, tools, skills, Remotion/HyperFrames, FFmpeg, and quality gates."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [OpenMontage, Video-Production, Remotion, FFmpeg, AI-Video, Agentic-Workflows]
    related_skills: [youtube-content, manim-video, comfyui, songwriting-and-ai-music, codex]
---

# OpenMontage

## Overview

[OpenMontage](https://github.com/calesthio/OpenMontage) is an open-source, agentic video production system. It turns a coding assistant into the orchestrator for research, scripting, asset generation, editing, composition, subtitles, audio, and post-render review.

Canonical upstream:

- Repo: `https://github.com/calesthio/OpenMontage`
- Agent guide: `AGENT_GUIDE.md`
- Project context: `PROJECT_CONTEXT.md`
- Providers guide: `docs/PROVIDERS.md`
- Review guide: `docs/PR_REVIEW_GUIDE.md`

OpenMontage is built around readable pipeline manifests, stage-director skills, Python tools, and renderers such as Remotion, HyperFrames, and FFmpeg. Treat it as a production workflow, not a single prompt-to-video API.

## When to Use

- The user explicitly mentions OpenMontage or wants an agentic video production stack.
- The task is to create or evaluate explainer videos, documentary montages, cinematic trailers, shorts, podcast clips, talking-head edits, local video workflows, or production pipelines through OpenMontage.
- The user wants to start from a reference video and derive a grounded production plan.
- The user asks for free/open video workflows using real footage, Archive.org, NASA, Wikimedia, FFmpeg, Remotion, or local TTS.

Do not use this skill for a quick one-off image generation or simple video summary unless the user specifically wants OpenMontage.

## Prerequisites

| Requirement | Baseline |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ for most workflows; Node 22+ recommended for HyperFrames |
| FFmpeg | Required for post-production and encoding |
| AI coding assistant | Claude Code, Cursor, Copilot, Windsurf, Codex, or Hermes with terminal/file tools |
| Optional GPU | Required only for local video-generation models |

Install FFmpeg first:

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg
```

## Install

Preferred setup:

```bash
git clone https://github.com/calesthio/OpenMontage.git
cd OpenMontage
make setup
```

Manual setup if `make` is unavailable:

```bash
pip install -r requirements.txt
cd remotion-composer
npm install
cd ..
pip install piper-tts
cp .env.example .env
```

Windows npm pitfall from upstream docs:

```bash
# If npm install fails with ERR_INVALID_ARG_TYPE
npx --yes npm install
```

## First Steps for an Agent

OpenMontage is intentionally agent-operated. Before doing production work in the repo:

1. Read `AGENT_GUIDE.md`.
2. Read `PROJECT_CONTEXT.md`.
3. Inspect the capability envelope and provider menu:

```bash
python -c "from tools.tool_registry import registry; import json; registry.discover(); print(json.dumps(registry.support_envelope(), indent=2))"
python -c "from tools.tool_registry import registry; import json; registry.discover(); print(json.dumps(registry.provider_menu(), indent=2))"
```

4. Treat every request as a pipeline-selection problem.
5. Read the selected pipeline manifest under `pipeline_defs/`.
6. Read the matching stage-director skills under `skills/pipelines/`.
7. Execute with checkpoints and review gates; do not improvise around the pipeline.

## Request Patterns

### Prompt from Scratch

```text
Make a 60-second animated explainer about how neural networks learn.
```

### Real-Footage Documentary Path

```text
Make a 75-second documentary montage about city life in the rain. Use real footage only, no narration, elegiac tone, with music.
```

### Reference-Driven Production

```text
Here's a YouTube Short I love. Make something like this, but about quantum computing.
```

For reference-driven work, inspect the source video first. Identify pacing, hook structure, scene rhythm, caption style, visual grammar, and audio treatment before proposing variants.

## Pipeline Selection

OpenMontage pipelines are complete production workflows. Start by selecting the right one, then follow its manifest.

| Pipeline | Best For |
|---|---|
| `animated-explainer` | Educational explainers, tutorials, concept breakdowns |
| `animation` | Motion graphics, kinetic typography, abstract sequences |
| `avatar-spokesperson` | Presenter/avatar-driven videos |
| `cinematic` | Trailers, teasers, cinematic brand clips |
| `clip-factory` | Batch short-form clips from a long source |
| `documentary-montage` | Real-footage edits from stock/open archives |
| `hybrid` | Existing footage plus AI-generated support visuals |
| `localization-dub` | Subtitles, dubbing, translation |
| `podcast-repurpose` | Podcast highlights and audiograms |
| `screen-demo` | Software walkthroughs and polished screen demos |
| `talking-head` | Speaker footage, presentations, interviews |

Default pipeline flow:

```text
research -> proposal -> script -> scene_plan -> assets -> edit -> compose
```

Do not skip proposal, cost, or approval checkpoints when the pipeline expects them.

## Tool and Renderer Model

OpenMontage has three layers:

```text
Layer 1: tools/ + pipeline_defs/     What exists: executable capabilities and orchestration contracts
Layer 2: skills/                     How to use it: OpenMontage conventions and quality bars
Layer 3: .agents/skills/             How it works: external technology knowledge packs
```

Renderer choices:

| Renderer | Use For |
|---|---|
| Remotion | React-based video, animated explainers, data scenes, captions, image-based motion |
| HyperFrames | HTML/CSS/GSAP motion graphics, kinetic typography, launch reels, custom web-to-video |
| FFmpeg | Core assembly, encoding, subtitles, audio muxing, trimming, color work |

`render_runtime` is selected at proposal time and locked through edit decisions. Silent swaps between Remotion and HyperFrames are governance violations.

## Zero-Key and Low-Cost Paths

OpenMontage can create useful outputs without paid video-generation APIs.

Zero-key/free local stack:

- Piper TTS for narration.
- Archive.org, NASA, and Wikimedia Commons for open footage.
- FFmpeg for assembly and encoding.
- Built-in subtitle generation.
- Remotion or HyperFrames for programmatic animation.

Free-key stock media options:

- Pexels API key.
- Pixabay API key.
- Unsplash access key.

Cloud provider keys unlock higher-end image/video/audio generation but should be treated as optional. Do not ask the user to paste secrets into chat; point them to `.env` variable names or retrieve from the approved local secret store if this is Jason's environment.

Common `.env` variables:

```bash
FAL_KEY=...
PEXELS_API_KEY=...
PIXABAY_API_KEY=...
UNSPLASH_ACCESS_KEY=...
SUNO_API_KEY=...
ELEVENLABS_API_KEY=...
OPENAI_API_KEY=...
XAI_API_KEY=...
GOOGLE_API_KEY=...
HEYGEN_API_KEY=...
RUNWAY_API_KEY=...
```

Local GPU unlock:

```bash
make install-gpu

# .env
VIDEO_GEN_LOCAL_ENABLED=true
VIDEO_GEN_LOCAL_MODEL=wan2.1-1.3b
```

## Production Governance

OpenMontage has quality gates. Preserve them.

- Pre-compose validation checks delivery promises, slideshow risk, and renderer-family selection before render.
- Post-render self-review uses ffprobe validation, frame sampling, audio level checks, delivery-promise verification, and subtitle checks.
- Provider selection is scored across task fit, output quality, control features, reliability, cost efficiency, latency, and continuity.
- Decision logs should record provider choices, style/playbook choices, music/voice selections, renderer decisions, fallback reasoning, and costs.
- Budget controls estimate, reserve, reconcile, warn/cap, and pause above configured approval thresholds.

Never present a final video if review gates fail. Fix the issue, re-render, and verify again.

## Running Tests

From the repo root:

```bash
make test-contracts
make test
```

Use contract tests first when changing pipeline definitions, schemas, registry behavior, or tool signatures. Run a representative end-to-end render only after contract tests pass.

## Deliverable Verification

For any generated video, verify before handing it to the user:

```bash
ffprobe -v error -show_format -show_streams path/to/final.mp4
ffmpeg -i path/to/final.mp4 -vf "fps=1/10" -frames:v 6 /tmp/openmontage-frame-%02d.jpg
```

Check:

- File exists and has non-zero duration.
- Video and audio streams match the expected delivery profile.
- Frames are not black, broken, or duplicate slides unless intentionally static.
- Captions/subtitles exist when promised.
- Audio is not silent or clipping.
- Final format matches the target platform: landscape, Shorts/Reels vertical, square, or cinematic.

## Common Pitfalls

1. **Treating OpenMontage as prompt-to-video.** It is a pipeline system. Select a pipeline and read the stage-director skills.
2. **Skipping `AGENT_GUIDE.md`.** The repo's operating contract lives there.
3. **Ignoring zero-key paths.** The system can produce real footage montages and image-based videos without paid video APIs.
4. **Calling still-image animation “real footage.”** If the user asks for real footage only, use documentary montage / stock/open archives.
5. **Changing render runtime midstream.** Lock `render_runtime` once the proposal is approved.
6. **Presenting unverified renders.** Always run ffprobe/frame/audio/subtitle checks.
7. **Leaking or requesting API keys in chat.** Use `.env` and approved secret retrieval workflows.
8. **Skipping source-media inspection.** For user-supplied footage, inspect codec, resolution, duration, audio channels, scene boundaries, and actual visual content.

## Verification Checklist

- [ ] Repo is cloned from `https://github.com/calesthio/OpenMontage`.
- [ ] `AGENT_GUIDE.md` and `PROJECT_CONTEXT.md` were read before production work.
- [ ] Capability envelope and provider menu were inspected.
- [ ] Pipeline manifest and stage skills were read.
- [ ] Cost/provider decisions were recorded before spend.
- [ ] Render output passed ffprobe, frame sampling, audio, delivery-promise, and subtitle checks.
- [ ] Final artifact path and exact verification commands are included in the report.
