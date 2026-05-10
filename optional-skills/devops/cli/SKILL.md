---
name: inference-sh-cli
description: "Run 150+ AI apps via inference.sh CLI (infsh) — image generation, video creation, LLMs, search, 3D, social automation. Uses the terminal tool. Triggers: inference.sh, infsh, ai apps, flux, veo, image generation, video generation, seedream, seedance, tavily"
version: 1.0.0
author: okaris
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [AI, image-generation, video, LLM, search, inference, FLUX, Veo, Claude]
    related_skills: []
---

# inference.sh CLI

Run 150+ AI apps in the cloud with a simple CLI. No GPU required.

All commands use the **terminal tool** to run `infsh` commands.

## When to Use

- User asks to generate images (FLUX, Reve, Seedream, Grok, Gemini image)
- User asks to generate video (Veo, Wan, Seedance, OmniHuman)
- User asks about inference.sh or infsh
- User wants to run AI apps without managing individual provider APIs
- User asks for AI-powered search (Tavily, Exa)
- User needs avatar/lipsync generation

## Prerequisites

The `infsh` CLI must be installed and authenticated. Check with:

```bash
infsh me
```

If not installed:

```bash
curl -fsSL https://cli.inference.sh | sh
infsh login
```

See `references/authentication.md` for full setup details.

## Workflow

### 1. Always Search First

Never guess app names — always search to find the correct app ID:

```bash
infsh app list --search flux
infsh app list --search video
infsh app list --search image
```

### 2. Run an App

Use the exact app ID from the search results. Always use `--json` for machine-readable output:

```bash
infsh app run <app-id> --input '{"prompt": "your prompt here"}' --json
```

### 3. Parse the Output

The JSON output contains URLs to generated media. Present these to the user with `MEDIA:<url>` for inline display.

## Common Commands

### Image Generation

```bash
# Search for image apps
infsh app list --search image

# FLUX Dev with LoRA
infsh app run falai/flux-dev-lora --input '{"prompt": "sunset over mountains", "num_images": 1}' --json

# Gemini image generation
infsh app run google/gemini-2-5-flash-image --input '{"prompt": "futuristic city", "num_images": 1}' --json

# Seedream (ByteDance)
infsh app run bytedance/seedream-5-lite --input '{"prompt": "nature scene"}' --json

# Grok Imagine (xAI)
infsh app run xai/grok-imagine-image --input '{"prompt": "abstract art"}' --json
```

### Video Generation

```bash
# Search for video apps
infsh app list --search video

# Veo 3.1 (Google)
infsh app run google/veo-3-1-fast --input '{"prompt": "drone shot of coastline"}' --json

# Seedance (ByteDance)
infsh app run bytedance/seedance-1-5-pro --input '{"prompt": "dancing figure", "resolution": "1080p"}' --json

# Wan 2.5
infsh app run falai/wan-2-5 --input '{"prompt": "person walking through city"}' --json
```

### Local File Uploads

The CLI automatically uploads local files when you provide a path:

```bash
# Upscale a local image
infsh app run falai/topaz-image-upscaler --input '{"image": "/path/to/photo.jpg", "upscale_factor": 2}' --json

# Image-to-video from local file
infsh app run falai/wan-2-5-i2v --input '{"image": "/path/to/image.png", "prompt": "make it move"}' --json

# Avatar with audio
infsh app run bytedance/omnihuman-1-5 --input '{"audio": "/path/to/audio.mp3", "image": "/path/to/face.jpg"}' --json
```

### Search & Research

```bash
infsh app list --search search
infsh app run tavily/tavily-search --input '{"query": "latest AI news"}' --json
infsh app run exa/exa-search --input '{"query": "machine learning papers"}' --json
```

### Other Categories

```bash
# 3D generation
infsh app list --search 3d

# Audio / TTS
infsh app list --search tts

# Twitter/X automation
infsh app list --search twitter
```

## Pitfalls

1. **Never guess app IDs** — always run `infsh app list --search <term>` first. App IDs change and new apps are added frequently.
2. **Always use `--json`** — raw output is hard to parse. The `--json` flag gives structured output with URLs.
3. **Check authentication** — if commands fail with auth errors, run `infsh login` or verify `INFSH_API_KEY` is set.
4. **Long-running apps** — video generation can take 30-120 seconds. The terminal tool timeout should be sufficient, but warn the user it may take a moment.
5. **Input format** — the `--input` flag takes a JSON string. Make sure to properly escape quotes.

## Reference Docs

- `references/authentication.md` — Setup, login, API keys
- `references/app-discovery.md` — Searching and browsing the app catalog
- `references/running-apps.md` — Running apps, input formats, output handling
- `references/cli-reference.md` — Complete CLI command reference
