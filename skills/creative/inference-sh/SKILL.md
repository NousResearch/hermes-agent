---
name: inference-sh
description: Generate images, video, audio, and avatars via CLI.
version: 1.0.0
author: inference.sh
license: MIT
metadata:
  hermes:
    tags: [image-generation, video-generation, text-to-speech, avatars, ai-media]
    related_skills: [inference-sh-workflows, comfyui, songwriting-and-ai-music]
    requires_toolsets: [terminal]
required_environment_variables:
  - name: INFERENCE_API_KEY
    prompt: "inference.sh API Key"
    help: "Get from https://inference.sh/settings/api-keys"
    required_for: full functionality
---

# inference.sh — AI Media Generation

Generate images, videos, audio, talking-head avatars, and music from the terminal. One API key, 250+ models, pay-per-use.

## When to Use

- Generate images from text (product shots, illustrations, logos, thumbnails)
- Create videos from text or images (promos, shorts, animations)
- Generate talking-head avatars from a portrait and script
- Convert text to speech with emotion and multilingual support
- Create music, sound effects, or podcast audio
- Upscale, edit, or remix existing media

## Prerequisites

Install the CLI via the `terminal` tool:

```bash
curl -fsSL cli.inference.sh | sh
```

Log in (requires API key from https://inference.sh/settings/api-keys):

```bash
belt login
```

## How to Run

All generation follows the same pattern through the `terminal` tool:

```bash
belt app run <model> --input '{"prompt": "..."}'
```

Check available models:

```bash
belt app store --category image    # or: video, audio, avatar, tts
belt app store search "flux"       # search by name
```

Get a model's input schema:

```bash
belt app sample <model> --save input.json
```

## Quick Reference

### Image Generation

| Model | Best For | Notes |
|-------|----------|-------|
| `flux/dev-lora` | General quality | LoRA style support |
| `seedream/4.5` | 4K cinematic, text rendering | ByteDance |
| `gpt-image-2` | Editing, inpainting, multi-ref | OpenAI |
| `gemini/image-3-pro` | High fidelity | Google |
| `grok-imagine` | Fast creative | xAI |
| `reve/image` | Natural language editing | Background swap, text overlay |
| `imagineart/1.5-pro` | Ultra-high-fidelity 4K | — |
| `p-image` | Fast and cheap | $0.0001/image with LoRA |

```bash
# Generate an image
belt app run flux/dev-lora --input '{"prompt": "a robot painting in a studio, oil on canvas style"}'

# Edit an existing image
belt app run gpt-image-2 --input '{"prompt": "remove the background", "image": "photo.png"}'

# Upscale
belt app run topaz/image-upscaler --input '{"image": "output.png", "scale": 4}'
```

### Video Generation

| Model | Best For | Notes |
|-------|----------|-------|
| `veo/3.1` | Highest quality | Google, frame interpolation |
| `seedance/2.0` | Audio-synced video | Text/image/ref-to-video, up to 1080p |
| `happyhorse/1.0` | Physically realistic | Alibaba, up to 15s |
| `wan/2.5-i2v` | Animate any image | Image-to-video |
| `grok-video` | Fast creative | xAI |
| `p-video` | Speed and cost | With audio support |

```bash
# Text to video
belt app run veo/3.1 --input '{"prompt": "aerial drone shot of coastal cliffs at golden hour"}'

# Image to video (animate a still)
belt app run seedance/2.0 --input '{"prompt": "gentle wind blowing", "image": "landscape.png", "generate_audio": true}'

# Extend a video
belt app run seedance/2.0 --input '{"prompt": "camera pulls back to reveal the city", "video": "clip.mp4"}'
```

### Talking-Head Avatars

| Model | Best For | Notes |
|-------|----------|-------|
| `p-video-avatar` | Fast, cheap, built-in TTS | 30 voices, 10 languages, 1080p |
| `omnihuman/1.5` | Multi-character | ByteDance, audio-driven |
| `fabric/1.0` | Lipsync with built-in TTS | — |
| `pixverse-lipsync` | Realistic lipsync | — |

```bash
# Avatar with built-in TTS (recommended)
belt app run p-video-avatar --input '{"image": "headshot.png", "text": "Welcome to our product demo.", "voice_id": "en-US-male-1"}'

# Avatar with separate audio
belt app run omnihuman/1.5 --input '{"image": "headshot.png", "audio": "narration.mp3"}'
```

### Text-to-Speech

| Model | Best For | Notes |
|-------|----------|-------|
| `inworld/tts-2` | Emotion steering, 100+ languages | Use `[happy]`, `[serious]` brackets |
| `elevenlabs/tts` | Premium quality, 32 languages | 22+ voices, Flash v2.5 for speed |
| `kokoro/tts` | Natural and fast | Voices: `am_michael`, `af_sarah` |
| `dia/tts` | Conversational | Expressive dialogue |

```bash
# TTS with emotion
belt app run inworld/tts-2 --input '{"text": "[excited] Check this out!", "voice_id": "en-US-narrator-1"}'

# Premium TTS
belt app run elevenlabs/tts --input '{"text": "Hello world", "voice_id": "rachel"}'
```

### Music and Sound Effects

```bash
# Generate a song (up to 10 min)
belt app run elevenlabs/music --input '{"prompt": "upbeat lo-fi hip hop, rainy day vibes", "duration": 120}'

# Add sound effects to a silent video
belt app run hunyuanvideo/foley --input '{"video": "clip.mp4"}'
```

## Common Pipelines

**Product promo video:**
```bash
# 1. Generate product shot
belt app run seedream/4.5 --input '{"prompt": "minimalist product photo of wireless earbuds on marble"}'
# 2. Animate to video with audio
belt app run seedance/2.0 --input '{"prompt": "slow rotation reveal", "image": "output.png", "generate_audio": true}'
# 3. Add voiceover
belt app run p-video-avatar --input '{"image": "presenter.png", "text": "Introducing our new earbuds."}'
```

**Podcast episode:**
```bash
# 1. Generate multi-voice dialogue
belt app run elevenlabs/tts --input '{"text": "Host: Welcome back...", "voice_id": "josh"}'
# 2. Generate background music
belt app run elevenlabs/music --input '{"prompt": "soft podcast intro jingle", "duration": 15}'
```

## Pitfalls

- **Async tasks:** Long generations (video, avatars) run async. Use `belt task get <id>` to poll status and retrieve output.
- **File references:** Use local file paths for `image`, `video`, `audio` fields — the CLI uploads them automatically.
- **Seedance audio:** Always set `"generate_audio": true` for seedance image-to-video and ref-to-video unless you want silent output.
- **Rate limits:** Some models have per-minute limits. If you hit 429 errors, wait and retry.
- **Model availability:** Run `belt app get <model>` to check if a model is currently available before building a pipeline around it.

## Verification

```bash
# Verify CLI is installed and authenticated
belt whoami

# Verify image generation works
belt app run p-image --input '{"prompt": "a red circle on white background"}'
```
