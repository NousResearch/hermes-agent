---
name: inference-sh
description: "Generate images, video, audio, avatars, and music via the inference.sh CLI (belt). 250+ cloud AI models with one API key. Use when the user asks to generate images (FLUX, Seedream, GPT-Image-2, Gemini, Grok Imagine, Reve), create video (Veo, Seedance, Wan, HappyHorse), make avatars (P-Video-Avatar, OmniHuman, Fabric, PixVerse), generate speech or music (ElevenLabs, Inworld TTS, Kokoro, DIA), upscale media, remove backgrounds, or add sound effects to video."
version: 2.0.0
author: okaris
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [image-generation, video-generation, text-to-speech, avatars, music, ai-media, FLUX, Veo, Seedance, Seedream, ElevenLabs, inference-sh]
    related_skills: [inference-sh-workflows, comfyui, songwriting-and-ai-music]
    requires_toolsets: [terminal]
required_environment_variables:
  - name: INFERENCE_API_KEY
    prompt: "inference.sh API Key"
    help: "Sign up at https://inference.sh and get your key from https://inference.sh/settings/api-keys"
    required_for: full functionality
---

# inference.sh — AI Media Generation

Generate images, videos, audio, talking-head avatars, and music from the terminal. One API key, 250+ models, pay-per-use. No GPU required.

All commands use the `terminal` tool to run `belt` commands.

## When to Use

- User asks to generate images (FLUX, Seedream, GPT-Image-2, Gemini, Grok Imagine, Reve, P-Image)
- User asks to create video (Veo 3.1, Seedance 2.0, Wan 2.5, HappyHorse, Grok Video)
- User asks for talking-head avatars or lipsync (P-Video-Avatar, OmniHuman, Fabric, PixVerse)
- User asks for text-to-speech (Inworld TTS-2, ElevenLabs, Kokoro, DIA)
- User asks for music generation (ElevenLabs Music, Diffrythm)
- User wants to upscale images/video, remove backgrounds, or add sound effects
- User mentions inference.sh or belt CLI

## Prerequisites

The `belt` CLI must be installed and authenticated. Check with:

```bash
belt whoami
```

If not installed:

```bash
curl -fsSL cli.inference.sh | sh
belt login
```

See `references/authentication.md` for full setup details.

## Workflow

### 1. Search First

Never guess app IDs — always search to find the correct app:

```bash
belt app store search "flux"
belt app store --category image
belt app store --category video
```

### 2. Run an App

Use the exact app ID from search results:

```bash
belt app run <app-id> --input '{"prompt": "your prompt here"}'
```

### 3. Handle Output

The output contains URLs to generated media. For long-running tasks (video, avatars), use `--no-wait` and poll:

```bash
belt app run <app-id> --input input.json --no-wait
belt task get <task-id>
```

## Image Generation

| Model | App ID | Best For |
|-------|--------|----------|
| FLUX Dev LoRA | `flux/dev-lora` | Quality with LoRA styles |
| Seedream 4.5 | `seedream/4.5` | 4K cinematic, text in images |
| GPT-Image-2 | `gpt-image-2` | Editing, inpainting, multi-reference |
| Gemini 3 Pro | `gemini/image-3-pro` | High fidelity |
| Grok Imagine | `grok-imagine` | Fast creative |
| Reve | `reve/image` | Natural language editing |
| P-Image | `p-image` | Fastest and cheapest ($0.0001) |

```bash
# Generate an image
belt app run flux/dev-lora --input '{"prompt": "a robot painting in a studio, oil on canvas style"}'

# Edit an existing image
belt app run gpt-image-2 --input '{"prompt": "remove the background", "image": "photo.png"}'

# Upscale
belt app run topaz/image-upscaler --input '{"image": "output.png", "scale": 4}'

# Remove background
belt app run birefnet/bg-remove --input '{"image": "product.png"}'
```

## Video Generation

| Model | App ID | Best For |
|-------|--------|----------|
| Veo 3.1 | `veo/3.1` | Highest quality, frame interpolation |
| Seedance 2.0 | `seedance/2.0` | Audio-synced video, multi-modal input |
| HappyHorse 1.0 | `happyhorse/1.0` | Physically realistic motion |
| Wan 2.5 I2V | `wan/2.5-i2v` | Animate any image |
| Grok Video | `grok-video` | Fast creative |
| P-Video | `p-video` | Speed and cost |

```bash
# Text to video
belt app run veo/3.1 --input '{"prompt": "aerial drone shot of coastal cliffs at golden hour"}'

# Image to video with audio
belt app run seedance/2.0 --input '{"prompt": "gentle wind blowing", "image": "landscape.png", "generate_audio": true}'

# Extend a video
belt app run seedance/2.0 --input '{"prompt": "camera pulls back to reveal the city", "video": "clip.mp4"}'

# Add sound effects to silent video
belt app run hunyuanvideo/foley --input '{"video": "silent-clip.mp4"}'
```

## Talking-Head Avatars

| Model | App ID | Best For |
|-------|--------|----------|
| P-Video-Avatar | `p-video-avatar` | Built-in TTS, 30 voices, cheapest |
| OmniHuman 1.5 | `omnihuman/1.5` | Multi-character, audio-driven |
| Fabric 1.0 | `fabric/1.0` | Lipsync with built-in TTS |
| PixVerse Lipsync | `pixverse-lipsync` | Most realistic lip movement |

```bash
# Avatar with built-in TTS (recommended — one step)
belt app run p-video-avatar --input '{"image": "headshot.png", "text": "Welcome to our product demo.", "voice_id": "en-US-male-1"}'

# Avatar with separate audio
belt app run omnihuman/1.5 --input '{"image": "headshot.png", "audio": "narration.mp3"}'
```

## Text-to-Speech

| Model | App ID | Best For |
|-------|--------|----------|
| Inworld TTS-2 | `inworld/tts-2` | Emotion steering, 100+ languages |
| ElevenLabs | `elevenlabs/tts` | Premium quality, 32 languages |
| Kokoro | `kokoro/tts` | Natural and fast |
| DIA | `dia/tts` | Conversational dialogue |

```bash
# TTS with emotion control
belt app run inworld/tts-2 --input '{"text": "[excited] Check this out!", "voice_id": "en-US-narrator-1"}'

# Premium TTS
belt app run elevenlabs/tts --input '{"text": "Hello world", "voice_id": "rachel"}'
```

## Music and Sound

```bash
# Generate a song (up to 10 min, commercial license)
belt app run elevenlabs/music --input '{"prompt": "upbeat lo-fi hip hop, rainy day vibes", "duration": 120}'

# Sound effects from text
belt app run elevenlabs/sound-effects --input '{"prompt": "thunderstorm with distant rumbling"}'

# Transcribe audio (98%+ accuracy, 90+ languages)
belt app run elevenlabs/scribe-v2 --input '{"audio": "recording.mp3"}'
```

## Common Pipelines

**Product promo video:**
```bash
# 1. Generate product shot
belt app run seedream/4.5 --input '{"prompt": "minimalist product photo of wireless earbuds on marble"}'
# 2. Animate to video with audio
belt app run seedance/2.0 --input '{"prompt": "slow rotation reveal", "image": "output.png", "generate_audio": true}'
```

**AI presenter:**
```bash
# 1. Generate a portrait
belt app run seedream/4.5 --input '{"prompt": "professional headshot, business casual, neutral background"}'
# 2. Create talking avatar
belt app run p-video-avatar --input '{"image": "output.png", "text": "Hello, welcome to the demo.", "voice_id": "en-US-female-1"}'
```

**Podcast episode:**
```bash
# 1. Generate speech
belt app run elevenlabs/tts --input '{"text": "Welcome back to the show...", "voice_id": "josh"}'
# 2. Generate background music
belt app run elevenlabs/music --input '{"prompt": "soft podcast intro jingle", "duration": 15}'
```

## Pitfalls

1. **Always search first** — run `belt app store search <term>` before running. App IDs change and new apps are added frequently.
2. **Async tasks** — video and avatar generation runs async. Use `belt task get <id>` to poll status.
3. **File uploads** — pass local file paths for `image`, `video`, `audio` fields. The CLI uploads automatically.
4. **Seedance audio** — always set `"generate_audio": true` for Seedance image-to-video unless you want silent output.
5. **Long timeouts** — video generation can take 30-120 seconds. Use `--no-wait` for background processing.

## Verification

```bash
# Verify CLI is installed and authenticated
belt whoami

# Quick test — generate a simple image
belt app run p-image --input '{"prompt": "a red circle on white background"}'
```

## Reference Docs

- `references/authentication.md` — Setup, login, API keys
- `references/app-discovery.md` — Searching and browsing the app catalog
- `references/running-apps.md` — Running apps, input formats, output handling
- `references/image-generation.md` — Full image model reference
- `references/video-generation.md` — Full video model reference
- `references/audio-and-tts.md` — TTS, music, transcription reference
- `references/avatars.md` — Talking-head avatar reference
