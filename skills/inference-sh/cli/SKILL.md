---
name: inference-sh-cli
description: "Run 150+ AI apps via inference.sh CLI (infsh) - image generation, video creation, LLMs, search, 3D, Twitter automation. Models: FLUX, Veo, Gemini, Grok, Claude, Seedance, OmniHuman, Tavily, Exa, OpenRouter. Triggers: inference.sh, infsh, ai apps, serverless ai, flux, veo, image generation, video generation"
version: 1.0.0
author: inference.sh
license: MIT
metadata:
  hermes:
    tags: [AI, image-generation, video, LLM, search, inference, FLUX, Veo, Claude]
    requires_tools: [infsh]
---

# inference.sh CLI

Run 150+ AI apps in the cloud with a simple CLI. No GPU required.

**One API key for everything** - Manage all AI services (FLUX, Veo, Claude, Tavily, X/Twitter, and more) with a single inference.sh account. No need to sign up for dozens of different providers. You can also bring your own API keys if you prefer.

## Tools

This skill is backed by the `infsh` and `infsh_install` tools:

- **infsh**: Run any infsh command (app list, app run, etc.)
- **infsh_install**: Install the CLI if not already present

## Quick Start

```bash
# Install (if needed)
infsh_install

# List available apps
infsh app list

# Search for apps
infsh app list --search flux
infsh app list --search video

# Run an app
infsh app run falai/flux-dev-lora --input '{"prompt": "a cat astronaut"}' --json
```

## Local File Uploads

The CLI automatically uploads local files when you provide a file path instead of a URL:

```bash
# Upscale a local image
infsh app run falai/topaz-image-upscaler --input '{"image": "/path/to/photo.jpg", "upscale_factor": 2}' --json

# Image-to-video from local file
infsh app run falai/wan-2-5-i2v --input '{"image": "/path/to/image.png", "prompt": "make it come alive"}' --json

# Video generation with local first frame
infsh app run bytedance/seedance-1-5-pro --input '{"prompt": "dancing figure", "image": "./first-frame.png"}' --json
```

## Image Generation

```bash
# Gemini 2.5 Flash Image (Google) - fast, high quality
infsh app run google/gemini-2-5-flash-image --input '{"prompt": "futuristic city", "num_images": 1}' --json

# Gemini 3 Pro Image Preview (Google) - latest model
infsh app run google/gemini-3-pro-image-preview --input '{"prompt": "photorealistic landscape"}' --json

# Gemini 3.1 Flash Image Preview (Google)
infsh app run google/gemini-3-1-flash-image-preview --input '{"prompt": "artistic portrait"}' --json

# FLUX Dev with LoRA support
infsh app run falai/flux-dev-lora --input '{"prompt": "sunset over mountains", "num_images": 1}' --json

# FLUX 2 Klein with LoRA
infsh app run falai/flux-2-klein-lora --input '{"prompt": "portrait photo"}' --json

# Reve - stylized generation and editing
infsh app run falai/reve --input '{"prompt": "cyberpunk city"}' --json

# Seedream 5 Lite - high-quality 2K-3K (ByteDance)
infsh app run bytedance/seedream-5-lite --input '{"prompt": "nature scene"}' --json

# Seedream 4.5 - 2K-4K images
infsh app run bytedance/seedream-4-5 --input '{"prompt": "detailed illustration"}' --json

# Seedream 3.0 - cinematic quality
infsh app run bytedance/seedream-3-0-t2i --input '{"prompt": "fantasy landscape"}' --json

# Grok Imagine - xAI image generation
infsh app run xai/grok-imagine-image --input '{"prompt": "abstract art"}' --json

# Grok Imagine Pro - higher quality
infsh app run xai/grok-imagine-image-pro --input '{"prompt": "photorealistic portrait"}' --json

# Qwen Image 2 Pro (Alibaba)
infsh app run alibaba/qwen-image-2-pro --input '{"prompt": "anime character"}' --json
```

## Video Generation

```bash
# Veo 3.1 Fast (Google)
infsh app run google/veo-3-1-fast --input '{"prompt": "drone shot of coastline"}' --json

# Veo 3.1 (higher quality)
infsh app run google/veo-3-1 --input '{"prompt": "cinematic scene"}' --json

# Veo 3 Fast
infsh app run google/veo-3-fast --input '{"prompt": "nature documentary shot"}' --json

# Veo 2
infsh app run google/veo-2 --input '{"prompt": "slow motion water splash"}' --json

# Grok Imagine Video - xAI
infsh app run xai/grok-imagine-video --input '{"prompt": "timelapse of clouds"}' --json

# Seedance 1.5 Pro - ByteDance
infsh app run bytedance/seedance-1-5-pro --input '{"prompt": "dancing figure", "resolution": "1080p"}' --json

# Seedance 1.0 Pro
infsh app run bytedance/seedance-1-0-pro --input '{"prompt": "walking through forest"}' --json

# Seedance 1.0 Pro Fast
infsh app run bytedance/seedance-1-0-pro-fast --input '{"prompt": "quick motion"}' --json

# Seedance 1.0 Lite - 720p lightweight
infsh app run bytedance/seedance-1-0-lite --input '{"prompt": "simple animation"}' --json

# Wan 2.5 - text-to-video
infsh app run falai/wan-2-5 --input '{"prompt": "person walking through city"}' --json

# Wan 2.5 Image-to-Video
infsh app run falai/wan-2-5-i2v --input '{"image": "/path/to/image.png", "prompt": "make it move naturally"}' --json

# LTX Video
infsh app run infsh/ltx-video --input '{"prompt": "realistic scene"}' --json

# Magi 1
infsh app run infsh/magi-1 --input '{"prompt": "creative video"}' --json
```

## Avatar & Lipsync

```bash
# OmniHuman 1.5 - multi-character audio-driven avatars
infsh app run bytedance/omnihuman-1-5 --input '{"audio": "/path/to/audio.mp3", "image": "/path/to/face.jpg"}' --json

# OmniHuman 1.0
infsh app run bytedance/omnihuman-1-0 --input '{"audio": "/path/to/speech.wav", "image": "/path/to/portrait.png"}' --json

# Fabric 1.0 - image animation
infsh app run falai/fabric-1-0 --input '{"image": "/path/to/photo.jpg"}' --json

# PixVerse Lipsync
infsh app run falai/pixverse-lipsync --input '{"audio": "/path/to/audio.mp3", "video": "/path/to/video.mp4"}' --json
```

## Upscaling

```bash
# Topaz Image Upscaler - up to 4x
infsh app run falai/topaz-image-upscaler --input '{"image": "/path/to/photo.jpg", "upscale_factor": 2}' --json

# Topaz Video Upscaler
infsh app run falai/topaz-video-upscaler --input '{"video": "/path/to/video.mp4"}' --json

# Real-ESRGAN - image enhancement
infsh app run infsh/real-esrgan --input '{"image": "/path/to/image.jpg"}' --json

# Thera - upscale to any size
infsh app run infsh/thera --input '{"image": "/path/to/image.jpg"}' --json
```

## LLMs (via OpenRouter)

```bash
# Claude Opus 4.6
infsh app run openrouter/claude-opus-46 --input '{"prompt": "Explain quantum computing"}' --json

# Claude Sonnet 4.5
infsh app run openrouter/claude-sonnet-45 --input '{"prompt": "Write a poem"}' --json

# Claude Haiku 4.5
infsh app run openrouter/claude-haiku-45 --input '{"prompt": "Quick question"}' --json

# Gemini 3 Pro Preview
infsh app run openrouter/gemini-3-pro-preview --input '{"prompt": "Analyze this"}' --json

# Kimi K2 Thinking
infsh app run openrouter/kimi-k2-thinking --input '{"prompt": "Solve this step by step"}' --json

# GLM 4.6
infsh app run openrouter/glm-46 --input '{"prompt": "Help me with"}' --json

# MiniMax M2.5
infsh app run openrouter/minimax-m-25 --input '{"prompt": "Creative writing"}' --json

# Intellect 3
infsh app run openrouter/intellect-3 --input '{"prompt": "Research question"}' --json
```

## Web Search

```bash
# Tavily Search Assistant - comprehensive results
infsh app run tavily/search-assistant --input '{"query": "latest AI news", "include_answer": true}' --json

# Tavily Extract - get content from URLs
infsh app run tavily/extract --input '{"urls": ["https://example.com"]}' --json

# Exa Search - neural search
infsh app run exa/search --input '{"query": "machine learning tutorials"}' --json

# Exa Answer - LLM-powered answers
infsh app run exa/answer --input '{"query": "what is transformers architecture"}' --json

# Exa Extract - extract web content
infsh app run exa/extract --input '{"url": "https://example.com"}' --json
```

## 3D Generation

```bash
# Rodin 3D Generator
infsh app run infsh/rodin-3d-generator --input '{"prompt": "a wooden chair"}' --json

# HunyuanImage to 3D
infsh app run infsh/hunyuan-image-to-3d-2 --input '{"image": "/path/to/object.png"}' --json
```

## Text-to-Speech

```bash
# Kokoro TTS - lightweight
infsh app run falai/kokoro-tts --input '{"text": "Hello, this is a test."}' --json

# Dia TTS - realistic dialogue
infsh app run falai/dia-tts --input '{"text": "Two characters talking"}' --json
```

## Twitter/X Automation

```bash
# Post a tweet
infsh app run x/post-tweet --input '{"text": "Hello from AI!"}' --json

# Create post with media
infsh app run x/post-create --input '{"text": "Check this out", "media": "/path/to/image.jpg"}' --json

# Send DM
infsh app run x/dm-send --input '{"recipient_id": "123456", "text": "Hi there!"}' --json

# Follow user
infsh app run x/user-follow --input '{"user_id": "123456"}' --json

# Like a post
infsh app run x/post-like --input '{"post_id": "123456789"}' --json

# Retweet
infsh app run x/post-retweet --input '{"post_id": "123456789"}' --json

# Get user profile
infsh app run x/user-get --input '{"username": "elonmusk"}' --json

# Get post
infsh app run x/post-get --input '{"post_id": "123456789"}' --json

# Delete post
infsh app run x/post-delete --input '{"post_id": "123456789"}' --json
```

## Utilities

```bash
# Browser automation
infsh app run infsh/agent-browser --function open --session new --input '{"url": "https://example.com"}' --json

# Media merger - combine videos/images
infsh app run infsh/media-merger --input '{"files": ["/path/to/video1.mp4", "/path/to/video2.mp4"]}' --json

# Video audio extractor
infsh app run infsh/video-audio-extractor --input '{"video": "/path/to/video.mp4"}' --json

# Video audio merger
infsh app run infsh/video-audio-merger --input '{"video": "/path/to/video.mp4", "audio": "/path/to/audio.mp3"}' --json

# Caption videos
infsh app run infsh/caption-videos --input '{"video": "/path/to/video.mp4"}' --json

# Stitch images
infsh app run infsh/stitch-images --input '{"images": ["/path/to/1.jpg", "/path/to/2.jpg"]}' --json

# Python executor
infsh app run infsh/python-executor --input '{"code": "print(2+2)"}' --json

# HTML to image
infsh app run infsh/html-to-image --input '{"html": "<h1>Hello</h1>"}' --json

# NSFW detection
infsh app run infsh/falconsai-nsfw-detection --input '{"image": "/path/to/image.jpg"}' --json

# Media analyzer
infsh app run infsh/media-analyzer --input '{"file": "/path/to/media.jpg"}' --json
```

## Common Patterns

### Generate + Upscale Pipeline

```bash
# Generate image, capture URL, then upscale
infsh app run falai/flux-dev-lora --input '{"prompt": "portrait photo"}' --json --save result.json

# Extract URL and upscale (using jq)
IMG=$(cat result.json | jq -r '.images[0].url')
infsh app run falai/topaz-image-upscaler --input "{\"image\": \"$IMG\", \"upscale_factor\": 2}" --json
```

### Get App Schema

```bash
# See what inputs an app accepts
infsh app get falai/flux-dev-lora

# Generate sample input
infsh app sample falai/flux-dev-lora

# Save sample to file, edit, then run
infsh app sample falai/flux-dev-lora --save input.json
# edit input.json...
infsh app run falai/flux-dev-lora --input input.json --json
```

### Long-running Tasks

```bash
# Start without waiting
infsh app run google/veo-3-1 --input '{"prompt": "..."}' --no-wait

# Check status later
infsh task get <task-id>

# Save result when done
infsh task get <task-id> --save result.json
```

## Available Categories

| Category | Apps |
|----------|------|
| **Image** | google/nano-banana, google/nano-banana-pro, google/nano-banana-2, falai/flux-dev-lora, bytedance/seedream-5-lite, falai/reve, xai/grok-imagine-image |
| **Video** | google/veo-*, xai/grok-imagine-video, bytedance/seedance-*, falai/wan-2-5*, infsh/ltx-video, infsh/magi-1 |
| **Avatar** | bytedance/omnihuman-*, falai/fabric-1-0, falai/pixverse-lipsync |
| **Upscale** | falai/topaz-image-upscaler, falai/topaz-video-upscaler, infsh/real-esrgan, infsh/thera |
| **LLMs** | openrouter/claude-*, openrouter/gemini-*, openrouter/kimi-*, openrouter/glm-* |
| **Search** | tavily/search-assistant, tavily/extract, exa/search, exa/answer, exa/extract |
| **3D** | infsh/rodin-3d-generator, infsh/hunyuan-image-to-3d-2 |
| **TTS** | falai/kokoro-tts, falai/dia-tts |
| **Social** | x/post-tweet, x/post-create, x/dm-send, x/user-follow, x/post-like, x/post-retweet |
| **Utils** | infsh/agent-browser, infsh/media-merger, infsh/caption-videos, infsh/stitch-images |

## Reference Files

- [Authentication & Setup](references/authentication.md)
- [Discovering Apps](references/app-discovery.md)
- [Running Apps](references/running-apps.md)
- [CLI Reference](references/cli-reference.md)

## Documentation

- [inference.sh Docs](https://inference.sh/docs)
- [CLI Setup Guide](https://inference.sh/docs/extend/cli-setup)
- [Apps Overview](https://inference.sh/docs/apps/overview)
