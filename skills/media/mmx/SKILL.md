---
name: mmx
description: MiniMax media CLI — music generation, song covers, text-to-speech, image generation, video generation, vision, and web search.
version: 1.0.0
author: MiniMax-AI
license: MIT
metadata:
  hermes:
    tags: [creative, music, speech, tts, image, video, vision, search, minimax]
    category: media
setup:
  help: "Get a MiniMax API key at https://platform.minimax.io or authenticate via browser with `mmx auth login`"
  install:
    - npm install -g mmx-cli
  collect_secrets:
    - env_var: MINIMAX_API_KEY
      prompt: "MiniMax API key (or skip and run `mmx auth login` for browser OAuth)"
      provider_url: "https://platform.minimax.io/subscribe/token-plan"
      secret: true
      optional: true
---

# MiniMax Media CLI (mmx)

Use `mmx` via the terminal tool for all MiniMax media generation tasks. Always add `--output json --non-interactive --no-color` for machine-readable output.

## When to Use

- User asks to generate, compose, or create music/songs
- User asks for text-to-speech or voice synthesis
- User asks to generate images from text
- User asks to generate videos from text or images
- User asks to describe/analyze an image (vision)
- User asks for a song cover or remix

## Prerequisites

```bash
which mmx || npm install -g mmx-cli
```

Auth (one of):
- `mmx auth login` (browser OAuth)
- `mmx auth login --api-key $MINIMAX_API_KEY`
- Pass `--api-key KEY` per command

## Global Flags

Always append to every command:

```
--output json --non-interactive --no-color
```

Optional: `--api-key KEY`, `--region cn` (China endpoint), `--timeout SECONDS`.

## Commands

### 1. Music Generation

```bash
# Instrumental
mmx music generate --prompt "upbeat jazz, piano and saxophone" --instrumental --out /tmp/music.mp3

# With lyrics
mmx music generate --prompt "indie folk ballad" --lyrics "[Verse]
Walking through the rain..." --out /tmp/song.mp3

# Auto-generate lyrics from prompt
mmx music generate --prompt "summer pop anthem about road trips" --lyrics-optimizer --out /tmp/song.mp3

# With detailed control
mmx music generate --prompt "cinematic orchestral" \
  --instrumental --genre orchestral --mood epic \
  --bpm 120 --key "D minor" --instruments "strings, brass, timpani" \
  --out /tmp/epic.mp3
```

Extra flags: `--vocals`, `--tempo`, `--avoid`, `--use-case`, `--structure`, `--references`, `--extra`, `--format`, `--sample-rate`, `--bitrate`.

**Rules**: exactly one of `--lyrics`, `--instrumental`, or `--lyrics-optimizer` is required.

### 2. Music Cover

```bash
# Cover from local file
mmx music cover --prompt "acoustic folk, warm male vocal" --audio-file /path/to/original.mp3 --out /tmp/cover.mp3

# Cover from URL
mmx music cover --prompt "lo-fi hip hop remix" --audio "https://example.com/song.mp3" --out /tmp/cover.mp3
```

Extra flags: `--lyrics`, `--seed`, `--format`, `--sample-rate`, `--bitrate`, `--channel`.

### 3. Text-to-Speech

```bash
mmx speech synthesize --text "Hello, world!" --voice English_expressive_narrator --out /tmp/speech.mp3

# From file
mmx speech synthesize --text-file article.txt --voice Chinese_calm_female --out /tmp/narration.mp3
```

Extra flags: `--speed`, `--volume`, `--pitch`, `--language`, `--format`, `--sample-rate`, `--model`.

#### List Voices

```bash
mmx speech voices
mmx speech voices --language chinese
mmx speech voices --language japanese
```

### 4. Image Generation

```bash
mmx image generate --prompt "A futuristic city at sunset" --out-dir /tmp

# With aspect ratio and count
mmx image generate --prompt "watercolor landscape" --aspect-ratio 16:9 --n 2 --out-dir /tmp
```

Shorthand: `mmx image "prompt text" --out-dir /tmp`

Extra flags: `--subject-ref`, `--out-prefix`.

### 5. Video Generation

Video is async — it returns a task ID, polls until done, then downloads.

```bash
# Wait for completion and download
mmx video generate --prompt "A cat playing piano" --download /tmp/video.mp4

# Image-to-video
mmx video generate --prompt "camera slowly zooms out" --first-frame /tmp/photo.jpg --download /tmp/video.mp4

# Async (returns task ID immediately)
mmx video generate --prompt "ocean waves at sunset" --no-wait
```

Check status / download separately:

```bash
mmx video task get --task-id TASK_ID
mmx video download --file-id FILE_ID --out /tmp/video.mp4
```

Extra flags: `--model MiniMax-Hailuo-2.3-Fast`, `--poll-interval`.

### 6. Vision (Image Description)

```bash
mmx vision describe --image /tmp/photo.jpg --prompt "What objects are in this image?"

# From URL
mmx vision describe --image "https://example.com/photo.jpg"
```

Shorthand: `mmx vision photo.jpg`

### 7. Web Search

```bash
mmx search query --q "MiniMax AI latest news"
```

Shorthand: `mmx search "query text"`

### 8. Quota

```bash
mmx quota show
```

## Output Handling

- All commands with `--output json` return structured JSON.
- Audio/image/video files are saved to the path specified by `--out` / `--out-dir`.
- Deliver generated media files to the user via `MEDIA:/path/to/file`.

## Pitfalls

- Music: must specify exactly ONE of `--lyrics`, `--instrumental`, or `--lyrics-optimizer`.
- Video: generation takes 1-5 minutes. Use `--download` to auto-save, or `--no-wait` + poll with `video task get`.
- Speech: max 10k characters per request. Split longer text.
- Always use `--out` or `--out-dir` to save files — without it, binary data goes to stdout.
- If auth fails, run `mmx auth status` to diagnose, then `mmx auth login`.
