---
name: mmx-cli
description: >-
  MiniMax MMX CLI — generate images, video, music, speech, and text from the
  terminal. Installed globally as `mmx` via npm (package `mmx-cli`). API key
  stored in `~/.mmx/config.json`, sourced from Infisical as `MINIMAX_API_KEY`.
  Use when the user asks to generate media and Higgsfield is unavailable,
  or for the specific MiniMax models (music-2.6, image-01, video gen).
version: 1.0.0
tags: [minimax, mmx, image, video, music, ai-generation]
related_skills:
  - higgsfield-generate
  - ffmpeg-media
  - besoul-content-operations
---

# MMX CLI (MiniMax)

## Install

```bash
npm install -g mmx-cli
```

## Auth

The API key lives in Infisical as `MINIMAX_API_KEY`. Authenticate with:

```bash
# Non-interactive (use from a script or agent)
MINIMAX_KEY="$MINIMAX_API_KEY"
npx --yes mmx-cli auth login --api-key "$MINIMAX_KEY"
```

Or interactively:
```bash
npx mmx-cli auth login
npx mmx-cli auth status   # verify
```

The binary is installed as a `.mjs` file under npm's global node_modules.
Use `npx mmx-cli <command>` to invoke it (wrapper symlink may not work).

## Quota

```bash
npx mmx-cli quota show   # check remaining credits
```

Typical: 99% general, 3/3 video per day, 21/week.

## Available resources

| mmx <resource> | Purpose |
|---|---|
| image | Generate images (image-01) |
| video | Generate video (async, progress tracking) |
| music | Generate songs/instrumentals (music-2.6) |
| speech | TTS with 30+ voices |
| text | Multi-turn chat, streaming |
| vision | Image understanding |
| search | Web search |

## Image generation

```bash
# Vertical for IG (9:16)
npx mmx-cli image generate \
  --prompt "cinematic twilight scene, lone figure on hill, dramatic clouds" \
  --aspect-ratio 9:16 \
  --prompt-optimizer \
  --out /tmp/output.jpg
```

Supported ratios: 16:9, 1:1, 9:16, 4:3, 3:4
Custom dimensions: --width / --height (512-2048, multiple of 8)

## Music generation

```bash
# Instrumental (best for reel backgrounds)
npx mmx-cli music generate \
  --prompt "cinematic orchestral, building tension, emotional rising strings" \
  --instrumental \
  --mood "uplifting, inspiring" \
  --out /tmp/music.mp3
```

Important: the music model generates FULL songs (60-90s), not short clips.
For 7-10s reel backgrounds, trim with FFmpeg:
```bash
ffmpeg -y -i music_long.mp3 -t 9 -acodec copy music_9s.mp3
```

Or generate a short ambient pad with FFmpeg synths directly:
```bash
ffmpeg -y -f lavfi -i "sine=f=55:d=9,volume=0.25" \
  -f lavfi -i "sine=f=110:d=9,volume=0.12" \
  -f lavfi -i "sine=f=220:d=9,volume=0.06" \
  -f lavfi -i "anoisesrc=d=9:c=pink:a=0.5,lowpass=f=400" \
  -filter_complex "...amix=inputs=4:duration=first[out]" \
  -map "[out]" -ac 2 -ar 44100 -b:a 192k \
  /tmp/music_9s.mp3
```

## Video generation

```bash
npx mmx-cli video generate \
  --prompt "Ocean waves at sunset" \
  --out /tmp/output.mp4
```

(Async — polls for completion automatically.)

## Common patterns for BeSoul reels

### Cinematic still + text overlay + music (the proven combo)

1. Generate background image with `mmx image`:
   ```bash
   npx mmx-cli image generate --prompt "..." --aspect-ratio 9:16 --out frame.jpg
   ```

2. Generate background music or trim existing:
   ```bash
   npx mmx-cli music generate --prompt "cinematic pad" --instrumental --out music.mp3
   ffmpeg -y -i music.mp3 -t 9 -acodec copy music_9s.mp3
   ```

3. Add text with ImageMagick (more reliable than FFmpeg drawtext):
   ```bash
   magick frame.jpg \
     -fill white -font DejaVu-Sans-Bold -pointsize 64 \
     -stroke black -strokewidth 4 -gravity center \
     -annotate +0-50 "Your text here" \
     output.png
   ```

4. Compose with FFmpeg:
   Create per-phrase images, then concatenate to video and mux audio.
   Full script: `besoul-content-operations/scripts/build_reel.py`

## Known issues

- `mmx` binary is not in PATH after npm install; use `npx mmx-cli` instead
- `--api-key` passed inline in bash environment can leak via process listing;
  prefer config file auth
- Music model always generates 60-90s tracks regardless of prompts — trim after
