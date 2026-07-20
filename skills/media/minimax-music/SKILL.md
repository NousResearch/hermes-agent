---
name: minimax-music
description: Generate songs and covers with the MiniMax Music API.
version: 1.0.0
author: "Kape (@kapelame), with Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [music, audio, generation, minimax, lyrics, songs]
    category: media
    related_skills: [songwriting-and-ai-music]
---

# MiniMax Music Skill

Generate songs, instrumental tracks, and music covers with the MiniMax Music
API. This skill sends one non-streaming request and saves the returned audio;
it does not edit, mix, or master existing audio.

## When to Use

- Generate a song from lyrics and a style prompt.
- Generate an instrumental track from a prompt.
- Create a cover from a reference audio URL, base64 audio, or a preprocessed
  cover feature ID.
- Save the result as MP3, WAV, or PCM audio.

## Prerequisites

- Use `MINIMAX_API_KEY` for the global endpoint.
- Use `MINIMAX_CN_API_KEY` with `--region cn` for the China endpoint.
- Python 3 and network access to the selected MiniMax API endpoint.
- A writable output path.

## How to Run

Use the `terminal` tool from the repository root:

```bash
python skills/media/minimax-music/scripts/generate_music.py \
  --prompt "Upbeat synth pop with a bright chorus" \
  --lyrics "[Verse]\nCity lights...\n[Chorus]\nWe rise..." \
  --output song.mp3
```

For instrumental music, omit lyrics and add `--instrumental`:

```bash
python skills/media/minimax-music/scripts/generate_music.py \
  --prompt "Ambient piano for quiet evening study" \
  --instrumental \
  --output instrumental.mp3
```

For a direct-audio cover, provide exactly one reference source:

```bash
python skills/media/minimax-music/scripts/generate_music.py \
  --model music-cover \
  --prompt "Acoustic folk with warm vocals" \
  --audio-url "https://example.com/reference.mp3" \
  --output cover.mp3
```

## Quick Reference

| Option | Values or behavior |
|---|---|
| `--model` | `music-3.0` by default; also supports `music-2.6`, free variants, and cover variants |
| `--region` | `global` by default; use `cn` for the China endpoint and credential |
| `--audio-format` | `mp3`, `wav`, or `pcm` |
| `--output-format` | `url` downloads the expiring URL; `hex` decodes inline audio |
| `--instrumental` | Generates without vocals |
| `--aigc-watermark` | Adds the China-region watermark field |
| `--audio-url` | Reference audio URL for a cover request |
| `--audio-base64` | Base64 reference audio for a cover request |
| `--cover-feature-id` | Preprocessed cover feature; requires `--lyrics` |

## Procedure

1. Choose `global` or `cn` and set the matching API key.
2. Choose a generation model for new music or a cover model for reference
   audio.
3. For a direct cover, pass exactly one of `--audio-url` or
   `--audio-base64`. Do not combine either one with `--cover-feature-id`.
4. When using `--cover-feature-id`, provide replacement lyrics with
   `--lyrics`.
5. Run the script with `terminal` and confirm the requested output file was
   created. URL results are downloaded immediately because they expire after
   24 hours.

## Pitfalls

- Global and China credentials are separate; `--region cn` never falls back
  to `MINIMAX_API_KEY`.
- Cover audio must be 6-360 seconds and no larger than 50 MB.
- Cover-only inputs are rejected for text-to-music models.
- `--aigc-watermark` is only sent to the China endpoint.
- Generated URL results expire after 24 hours.

## Verification

Run the focused tests:

```bash
scripts/run_tests.sh tests/skills/test_minimax_music_skill.py -q
```

For a live request, confirm the output file exists and can be decoded by the
expected audio player before returning it to the user.
