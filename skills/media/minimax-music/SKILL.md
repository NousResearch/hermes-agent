---
name: minimax-music
description: "Generate songs and instrumental music with the MiniMax Music API."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [music, audio, generation, minimax, lyrics, songs]
---

# MiniMax Music Generation

Use this skill when the user wants to generate a song, instrumental track, or
music cover with MiniMax. Set `MINIMAX_API_KEY`, then call the bundled script.

```bash
python skills/media/minimax-music/scripts/generate_music.py \
  --prompt "Upbeat synth pop with a bright chorus" \
  --lyrics "[Verse]\n..." \
  --output song.mp3
```

The default model is `music-3.0`. Generation models are `music-3.0`,
`music-2.6`, `music-3.0-free`, and `music-2.6-free`. Cover models are
`music-cover` and `music-cover-free`; pass either `--audio-url` or
`--audio-base64` for a cover request (6–360 seconds, at most 50 MB).

Use `--region cn` for the China endpoint and `--aigc-watermark` when required.
The global endpoint is used by default. `--instrumental` omits vocals;
`--audio-format` accepts `mp3`, `wav`, or `pcm`. The script requests a URL by
default and downloads it immediately because generated URLs expire after 24
hours. Use `--output-format hex` when the API must return audio bytes inline.

