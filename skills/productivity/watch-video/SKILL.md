---
name: watch-video
description: "Watch videos via local Ollama vision model — extract transcript, frames, and structured notes."
version: 0.1.0
author: Warren Ledingham, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [video, youtube, transcription, vision, ollama, analysis, notes]
    category: productivity
    related_skills: [pdf, document-to-action-items, web-extract]
---

# Watch Video Skill

Analyze videos by extracting captions (via yt-dlp) and visual frames (via ffmpeg), then process frames through a local Ollama vision model (qwen3-vl:4b) to generate structured markdown notes with timeline, key quotes, and visual descriptions. Fully offline — no external API keys required.

## When to Use

- "Watch this YouTube video and give me notes"
- "Extract the key points from this tutorial video"
- "Analyze a video without captions using local vision"
- "Create study notes from a lecture recording"
- "Get visual descriptions from a demo/screencast video"

Don't use for: real-time video streaming, multi-video comparison, video editing, or cloud-based transcription.

## Prerequisites

- **ffmpeg + ffprobe**: `brew install ffmpeg` / `winget install Gyan.FFmpeg` / `apt install ffmpeg`
- **yt-dlp**: `brew install yt-dlp` / `pipx install yt-dlp` / `winget install yt-dlp.yt-dlp` (must be on PATH)
- **Python 3.9+**: Available as `python` (or `python3` on macOS/Linux)
- **Ollama with vision model**: `ollama pull qwen3-vl:4b` (or other vision-capable model)
  - Verify: `curl http://localhost:11434/api/show -d '{"name":"qwen3-vl:4b"}' | grep -q vision`

Run the setup wizard to verify dependencies and create config:

```bash
python scripts/setup.py
```

It creates `~/.hermes/skills/config/watch-video.yaml` with defaults.

## How to Run

After installation, invoke via slash command:

```bash
/watch-video https://www.youtube.com/watch?v=VIDEO_ID
/watch-video /path/to/local/video.mp4
/watch-video https://youtu.be/abc123 --start 2:00 --end 5:00
/watch-video https://www.youtube.com/watch?v=... --max-frames 50 --resolution 768
```

Flags:
- `--start T` / `--end T`: Focus on a time window (HH:MM:SS or seconds)
- `--max-frames N`: Cap frame count (default 80, hard max 100)
- `--resolution W`: Frame width in pixels (default 512)
- `--fps F`: Override adaptive fps (capped at 2)
- `--out-dir DIR`: Output directory for notes (default `~/.hermes/watch-video-notes`)
- `--ollama-url URL`: Ollama base URL (default `http://localhost:11434`)
- `--vision-model NAME`: Vision model name (default `qwen3-vl:4b`)

The skill writes a markdown notes file and prints its path to stdout.

## Quick Reference

| Task | Script | Command |
|------|--------|---------|
| Full pipeline | `pipeline.py` | `python scripts/pipeline.py <url> [options]` |
| Download + captions | `download.py` | `python scripts/download.py <url> --work-dir <dir> --json` |
| Extract frames | `extract_frames.py` | `python scripts/extract_frames.py <video> --work-dir <dir> --json` |
| Vision analysis | `vision_analyze.py` | `python scripts/vision_analyze.py --frames frames.json --json` |
| Merge timeline | `merge_timeline.py` | `python scripts/merge_timeline.py --transcript t.json --vision v.json --json` |
| Write notes | `write_notes.py` | `python scripts/write_notes.py --timeline tl.json --meta m.json --json` |

## Procedure

### 1. Install and Verify

```bash
# Install skill (from local path or GitHub URL)
hermes skills install /path/to/watch-video

# Verify installation
hermes skills list | grep watch-video
hermes -c "/watch-video --help"

# Run setup wizard (checks ffmpeg, yt-dlp, Ollama, vision model)
python scripts/setup.py
```

### 2. Analyze a Video

```bash
# Basic usage — YouTube video with captions
hermes -c "/watch-video https://www.youtube.com/watch?v=jNQXAC9IVRw"

# Local file, focused window
hermes -c "/watch-video ~/Videos/demo.mp4 --start 0:30 --end 2:00"

# Custom frame budget for long video
hermes -c "/watch-video https://youtu.be/longvideo --max-frames 100 --resolution 768"
```

### 3. Read the Notes

```bash
# Notes saved to ~/.hermes/watch-video-notes/<video-id>-<timestamp>.md
cat ~/.hermes/watch-video-notes/*.md | head -80
```

### 4. Advanced: Run Pipeline Stages Individually

```bash
# Download only
python scripts/download.py "https://youtu.be/..." --work-dir /tmp/wv-test --json

# Extract frames from downloaded video
python scripts/extract_frames.py /tmp/wv-test/video.mp4 --work-dir /tmp/wv-test --max-frames 20 --json

# Analyze frames with Ollama
python scripts/vision_analyze.py --frames /tmp/wv-test/frames.json --json

# Merge and write notes
python scripts/merge_timeline.py --transcript /tmp/wv-test/transcript.json --vision /tmp/wv-test/vision.json --json
python scripts/write_notes.py --timeline /tmp/wv-test/timeline.json --meta /tmp/wv-test/meta.json --out-dir ~/.hermes/watch-video-notes --json
```

## Pitfalls

- **No captions + no vision model**: If video has no captions AND Ollama vision model unavailable, pipeline fails with clear error. Install qwen3-vl:4b or ensure captions exist.
- **Ollama not running**: Skill fails fast with connection error. Start Ollama: `ollama serve`.
- **Wrong vision model**: If configured model lacks vision capability, skill fails at startup. Use `ollama list` and verify `capabilities` includes `vision`.
- **Large videos**: Videos > 60 min get sparse frames (adaptive policy). Use `--start/--end` for dense coverage of specific sections.
- **Disk space**: Temp workspace stores video + frames (~500MB for 30 min video). Cleanup runs automatically but verify disk space for long videos.
- **yt-dlp failures**: Age-gated, region-locked, or members-only content may fail. yt-dlp prints clear error — not a skill bug.
- **Windows Python**: Microsoft Store `python3` stub doesn't run scripts. Use python.org installer (`python` on PATH).

## Verification

- [ ] `hermes skills install` succeeds and registers `/watch-video` slash command
- [ ] Setup wizard validates all dependencies (ffmpeg, yt-dlp, Ollama, vision model)
- [ ] YouTube video with captions → notes with transcript + visual notes
- [ ] Local video file → notes with vision-only timeline
- [ ] Focused mode (`--start/--end`) increases frame density in window
- [ ] Frame count respects `--max-frames` cap (≤100)
- [ ] Temp workspace cleaned up on success and failure (check `~/.hermes/cache/watch-video/`)
- [ ] Notes file contains all 6 sections: Title, Summary, TL;DR, Timeline, Key Quotes, Visual Notes
- [ ] Exit codes: 0=success, 1=download, 2=frames, 3=vision, 4=write, 5=cleanup
- [ ] Tests pass: `scripts/run_tests.sh tests/skills/test_watch_video_skill.py -q`