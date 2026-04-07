---
name: x-spaces-transcript
category: media
description: Extract audio from X/Twitter Spaces and transcribe with Whisper for summarization.
tags: [twitter, spaces, audio, whisper, transcription, summarization]
related_skills: [youtube-content, xitter]
---

# X Spaces → Transcript

Extract, transcribe, and summarize X/Twitter Spaces recordings. X Spaces don't expose public transcripts, so this pipeline downloads the audio via yt-dlp and runs Whisper locally.

## Requirements

```bash
pip install yt-dlp openai-whisper
```

**Note:** `openai-whisper` is a separate package from `openai`. If `whisper` command is not found, install it explicitly — don't assume it's already present.

- **yt-dlp** — downloads Spaces audio from the HLS stream
- **openai-whisper** — local speech-to-text (runs on CPU, no API key needed)

## Recommended Workflow

1. Try `python3 ~/.hermes/scripts/fetch-url.py "URL"` first — it returns metadata (title, duration, host) even though X Spaces pages are behind login walls
2. If you need the actual content (transcript), fall back to the pipeline below
3. After transcription, synthesize into key points — don't just dump raw text

## Quick Start

### All-in-one script

```bash
python3 SKILL_DIR/scripts/fetch_spaces.py "https://x.com/i/spaces/1yxBeMYdqgnJN"
```

Options:
```bash
# Higher accuracy transcription
python3 SKILL_DIR/scripts/fetch_spaces.py "URL" --model small

# Plain text only (no metadata header)
python3 SKILL_DIR/scripts/fetch_spaces.py "URL" --text-only

# Save to file
python3 SKILL_DIR/scripts/fetch_spaces.py "URL" -o transcript.txt

# Just download audio, skip transcription
python3 SKILL_DIR/scripts/fetch_spaces.py "URL" --audio-only

# Keep the audio file after transcription
python3 SKILL_DIR/scripts/fetch_spaces.py "URL" --keep-audio
```

### Manual steps

If you want more control, run each step yourself:

```bash
# Step 1: Download audio
cd /tmp && yt-dlp -x --audio-format mp3 \
  -o "spaces_audio.%(ext)s" "https://x.com/i/spaces/..."

# Step 2: Transcribe
whisper /tmp/spaces_audio.mp3 --model base --language en \
  --output_format txt --output_dir /tmp
```

## Whisper Model Selection

| Model   | Size   | Speed (35 min audio) | Quality | Use When |
|---------|--------|---------------------|---------|----------|
| tiny    | 39 MB  | ~5s                 | Low     | Quick proof-of-concept |
| base    | 74 MB  | ~15s                | Good    | Default — good enough for summaries |
| small   | 244 MB | ~45s                | Better  | Accurate quotes needed |
| medium  | 769 MB | ~2 min              | High    | Multi-speaker, accents |
| large   | 1.5 GB | ~5 min              | Best    | Final deliverable quality |

All models run on CPU (FP32). No GPU required. A "FP16 not supported" warning is normal.

## Summarization Workflow

After getting the transcript, synthesize it. Don't just dump it — organize by themes:

1. **Key points** — the actionable insights, numbered
2. **Speaker context** — who they are, why it matters
3. **Quotes** — notable direct quotes worth preserving
4. **Hot takes** — contrarian or opinionated statements

Save the summary to Obsidian if the user has a vault configured (use `obsidian-nicks-mind-map-filing` skill for routing).

## URL Formats

The pipeline handles any X Spaces URL:
- `https://x.com/i/spaces/SPACE_ID`
- `https://x.com/i/spaces/SPACE_ID?s=20` (with tracking params)

## Gotchas

- **Download timeout**: 30+ minute Spaces need `timeout=300`. The default 60s is not enough — the download will abort mid-stream with no usable file. HLS chunked downloads run at ~23x speed but total elapsed is ~90s for a 35-min Space. If you get a timeout at 60s, just retry with the longer timeout — the second attempt will succeed.

- **False error on download**: yt-dlp may report `Unable to download video: [Errno 2] No such file or directory: 'spaces_audio.m4a.part'` — this is a file rename issue. The `.mp3` file will still exist. Always verify with `ls -la /tmp/spaces_audio*`.

- **Verbose output**: The download logs every HLS chunk (~700 lines for a 35-min Space). The actual download is fast despite the noise. Filter stderr if needed.

- **Login wall**: X Spaces pages are behind a login wall. `web_extract` and `fetch-url.py` only return metadata (title, duration). Audio download is the only way to get content.

- **No speaker diarization**: Whisper doesn't identify who's speaking. For multi-speaker Spaces, you'll need to manually note speaker transitions or use a diarization tool.

- **Language**: Default is English. Pass `--language` for other languages. Whisper auto-detects if unsure, but explicit language is more reliable.

## Security

The script enforces several security controls:

- **URL validation** — Only `https://x.com/i/spaces/<ID>` and `https://twitter.com/i/spaces/<ID>` are accepted. Rejects `file://`, other domains, and malformed paths.
- **SSL verification** — yt-dlp runs with default SSL verification (no `--no-check-certificates`). Downloads are integrity-checked.
- **No shell injection** — All subprocess calls use list form (no `shell=True`). Arguments cannot escape into shell interpretation.
- **Metadata sanitization** — Title/uploader/duration from X's API are stripped of control characters and capped at 200 chars before output.
- **Transcript fallback removed** — Whisper output is matched by exact filename only, preventing accidental reads of unrelated `.txt` files in the work directory.
- **Symlink protection** — Output file path is resolved and checked for symlinks before writing, preventing overwrite attacks.
- **Temp dir cleanup** — Auto-created temp directories are cleaned up via `finally` block, even on errors.

## Example Output

```
# Event ROI Reality Check
URL: https://x.com/i/spaces/1yxBeMYdqgnJN
Duration: 34:51
Host: ipshi86

---

[transcript text here]
```
