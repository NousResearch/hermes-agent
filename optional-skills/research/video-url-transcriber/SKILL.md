---
name: video-url-transcriber
description: Universal media URL transcriber using yt-dlp + ffmpeg + faster-whisper. Converts public video/audio URLs (not just YouTube) into timestamped JSON transcript segments.
version: 1.0.0
platforms: [macos, linux]
metadata:
  hermes:
    tags: [media, transcription, asr, yt-dlp, whisper]
    category: research
---

# Video URL Transcriber

Universal URL-to-transcript skill for Hermes.

This skill transcribes **public media URLs** from any platform currently supported by `yt-dlp`.
It is **not limited to YouTube**.

Pipeline:
`URL -> yt-dlp fetch -> ffmpeg normalize (mono 16k) -> faster-whisper ASR -> timestamped JSON`

## Requirements

System dependencies:

```bash
# macOS
brew install ffmpeg yt-dlp

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg yt-dlp
```

Python dependencies:

```bash
pip install yt-dlp faster-whisper
```

## Quick Usage

Dependency check:

```bash
python3 SKILL_DIR/scripts/transcribe_url.py --doctor
```

Transcribe a URL:

```bash
python3 SKILL_DIR/scripts/transcribe_url.py "https://youtu.be/VIDEO_ID"
```

Higher quality model:

```bash
python3 SKILL_DIR/scripts/transcribe_url.py "<url>" --model-size medium
```

Language hint:

```bash
python3 SKILL_DIR/scripts/transcribe_url.py "<url>" --language en
```

## Output Contract

The script returns JSON with:

- `source_url`
- `platform`
- `title`
- `duration_sec`
- `language`
- `transcript`
- `segments[]` with timestamps and optional word-level timing
- `status`

## Privacy + Keychain Policy

Default behavior never reads browser cookies or keychain data.

If and only if the user explicitly asks to use personal browser session data for an auth-gated URL, run:

```bash
python3 SKILL_DIR/scripts/transcribe_url.py "<url>" --cookies-from-browser chrome --allow-personal-cookies
```

Without `--allow-personal-cookies`, cookie usage is rejected.

## When to use this skill vs youtube-content

Use `video-url-transcriber` when:
- URL is not YouTube
- YouTube captions are missing/disabled
- You need audio-derived transcript quality independent of platform caption APIs

Use `youtube-content` when:
- It is YouTube-only and you want fast caption retrieval from existing subtitles

## Failure handling

If transcription fails, report exact stage:
- media fetch (`yt-dlp`)
- audio normalize (`ffmpeg`)
- ASR (`faster-whisper`)
