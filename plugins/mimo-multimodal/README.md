# mimo-multimodal

Hermes Agent plugin for **image, audio, and video understanding** via Xiaomi MiMo multimodal models.

## Features

- 🖼️ **Image Analysis** — Description, OCR, explanation, information extraction
- 🎵 **Audio Analysis** — Transcription, description, summarization, translation
- 🎬 **Video Analysis** — Scene description, summarization, OCR, action analysis

### Advanced Features

- **Local file support** — Auto base64 encoding for files under 50MB
- **Large file support** — Temp HTTP server for video files up to 300MB
- **Smart fps tuning** — Auto-adjusts sampling rate based on video duration
- **Multi-image support** — Multiple images in single request
- **Format validation** — Magic bytes detection + extension whitelist
- **Audio extraction** — Extract audio track from video via ffmpeg
- **Token estimation** — Preview token usage before sending

## Requirements

- Xiaomi MiMo API Key ([Get one here](https://platform.xiaomimimo.com/))
- `ffmpeg` (optional, for audio extraction from video)

## Installation

This plugin is bundled with Hermes Agent. Enable it in your `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - mimo-multimodal
```

## Configuration

Set your Xiaomi MiMo API key in `~/.hermes/.env`:

```bash
XIAOMI_API_KEY=your_api_key_here
```

Or set the environment variable:

```bash
export XIAOMI_API_KEY=your_api_key_here
```

### Custom Base URL (optional)

If you need to use a different API endpoint:

```bash
XIAOMI_BASE_URL=https://token-plan-sgp.xiaomimimo.com/v1
```

## Usage

### Image Understanding

```python
# Basic description
image_understand(source="path/to/image.jpg")

# OCR (extract text)
image_understand(source="screenshot.png", prompt="ocr")

# Custom prompt
image_understand(source="photo.jpg", prompt="What brand logos are visible in this image?")
```

**Presets:** `describe`, `ocr`, `explain`, `compare`, `extract_info`, `caption`

**Supported formats:** JPEG, PNG, GIF, WebP, BMP

### Audio Understanding

```python
# Transcription
audio_understand(source="recording.mp3")

# Summarization
audio_understand(source="meeting.wav", prompt="summarize")

# Translation
audio_understand(source="speech.flac", prompt="translate")
```

**Presets:** `transcribe`, `describe`, `summarize`, `translate`, `extract_info`

**Supported formats:** MP3, WAV, FLAC, M4A, OGG, AAC, WMA

### Video Understanding

```python
# Basic description
video_understand(source="clip.mp4")

# Scene breakdown
video_understand(source="movie.mkv", prompt="scenes")

# With audio extraction
video_understand(source="interview.mp4", extract_audio=True)

# Custom fps
video_understand(source="surveillance.mp4", fps=0.5)
```

**Presets:** `describe`, `summarize`, `scenes`, `ocr`, `action`, `count`

**Supported formats:** MP4, MOV, AVI, WMV, MKV, WEBM, FLV

## Token Usage

The plugin automatically estimates token usage for video files:

| Duration | Auto fps | Estimated Tokens |
|----------|----------|------------------|
| < 10s    | 5.0      | ~3,000           |
| < 1min   | 2.0      | ~7,000           |
| < 5min   | 1.0      | ~35,000          |
| < 10min  | 0.5      | ~35,000          |
| 10min+   | 0.2      | ~14,000          |

## File Size Limits

| Type  | Limit | Fallback |
|-------|-------|----------|
| Image | 50MB  | Error    |
| Audio | 50MB  | Error    |
| Video | 50MB  | Temp HTTP server (up to 300MB) |

## Examples

### Analyze a screenshot

```python
result = image_understand(
    source="~/Desktop/screenshot.png",
    prompt="Extract all text and explain what this UI does"
)
```

### Transcribe a podcast

```python
result = audio_understand(
    source="https://example.com/podcast.mp3",
    prompt="transcribe"
)
```

### Analyze security footage

```python
result = video_understand(
    source="/recordings/camera1.mp4",
    prompt="Describe any people or vehicles that appear",
    fps=0.5,
    media_resolution="max"
)
```

## Author

**Everett** — [GitHub](https://github.com/xiaopengsvip)

## License

MIT
