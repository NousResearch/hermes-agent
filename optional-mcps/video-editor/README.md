# Video Editor MCP Server

11 tools for conversational video editing via FFmpeg + faster-whisper.

## Tools

| Tool | Description |
|------|-------------|
| `video_probe` | Get video metadata via ffprobe |
| `video_transcribe` | Speech-to-text with word-level timestamps (faster-whisper) |
| `video_cut` | Extract video segments (stream copy or re-encode) |
| `video_overlay` | Overlay images with fade transitions |
| `video_captions` | Burn animated subtitles (SRT/ASS karaoke style) |
| `video_zoom` | Dynamic zoom/pan (Ken Burns effect) |
| `video_split_screen` | Multi-video layouts (side-by-side, grid, PiP) |
| `video_render` | Final composition with codec/quality settings |
| `video_transition` | Fade, crossfade, and wipe transitions |
| `video_audio_mix` | Audio mixing, normalization, ducking, replacement |
| `video_watermark` | Text or image watermarks |

## Requirements

- **FFmpeg** >= 6.0 (`ffmpeg` and `ffprobe` on PATH)
- **faster-whisper** for transcription (optional): `pip install faster-whisper`

## Installation

Install via the catalog:

```bash
hermes mcp install official/video-editor
```

Start a new Hermes session to load the video editor tools.

## Origin

Based on [PR #61293](https://github.com/NousResearch/hermes-agent/pull/61293) by @rafaumeu.
Re-scoped from core builtin tool to MCP catalog entry per AGENTS.md Footprint Ladder
guidance (level 5: MCP server, not level 6: core tool). All 11 tools preserved
with zero changes to the FFmpeg logic. The 3 tools that previously depended on
the non-existent `hermes_video_mcp` package (transition, audio_mix, watermark)
have been reimplemented as self-contained FFmpeg operations.
