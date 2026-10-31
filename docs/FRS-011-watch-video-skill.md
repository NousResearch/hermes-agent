# FRS-011: Watch Video Skill (Ollama Vision Integration)

Version: 1.0.0
Status: Draft for review

## Objective

Port the Claude "watch-video-skill" to Hermes as an installable skill with `/watch-video` slash command. Replace the Whisper API transcription fallback with local Ollama vision model (qwen3-vl:4b) for frame analysis and audio transcription via video frame understanding. The skill extracts transcripts from video captions (via yt-dlp) and supplements with vision-based frame analysis using the local qwen3-vl:4b model — no external API keys required.

---

## FRS-1: Video Ingestion & Transcript Extraction

### Objective
Download video/audio from URL or local path, extract existing captions/subtitles via yt-dlp, and prepare frames for vision analysis.

### Functional Requirements

**FR-1.1** — The skill SHALL accept a video URL (YouTube, Vimeo, TikTok, X, Twitch, Loom, Instagram, etc.) or local file path (mp4, mov, mkv, webm, avi) as input via slash command `/watch-video <url|path> [--start HH:MM:SS] [--end HH:MM:SS]`.

**FR-1.2** — The skill SHALL use yt-dlp to download the video/audio and extract available captions/subtitles in VTT/SRT format. If captions exist, they SHALL be parsed into a timestamped transcript.

**FR-1.3** — The skill SHALL extract frames from the video at adaptive intervals: ~1 frame/second for videos < 5 min, ~1 frame/5 seconds for 5-30 min, ~1 frame/10 seconds for 30-60 min, capped at 100 frames maximum (matching upstream claude-video behavior).

**FR-1.4** — The skill SHALL store frames as JPEG/PNG files in a temporary working directory (auto-cleaned on completion).

**FR-1.5** — The skill SHALL support `--start` and `--end` flags to focus on a specific time window (focused mode), packing more frames into that window.

**FR-1.6** — The skill SHALL support `--max-frames N` (default 80, hard max 100), `--resolution W` (default 512px width), `--fps F` (capped at 2 fps) configuration flags.

### Acceptance Criteria

- [ ] Given a YouTube URL, the skill downloads video and extracts captions when available
- [ ] Given a local video file, the skill extracts frames at adaptive intervals
- [ ] Frame count never exceeds 100 frames
- [ ] Focused mode (--start/--end) increases frame density in the target window
- [ ] Temporary files are cleaned up on completion (success or failure)

### Verification

```bash
# Test with a short YouTube video with captions
hermes -c "/watch-video https://www.youtube.com/watch?v=jNQXAC9IVRw"

# Test with local file
hermes -c "/watch-video /path/to/video.mp4 --start 0:30 --end 1:00"

# Verify output markdown file exists with transcript + frame analysis
```

---

## FRS-2: Ollama Vision Frame Analysis

### Objective
Use local Ollama qwen3-vl:4b model to analyze extracted frames and generate visual descriptions, supplementing or replacing transcript gaps.

### Functional Requirements

**FR-2.1** — The skill SHALL connect to local Ollama at `http://localhost:11434` (configurable via `OLLAMA_BASE_URL` env or config) and use the `qwen3-vl:4b` model (configurable via skill config `vision_model`).

**FR-2.2** — The skill SHALL verify the configured model supports vision via Ollama `/api/show` endpoint (checking `capabilities` includes "vision" or `model_info.*.vision.block_count` exists) before processing.

**FR-2.3** — For each extracted frame, the skill SHALL send the frame image + a structured prompt to the vision model requesting: visual description, any visible text (OCR), UI elements, charts/diagrams, speaker changes, scene transitions.

**FR-2.4** — The skill SHALL batch frames for efficiency: process frames in batches of 5-10 with a single prompt asking for timestamped visual notes for each frame.

**FR-2.5** — The skill SHALL match vision analysis output to transcript timestamps (when captions exist) or frame timestamps (when no captions), producing a unified timeline.

**FR-2.6** — The skill SHALL handle vision model timeouts gracefully (configurable timeout, default 120s per batch) and continue with remaining frames.

**FR-2.7** — The skill SHALL work fully offline — no external API calls (Groq, OpenAI, etc.) required.

### Acceptance Criteria

- [ ] Given frames + Ollama qwen3-vl:4b, the skill produces visual descriptions for each frame
- [ ] Vision output is timestamp-aligned with transcript or frame timestamps
- [ ] Batch processing works (multiple frames per vision call)
- [ ] Timeout on one batch doesn't fail the entire pipeline
- [ ] No external network calls beyond Ollama localhost and yt-dlp download

### Verification

```bash
# Verify Ollama vision model detection
curl -s http://localhost:11434/api/show -d '{"name": "qwen3-vl:4b"}' | grep -q vision && echo "Vision supported"

# Test frame analysis on a known video
hermes -c "/watch-video https://www.youtube.com/watch?v=jNQXAC9IVRw --max-frames 10"
```

---

## FRS-3: Structured Markdown Notes Output

### Objective
Produce a structured markdown notes file matching the upstream skill's output format: Title, TL;DR, Timeline, Key Quotes, Visual Notes.

### Functional Requirements

**FR-3.1** — The skill SHALL write a markdown file to `~/.hermes/watch-video-notes/<video-id>-<timestamp>.md` (configurable output directory).

**FR-3.2** — The notes file SHALL contain:
- **Title**: Video title + source URL
- **One-line Summary**: Single sentence capturing the video's core message
- **TL;DR**: 3-5 bullet points of key takeaways
- **Timeline**: Chronological sections with timestamps, combining transcript segments + visual notes
- **Key Quotes**: Verbatim quotes with timestamps (from transcript when available)
- **Visual Notes**: Frame-by-frame visual descriptions with timestamps

**FR-3.3** — When captions exist: timeline primarily driven by transcript, visual notes supplement.
**FR-3.4** — When no captions: timeline driven by frame timestamps + vision analysis, with note "Transcript: none available (vision-only analysis)".

**FR-3.5** — The skill SHALL output the notes file path to stdout for chaining.

**FR-3.6** — The skill SHALL support `--out-dir DIR` to override output directory.

### Acceptance Criteria

- [ ] Markdown file created with all 6 sections
- [ ] Timeline entries have timestamps (HH:MM:SS format)
- [ ] Visual notes reference frame numbers/timestamps
- [ ] File path printed to stdout
- [ ] Works with both captioned and non-captioned videos

### Verification

```bash
# Check output structure
cat ~/.hermes/watch-video-notes/*.md | head -100
```

---

## FRS-4: Slash Command Integration & Hermes Skill Packaging

### Objective
Package as a proper Hermes skill installable via `hermes skills install` with slash command `/watch-video`.

### Functional Requirements

**FR-4.1** — The skill SHALL have a valid `SKILL.md` with frontmatter: name, description (≤60 chars), version, author, license, platforms [linux, macos, windows], metadata.hermes.tags, metadata.hermes.category.

**FR-4.2** — The skill SHALL register a slash command `/watch-video` via Hermes skill command registration (ctx.register_cli_command).

**FR-4.3** — The skill SHALL include helper scripts in `scripts/` directory (Python CLIs with argparse, JSON I/O).

**FR-4.4** — The skill SHALL declare prerequisites in SKILL.md: ffmpeg, yt-dlp, Python 3.9+, Ollama with vision model.

**FR-4.5** — The skill SHALL include a setup wizard script (`scripts/setup.py`) that checks dependencies and creates config template.

**FR-4.6** — The skill SHALL be installable from a GitHub URL or local path via `hermes skills install <url|path>`.

### Acceptance Criteria

- [ ] `hermes skills install <path-to-skill>` succeeds
- [ ] `/watch-video` appears in `hermes skills list` and slash command help
- [ ] Running `/watch-video <url>` executes the pipeline
- [ ] Setup wizard validates ffmpeg, yt-dlp, Ollama, vision model

### Verification

```bash
# Install skill
hermes skills install /path/to/watch-video-skill

# Verify installation
hermes skills list | grep watch-video
hermes -c "/watch-video --help"
```

---

## Dependencies Between Items

```
FRS-1 (Video Ingestion) ─────▶ FRS-2 (Vision Analysis) ─────▶ FRS-3 (Notes Output)
       │                            │                            │
       └──────────────▶ FRS-4 (Skill Packaging & CLI)
```

FRS-4 can be developed in parallel but requires FRS-1/2/3 scripts to exist.

---

## Testing Strategy

| Capability | Test Level | File | Coverage |
|-----------|-----------|------|----------|
| FRS-1 | Unit + Integration | `tests/skills/test_watch_video_skill.py` | FR-1.1 through FR-1.6 |
| FRS-2 | Unit + Integration | `tests/skills/test_watch_video_skill.py` | FR-2.1 through FR-2.7 |
| FRS-3 | Unit | `tests/skills/test_watch_video_skill.py` | FR-3.1 through FR-3.6 |
| FRS-4 | E2E | `tests/skills/test_watch_video_skill.py` | FR-4.1 through FR-4.6 |

All tests use stdlib + pytest + unittest.mock only. No live network except local Ollama.

---

## Boundaries

### Always do
- Run tests before commits (`scripts/run_tests.sh tests/skills/test_watch_video_skill.py -q`)
- Follow Hermes skill authoring standards (description ≤60 chars, platforms gating, script references)
- Validate Ollama vision capability before processing
- Clean up temporary files on success and failure

### Ask first
- Changing the vision model default (qwen3-vl:4b)
- Adding external API fallbacks (Groq, OpenAI)
- Modifying frame extraction algorithm (adaptive intervals)

### Never do
- Make external API calls for transcription (no Groq, OpenAI Whisper)
- Hardcode paths like `/tmp` — use `tempfile.gettempdir()`
- Skip dependency checks in setup wizard
- Leave temporary files on disk after completion

---

## Success Criteria (Overall)

- [ ] All FRS items have passing tests
- [ ] All acceptance criteria are verified
- [ ] Skill installs via `hermes skills install` and registers `/watch-video` slash command
- [ ] End-to-end pipeline works: URL → download → extract frames → Ollama vision analysis → markdown notes
- [ ] Works fully offline (Ollama + yt-dlp only external calls)

---

## Open Questions

1. **Frame batching prompt design**: What's the optimal prompt structure for sending 5-10 frames in one vision call with timestamp alignment?
2. **Audio transcription via vision**: Can qwen3-vl:4b transcribe audio from video frames, or should we use a separate approach for non-captioned videos?
3. **Config file location**: Should skill config live at `~/.hermes/skills/config/watch-video.yaml` or inline in main config.yaml under `skills.config.watch-video`?
4. **Progress reporting**: Should the slash command emit progress updates via Hermes streaming, or just final result?

---

## Resolved Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Transcription fallback | Ollama vision (qwen3-vl:4b) only | User requirement: no Whisper API, fully local |
| Vision model | qwen3-vl:4b (configurable) | Already installed, confirmed vision capability |
| Frame extraction | Adaptive intervals, max 100 frames | Matches upstream claude-video v2 behavior |
| Output format | Markdown with 6 sections | Matches upstream skill for compatibility |
| Installation | Hermes skill (official or local) | User requirement: `hermes skills install` |
| Slash command | `/watch-video` | Matches upstream skill command |