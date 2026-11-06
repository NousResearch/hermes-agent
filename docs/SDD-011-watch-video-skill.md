# SDD-011: Watch Video Skill (Ollama Vision) — System Design Document

Version: 1.0.0
Status: Draft for review

## Objective

Define the internal architecture, component boundaries, data flow, and file structure for the Hermes Watch Video Skill with Ollama vision integration.

---

## Component Boundaries

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HERMES SKILL: watch-video                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │  CLI Entry   │───▶│  Pipeline    │───▶│  Vision      │───▶│  Notes   │  │
│  │  (slash cmd) │    │  Orchestrator│    │  Analyzer    │    │  Writer  │  │
│  └──────────────┘    └──────┬───────┘    └──────┬───────┘    └──────────┘  │
│                             │                   │                           │
│                    ┌────────┴────────┐          │                           │
│                    ▼                 ▼          ▼                           │
│            ┌─────────────┐    ┌─────────────┐  ┌─────────────┐             │
│            │ Video       │    │ Frame       │  │ Ollama      │             │
│            │ Downloader  │    │ Extractor   │  │ Client      │             │
│            │ (yt-dlp)    │    │ (ffmpeg)    │  │ (qwen3-vl)  │             │
│            └─────────────┘    └─────────────┘  └─────────────┘             │
│                    │                 │                                      │
│                    └────────┬────────┘                                      │
│                             ▼                                               │
│                    ┌─────────────────┐                                     │
│                    │ Temp Workspace  │                                     │
│                    │ Manager         │                                     │
│                    └─────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Key Interfaces |
|-----------|---------------|----------------|
| **CLI Entry** | Parse slash command args, validate input, invoke pipeline | `main(args: List[str]) -> int` |
| **Pipeline Orchestrator** | Coordinate stages, manage temp workspace, handle errors | `run(url, options) -> NotesResult` |
| **Video Downloader** | Download video, extract captions via yt-dlp | `download(url, work_dir) -> VideoAsset` |
| **Frame Extractor** | Extract frames at adaptive intervals via ffmpeg | `extract_frames(video_path, options) -> List[Frame]` |
| **Ollama Client** | Communicate with Ollama `/api/chat` for vision analysis | `analyze_frames(frames, prompt) -> VisionResult` |
| **Notes Writer** | Generate structured markdown from transcript + vision | `write_notes(transcript, vision, meta) -> Path` |
| **Temp Workspace Manager** | Create/cleanup temp directory, track all temp files | `create() -> Path`, `cleanup()` |

---

## Data Flow

```
INPUT: /watch-video <url> [--start S] [--end E] [--max-frames N] [--resolution W] [--fps F]

┌────────────────────────────────────────────────────────────────────────────┐
│ 1. VALIDATE & SETUP                                                         │
│    - Parse CLI args → PipelineOptions                                       │
│    - Create temp workspace: ~/.hermes/cache/watch-video/<uuid>/            │
│    - Verify Ollama reachable + vision model supports vision                │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 2. DOWNLOAD & CAPTION EXTRACTION (VideoDownloader)                         │
│    - yt-dlp --write-sub --sub-format vtt --skip-download (captions only)  │
│    - If captions exist: parse VTT → TranscriptSegment[]                    │
│    - yt-dlp -f bestvideo+bestaudio --output work_dir/video.%(ext)s        │
│    - Output: VideoAsset { path, title, duration, captions: Transcript[] }  │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 3. FRAME EXTRACTION (FrameExtractor)                                       │
│    - Calculate interval: adaptive based on duration                        │
│      duration < 5min  → 1 fps (≈300 frames max, capped at 100)            │
│      5-30min        → 0.2 fps (1 frame/5s)                                 │
│      30-60min       → 0.1 fps (1 frame/10s)                                │
│      >60min         → 0.05 fps (1 frame/20s), capped at 100               │
│    - Apply --start/--end window if specified (focused mode)               │
│    - ffmpeg -i video -vf "fps=X,scale=W:-1" -q:v 2 frames/frame_%04d.jpg  │
│    - Output: Frame[] { path, timestamp, index, width, height }             │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 4. VISION ANALYSIS (OllamaClient)                                          │
│    - Batch frames: groups of 5-8 frames per request                        │
│    - Build prompt with frame timestamps + analysis instructions            │
│    - POST http://localhost:11434/api/chat with images as base64            │
│    - Model: qwen3-vl:4b (configurable)                                     │
│    - Timeout: 120s per batch (configurable)                                │
│    - Retry: 1 retry on timeout/empty response                              │
│    - Output: VisionResult[] { frame_index, timestamp, description,         │
│                                ocr_text, ui_elements, scene_type }         │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 5. TIMELINE MERGE (PipelineOrchestrator)                                   │
│    - If captions exist: align vision results to transcript timestamps      │
│    - If no captions: use frame timestamps as primary timeline              │
│    - Merge into unified TimelineEntry[] { timestamp, transcript?, visual? }│
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 6. NOTES GENERATION (NotesWriter)                                          │
│    - LLM prompt (can use same Ollama model or Hermes main model) to       │
│      generate: Summary, TL;DR, Key Quotes from timeline                   │
│    - Render markdown template with all sections                            │
│    - Write to ~/.hermes/watch-video-notes/<video-id>-<ts>.md              │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 7. CLEANUP & OUTPUT                                                        │
│    - TempWorkspaceManager.cleanup() — remove all temp files                │
│    - Print notes file path to stdout                                       │
│    - Return exit code 0 on success, non-zero on failure                    │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

```python
# PipelineOptions (CLI parsed)
@dataclass
class PipelineOptions:
    url: str                           # YouTube URL or local file path
    start: Optional[str] = None        # HH:MM:SS or seconds
    end: Optional[str] = None          # HH:MM:SS or seconds
    max_frames: int = 80               # Cap at 100
    resolution: int = 512              # Frame width in pixels
    fps: Optional[float] = None        # Override adaptive fps (max 2)
    whisper_backend: str = "none"      # "none" | "groq" | "openai" (ignored, kept for compat)
    no_whisper: bool = True            # Always true for this skill
    out_dir: Optional[str] = None      # Override output directory
    ollama_url: str = "http://localhost:11434"
    vision_model: str = "qwen3-vl:4b"
    vision_timeout: int = 120          # Seconds per batch
    vision_batch_size: int = 8         # Frames per vision request

# VideoAsset (downloader output)
@dataclass
class VideoAsset:
    path: Path                         # Local video file
    title: str                         # Video title
    duration: float                    # Duration in seconds
    captions: List[TranscriptSegment]  # Empty if no captions
    metadata: dict                     # yt-dlp raw metadata

@dataclass
class TranscriptSegment:
    start: float                       # Start time in seconds
    end: float                         # End time in seconds
    text: str                          # Caption text

# Frame (extractor output)
@dataclass
class Frame:
    path: Path                         # Frame image file
    timestamp: float                   # Timestamp in seconds
    index: int                         # Frame sequence number
    width: int
    height: int

# VisionResult (Ollama output per frame)
@dataclass
class VisionResult:
    frame_index: int
    timestamp: float
    description: str                   # Visual description
    ocr_text: str                      # Any visible text (OCR)
    ui_elements: List[str]             # Buttons, menus, charts, etc.
    scene_type: str                    # "talking_head", "screen_demo", "slides", "b_roll", etc.
    speaker_change: bool               # Detected speaker change

# TimelineEntry (merged timeline)
@dataclass
class TimelineEntry:
    timestamp: float
    transcript: Optional[str] = None   # From captions
    visual: Optional[VisionResult] = None

# NotesResult (final output)
@dataclass
class NotesResult:
    notes_path: Path
    title: str
    summary: str
    tldr: List[str]
    timeline: List[TimelineEntry]
    key_quotes: List[Tuple[float, str]]  # (timestamp, quote)
    visual_notes: List[VisionResult]
```

---

## File Structure

```
skills/productivity/watch-video/
├── SKILL.md                           # Skill manifest (frontmatter + docs)
├── scripts/
│   ├── __init__.py
│   ├── setup.py                       # Dependency check + config wizard
│   ├── download.py                    # VideoDownloader CLI
│   ├── extract_frames.py              # FrameExtractor CLI
│   ├── vision_analyze.py              # OllamaClient CLI
│   ├── merge_timeline.py              # Timeline merger CLI
│   ├── write_notes.py                 # NotesWriter CLI
│   ├── pipeline.py                    # PipelineOrchestrator (main entry)
│   └── models.py                      # Shared dataclasses
├── references/
│   ├── ollama-vision-prompting.md     # Prompt engineering guide
│   ├── frame-extraction-guide.md      # ffmpeg parameters reference
│   └── yt-dlp-captions.md             # Caption extraction patterns
├── templates/
│   └── notes_template.md              # Markdown template for output
├── tests/
│   ├── test_download.py
│   ├── test_extract_frames.py
│   ├── test_vision_analyze.py
│   ├── test_merge_timeline.py
│   ├── test_write_notes.py
│   └── test_pipeline.py
└── .env.example                       # Config template (OLLAMA_BASE_URL, VISION_MODEL)
```

---

## CLI Script Interfaces

Each script is a standalone argparse CLI with JSON I/O for testability and composability.

### `scripts/pipeline.py` (Main Entry)
```bash
python scripts/pipeline.py <url> \
  [--start HH:MM:SS] [--end HH:MM:SS] \
  [--max-frames 80] [--resolution 512] [--fps 1.0] \
  [--ollama-url http://localhost:11434] \
  [--vision-model qwen3-vl:4b] \
  [--vision-timeout 120] \
  [--vision-batch-size 8] \
  [--out-dir ~/.hermes/watch-video-notes] \
  [--work-dir ~/.hermes/cache/watch-video] \
  [--json]  # Output result as JSON to stdout
```
Exit codes: 0=success, 1=download failed, 2=frame extraction failed, 3=vision failed, 4=write failed, 5=cleanup failed

### `scripts/download.py`
```bash
python scripts/download.py <url> --work-dir <dir> --json
# Output: {"video_path": "...", "title": "...", "duration": 123.4, "captions": [...], "metadata": {...}}
```

### `scripts/extract_frames.py`
```bash
python scripts/extract_frames.py <video_path> --work-dir <dir> \
  [--start 30] [--end 90] [--max-frames 80] [--resolution 512] [--fps 1.0] --json
# Output: {"frames": [{"path": "...", "timestamp": 30.0, "index": 0, "width": 512, "height": 288}, ...]}
```

### `scripts/vision_analyze.py`
```bash
python scripts/vision_analyze.py --frames frames.json \
  --ollama-url http://localhost:11434 --vision-model qwen3-vl:4b \
  --timeout 120 --batch-size 8 --json
# Output: {"results": [{"frame_index": 0, "timestamp": 30.0, "description": "...", "ocr_text": "", ...}, ...]}
```

### `scripts/merge_timeline.py`
```bash
python scripts/merge_timeline.py --transcript transcript.json --vision vision.json --json
# Output: {"timeline": [{"timestamp": 30.0, "transcript": "...", "visual": {...}}, ...]}
```

### `scripts/write_notes.py`
```bash
python scripts/write_notes.py --timeline timeline.json --meta meta.json --out-dir <dir> --json
# Output: {"notes_path": "/path/to/notes.md", "title": "...", "summary": "...", "tldr": [...], ...}
```

---

## Ollama Vision Prompt Strategy

### Batch Prompt Template
```
Analyze these {N} video frames with timestamps. For each frame, provide:
1. Visual description (2-3 sentences)
2. Any visible text (OCR) — UI text, code, slides, captions on screen
3. UI elements detected (buttons, menus, charts, code editor, terminal, etc.)
4. Scene type: talking_head | screen_demo | slides | b_roll | diagram | other
5. Speaker change detected? (yes/no)

Frames:
Frame 0 [00:30.0]: <base64_image>
Frame 1 [00:35.0]: <base64_image>
...
Frame N [MM:SS.mmm]: <base64_image>

Output as JSON array matching frame order:
[
  {"frame_index": 0, "description": "...", "ocr_text": "...", "ui_elements": ["..."], "scene_type": "...", "speaker_change": false},
  ...
]
```

### Notes Generation Prompt
```
Given this video timeline (transcript segments + visual analysis), generate:
1. One-line summary (single sentence)
2. TL;DR (3-5 bullet points)
3. Key quotes (verbatim from transcript, with timestamps)
4. Visual notes summary (key visual moments with timestamps)

Timeline:
{timeline_json}

Output as JSON:
{"summary": "...", "tldr": ["...", "..."], "key_quotes": [[timestamp, "quote"], ...], "visual_highlights": [[timestamp, "description"], ...]}
```

---

## Error Handling & Resilience

| Failure Point | Strategy |
|---------------|----------|
| yt-dlp download fails | Exit 1, print yt-dlp error, cleanup temp |
| No captions + no vision model | Exit 3, clear error "No captions and vision model unavailable" |
| Ollama unreachable | Exit 3, print connection error |
| Vision model no vision support | Exit 3, print model capability error |
| Vision timeout | Retry once, then skip batch (log warning), continue |
| Vision empty response | Retry once, then skip batch |
| ffmpeg frame extraction fails | Exit 2, print ffmpeg error |
| Disk full / permission denied | Exit 5, cleanup what's possible |
| User interrupt (Ctrl+C) | Cleanup temp, exit 130 |

---

## Configuration

### Skill Config (in `config.yaml` under `skills.config.watch-video`)
```yaml
skills:
  config:
    watch-video:
      ollama_url: "http://localhost:11434"
      vision_model: "qwen3-vl:4b"
      vision_timeout: 120
      vision_batch_size: 8
      default_max_frames: 80
      default_resolution: 512
      output_dir: "~/.hermes/watch-video-notes"
      cache_dir: "~/.hermes/cache/watch-video"
```

### Environment Variables (`.env` or shell)
```bash
OLLAMA_BASE_URL=http://localhost:11434
WATCH_VIDEO_VISION_MODEL=qwen3-vl:4b
WATCH_VIDEO_OUTPUT_DIR=~/.hermes/watch-video-notes
```

---

## Testing Strategy

### Unit Tests (mocked)
- `test_download.py`: Mock yt-dlp subprocess, test caption parsing, error handling
- `test_extract_frames.py`: Mock ffprobe/ffmpeg, test adaptive fps calculation, focused mode
- `test_vision_analyze.py`: Mock Ollama HTTP, test batching, prompt building, timeout/retry
- `test_merge_timeline.py`: Test caption+vision alignment, vision-only timeline
- `test_write_notes.py`: Test markdown rendering, template substitution
- `test_pipeline.py`: Integration test with mocked subprocesses

### Integration Tests (requires local Ollama)
- `@pytest.mark.integration` tests that hit real Ollama + yt-dlp
- Run manually: `scripts/run_tests.sh tests/skills/test_watch_video_skill.py::test_e2e_short_video -v`

---

## Security Considerations

- **No external API calls** except yt-dlp (video download) and Ollama (local)
- **Temp files**: Created in `~/.hermes/cache/watch-video/` with restricted permissions (0o700)
- **Input validation**: URLs validated via yt-dlp; local paths resolved and checked for traversal
- **No secrets**: No API keys required; Ollama runs locally without auth by default
- **Cleanup**: Guaranteed via `try/finally` in pipeline orchestrator

---

## Performance Targets

| Metric | Target |
|--------|--------|
| 5-min video (captions + 30 frames) | < 60s end-to-end |
| 30-min video (captions + 80 frames) | < 180s end-to-end |
| Vision batch (8 frames) | < 15s on M-series Mac |
| Memory usage | < 2GB peak |
| Disk temp usage | < 500MB (frames + video) |

---

## Future Extensibility

- **Model swapping**: Configurable `vision_model` allows upgrading to qwen2.5-vl, llava, etc.
- **Multi-model pipeline**: Could add separate audio transcription model (e.g., whisper.cpp via Ollama)
- **Streaming output**: Could emit timeline entries incrementally for long videos
- **Cache vision results**: Hash frame content + model + prompt → cache vision output