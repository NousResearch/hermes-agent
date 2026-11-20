# IDD-011: Watch Video Skill (Ollama Vision) — Intent Driven Design

Version: 1.0.0
Status: Draft for review

## Objective

Define the INTENT behind each component interface — what each component is designed to achieve, the contracts it fulfills, and the invariants it maintains. This is not an API listing (that's SDD), but the design intent: WHY this interface exists and WHAT guarantees it provides.

---

## Component Interface Intents

### 1. Pipeline Orchestrator (`scripts/pipeline.py`)

**Intent**: Provide a single, deterministic entry point that coordinates the entire video-to-notes pipeline with guaranteed cleanup and clear failure semantics.

**Contract Guarantees**:
- **Atomic pipeline execution**: Either all stages complete and a notes file is produced, or no notes file is produced and temp workspace is cleaned up.
- **Deterministic failure modes**: Each failure maps to a specific exit code (1-5) with actionable error message — no ambiguous "something went wrong".
- **Progress visibility**: Long-running stages emit structured progress to stderr (JSON lines) so callers can surface progress to users.
- **Config precedence**: CLI args > skill config > env vars > hardcoded defaults — documented and testable.

**Invariants**:
- Temp workspace created before any stage runs, cleaned up in `finally` block — *never* leaked on success, failure, or interrupt.
- All subprocess calls use explicit args list (no shell=True) — prevents injection.
- Input URL/path validated before any network/disk I/O.

**Why this interface exists**: The upstream Claude skill used a single monolithic script. Breaking into orchestrated stages enables: independent testing, parallel development, selective re-run of stages, and clear observability. The orchestrator is the *only* component that knows the full pipeline order — stages are ignorant of each other.

---

### 2. Video Downloader (`scripts/download.py`)

**Intent**: Extract maximum metadata and captions from a video source with minimal download, using yt-dlp as the sole extraction engine.

**Contract Guarantees**:
- **Caption-first extraction**: Attempts caption-only download (`--skip-download`) before full download — saves bandwidth/time when captions suffice.
- **Format normalization**: Output captions as `TranscriptSegment[]` with float timestamps (seconds) — independent of source format (VTT, SRT, TTML, etc.).
- **Metadata completeness**: Returns yt-dlp's full `info_dict` in `metadata` field — callers can access thumbnail, uploader, chapters, etc. without re-downloading.
- **Local file support**: If input is a local path, skips yt-dlp entirely, probes with `ffprobe` for duration, returns empty captions.

**Invariants**:
- Never writes video file if `--skip-download` succeeds (captions only).
- Video file always written to `{work_dir}/video.{ext}` — predictable path for frame extractor.
- Duration always > 0 (validated via ffprobe if yt-dlp metadata missing).

**Why this interface exists**: yt-dlp is the only maintained universal video extractor. Wrapping it isolates the pipeline from yt-dlp's CLI changes, output format quirks, and site-specific extractor updates. The `VideoAsset` dataclass is the stable contract — yt-dlp can change internally without breaking downstream.

---

### 3. Frame Extractor (`scripts/extract_frames.py`)

**Intent**: Produce a uniformly spaced, resolution-normalized frame sequence that optimally represents visual content within token/cost budgets.

**Contract Guarantees**:
- **Adaptive interval policy**: Frame rate automatically scales with video duration to maintain ~30-100 frames — short videos get dense coverage, long videos get representative sampling.
- **Focused mode fidelity**: When `--start/--end` specified, frame density increases in the target window (same total frame budget concentrated).
- **Resolution normalization**: All frames scaled to target width (default 512px) maintaining aspect ratio — ensures consistent vision model input size.
- **Timestamp precision**: Frame timestamps accurate to ffmpeg's frame boundary (~1/fps seconds) — matches transcript alignment needs.

**Invariants**:
- Frame count ≤ `max_frames` (hard cap 100) — *never* exceeded regardless of duration/fps settings.
- Frames named sequentially: `frame_{index:04d}.jpg` — enables deterministic ordering.
- Output JSON includes `width`/`height` per frame — vision prompt can reference actual dimensions.

**Why this interface exists**: The upstream skill hardcoded frame extraction logic. Separating it enables: unit testing interval logic without ffmpeg, swapping extraction backend (e.g., PyAV), and independent optimization of frame sampling strategy. The adaptive policy is a *policy decision* — not an implementation detail — and belongs in the interface contract.

---

### 4. Ollama Vision Client (`scripts/vision_analyze.py`)

**Intent**: Provide a reliable, batched interface to local Ollama vision models with automatic capability verification, timeout management, and structured output parsing.

**Contract Guarantees**:
- **Capability verification on init**: Calls `/api/show` to confirm model supports vision *before* processing any frames — fails fast with clear error if model lacks vision.
- **Batching with ordering guarantee**: Frames processed in batches but results returned in original frame order — `frame_index` preserved throughout.
- **Structured output enforcement**: Prompt engineered to elicit valid JSON array matching `VisionResult[]` schema — parsing validates required fields.
- **Timeout isolation**: Per-batch timeout (default 120s) — one slow batch doesn't block others; failed batches logged and skipped, pipeline continues.

**Invariants**:
- Ollama URL validated reachable before first batch — no "connection refused" mid-pipeline.
- Model name passed verbatim to Ollama — no alias resolution, no fallback (explicit config only).
- Base64 image encoding uses standard data URL format (`data:image/jpeg;base64,..."`) — compatible with Ollama's multimodal API.
- Temperature fixed at 0.1 for deterministic visual analysis — not configurable (reduces variance).

**Why this interface exists**: Direct HTTP to Ollama avoids dependency on `ollama-python` SDK (which adds bloat and version coupling). The batching strategy is critical for cost/latency — sending 80 frames individually would take 10x longer. The capability check prevents the "model loads but returns garbage" failure mode common with misconfigured vision models.

---

### 5. Timeline Merger (`scripts/merge_timeline.py`)

**Intent**: Fuse transcript segments (from captions) and vision results (from frames) into a single chronologically ordered timeline, preserving provenance of each piece.

**Contract Guarantees**:
- **Provenance preservation**: Every `TimelineEntry` knows whether its content came from captions, vision, or both — no information loss during merge.
- **Timestamp alignment**: When both sources exist for same timestamp, they're merged into one entry — not duplicated.
- **Gap handling**: Timeline covers full video duration — gaps in either source represented as `None` (not omitted).
- **Deterministic ordering**: Sorted by timestamp ascending, then by source priority (transcript first, then vision).

**Invariants**:
- Output timeline length = unique timestamps across both sources (union, not sum).
- If captions empty, timeline driven entirely by frame timestamps.
- If frames empty (extraction failed), timeline driven entirely by captions (with visual=null).

**Why this interface exists**: The upstream skill merged these implicitly in the LLM prompt. Explicit merger enables: deterministic output (same inputs → same timeline), independent testing of alignment logic, and the ability to swap LLM-based merge for rule-based merge. The `TimelineEntry` is the *single source of truth* for downstream notes generation.

---

### 6. Notes Writer (`scripts/write_notes.py`)

**Intent**: Transform the unified timeline into a structured, human-readable markdown document following a consistent template.

**Contract Guarantees**:
- **Template-driven**: Output structure defined by `templates/notes_template.md` — changing format requires only template edit, not code change.
- **LLM-assisted synthesis**: Summary, TL;DR, key quotes generated by LLM (configurable: local Ollama or Hermes main model) — not rule-based extraction.
- **Provenance in output**: Every timeline reference in notes includes timestamp — reader can trace back to source.
- **File naming convention**: `{video_id}-{timestamp}.md` — sortable, unique, identifies source.

**Invariants**:
- All 6 sections present in output (Title, Summary, TL;DR, Timeline, Key Quotes, Visual Notes) — *never* omitted.
- Timeline section renders every `TimelineEntry` — no filtering/summarization at write time.
- Output directory created if missing — `mkdir -p` semantics.

**Why this interface exists**: The upstream skill generated notes via a single massive prompt. Separating synthesis (LLM) from rendering (template) enables: testing rendering without LLM, swapping synthesis model, and maintaining consistent format across model versions. The template is the *contract* for output format.

---

### 7. Temp Workspace Manager (embedded in pipeline)

**Intent**: Guarantee temporary file isolation and cleanup — no orphaned files regardless of failure mode.

**Contract Guarantees**:
- **Unique workspace per run**: Directory name includes UUID — concurrent runs never collide.
- **Atomic cleanup**: `cleanup()` removes entire workspace tree — no selective file deletion logic to get wrong.
- **Failure transparency**: Cleanup runs in `finally` — succeeds even if pipeline crashes mid-stage.

**Invariants**:
- Workspace path always under `~/.hermes/cache/watch-video/` — respects Hermes cache convention.
- Permissions 0o700 — other users cannot access temp video/frames.
- No symlinks followed — prevents escape from workspace.

**Why this interface exists**: The upstream skill used `tempfile.mkdtemp()` with manual cleanup. A dedicated manager makes cleanup *unavoidable* (enforced by orchestrator's `try/finally`) and enables testing cleanup behavior independently.

---

## Cross-Cutting Interface Intents

### CLI Argument Contract (All Scripts)

**Intent**: Every script is a standalone CLI with consistent conventions — composable, testable, scriptable.

**Guarantees**:
- `--json` flag on every script: output machine-readable JSON to stdout, human logs to stderr.
- Exit codes: 0=success, 1-255=failure with documented meanings (per-script).
- No required positional args without `--help` — running bare shows usage.
- Path args accept both relative and absolute — resolved to absolute internally.

**Why**: Enables chaining scripts in shell, testing via subprocess, and debugging individual stages without running full pipeline.

### Configuration Contract

**Intent**: Configuration is explicit, layered, and validated at startup — no silent fallbacks to wrong values.

**Guarantees**:
- Config sources: CLI > skill config (config.yaml) > env vars > `.env` file > defaults.
- Validation at pipeline start: all required configs present, types correct, values in range.
- Unknown config keys rejected (strict schema) — prevents typos silently ignored.

**Why**: The upstream skill used scattered env vars and hardcoded values. Centralized config with validation prevents "works on my machine" due to implicit env.

---

## Data Flow Invariants (End-to-End)

| Invariant | Where Enforced |
|-----------|----------------|
| Frame timestamps monotonic increasing | FrameExtractor output validation |
| Timeline entries sorted by timestamp | TimelineMerger output validation |
| Notes file contains all 6 sections | NotesWriter template contract |
| Temp workspace deleted on exit | PipelineOrchestrator `finally` block |
| No external network calls except yt-dlp + Ollama localhost | Static analysis (grep for http/https in scripts) |
| Vision model capability verified before use | VisionClient `__init__` |
| Input URL validated before download | PipelineOrchestrator `validate_input()` |

---

## Extension Points (Designed for Future Change)

| Extension Point | Current Implementation | Future Swap |
|-----------------|------------------------|-------------|
| Video downloader | yt-dlp subprocess | Direct yt-dlp Python API, or site-specific extractors |
| Frame extractor | ffmpeg subprocess | PyAV, decord, or GPU-accelerated extraction |
| Vision model | Ollama qwen3-vl:4b | Any Ollama vision model, or local llama.cpp server, or OpenAI-compatible endpoint |
| Notes synthesis | LLM (Ollama or Hermes main) | Rule-based extraction, or different prompt strategy |
| Output format | Markdown template | JSON, HTML, PDF, Notion API, etc. |

Each extension point is isolated behind its script interface — changing implementation requires *only* modifying that script, not the orchestrator or other stages.

---

## Non-Goals (Explicitly Out of Scope)

- **Real-time streaming analysis**: Pipeline is batch-oriented; not designed for live video.
- **Multi-video comparison**: Single video per invocation; comparison is a separate skill.
- **Video editing/clipping**: Read-only analysis; no output video generation.
- **Cloud vision APIs**: Explicitly excluded per user requirement (fully local).
- **Speaker diarization**: Vision model may detect speaker changes visually, but no audio diarization.
- **Translation**: Output language matches input; no translation step.

---

## Verification Checklist

Before implementation, verify each interface intent is satisfied:

- [ ] PipelineOrchestrator: Atomic execution, deterministic exit codes, progress JSON lines
- [ ] VideoDownloader: Caption-first, format normalization, metadata completeness, local file support
- [ ] FrameExtractor: Adaptive intervals, focused mode, resolution normalization, timestamp precision
- [ ] VisionClient: Capability verification, batch ordering, structured output, timeout isolation
- [ ] TimelineMerger: Provenance preservation, timestamp alignment, gap handling, deterministic ordering
- [ ] NotesWriter: Template-driven, LLM-assisted synthesis, provenance in output, file naming
- [ ] TempWorkspace: Unique per run, atomic cleanup, failure transparency
- [ ] Cross-cutting: CLI `--json`, config layering, config validation