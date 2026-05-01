# Hackathon Build Status

**Last updated**: 1 May 2026 (late evening)  
**Deadline**: EOD Sunday 3 May 2026 (2 days remaining)

---

## Architecture

```
Telegram (trigger only)
  → Agent: transcribe (Whisper) + generate phonetics (Kimi K2.5, once)
  → Saves job JSON to ~/.hermes/caption-jobs/{id}.json
  → Burns initial draft video (FFmpeg)
  → Returns: draft video + "Edit at http://localhost:9119/captions/{id}"
        ↓
Dashboard /captions/:id  (no LLM, no tokens, visual editing)
  [video player]  |  [editable segment list: text / phonetic / EN-VI toggle]
                  |  [style panel: font, size, color, margin]
                  |  [Re-burn → FFmpeg only]  [Download]
```

---

## What We've Built

### 1. Core Video Captioning Tool — `tools/video_caption.py` ✅

> **Pivoted 1 May**: Original design was EN→VI translation. Correct use case is Vietnamese
> language *teaching* shorts — mixed EN/VI audio where the goal is phonetic guides for
> Vietnamese words, not translating English content.

A fully self-contained Hermes tool:

| Function | What it does |
|---|---|
| `transcribe(video_path)` | faster-whisper `medium` model, **auto language detect** (`language=None`), VAD filter — returns `{id, start, end, text, lang, phonetic}` segments |
| `generate_phonetics(segments, api_key)` | Single Kimi K2.5 call (NVIDIA NIM): classifies each segment as `"en"` or `"vi"`, corrects Whisper-garbled Vietnamese diacritics, generates English phonetic guide e.g. `[humm biet]` for VI segments |
| `build_ass(segments, output_path)` | ASS subtitle file with `MAIN` + `PHONETIC` styles: VI segments → Vietnamese text on top + italic phonetic below; EN segments → text only |
| `burn(video_path, ass_path, output_path)` | FFmpeg H.264 burn-in with fast preset |
| `_handle_caption(args)` | Dispatcher supporting 6 operations (see below) |

**Segment schema**: `{id, start, end, text, lang ("en"|"vi"), phonetic}`

**Caption layout on screen:**
```
Vietnamese segment:    không biết        ← MAIN style (bold, full size)
                       [humm biet]       ← PHONETIC style (italic, smaller, semi-transparent)

English segment:       Today we learn... ← MAIN style only
```

**Operations exposed to the agent:**
- `caption` — full pipeline (transcribe → generate_phonetics → build ASS → burn)
- `transcribe` — transcription only
- `generate_phonetics` — classify + correct + add phonetics via Kimi
- `build_ass` — ASS file generation only
- `burn` — burn a given ASS file into video
- `reburn` — apply corrected segments and re-burn (the edit loop)

**Kimi quirk handled**: Response sometimes lands in `reasoning_content` instead of `content` — fallback guard is in place.

**Tool registration**: Uses the standard `registry.register()` auto-discovery pattern — no manual import list needed.

---

### 2. Gateway Video Path Injection Fix — `gateway/run.py` ✅

Telegram caches received video files to disk, but the agent message previously contained no reference to the path — the agent had no way to know the file existed. Fixed by adding a video MIME type injection block (mirrors the existing document injection pattern):

```
[The user sent a video: 'name.mp4'. The file is saved at: /path/to/video.mp4.
You can process it with your available tools.]
```

This block is inserted in `_preprocess_inbound_text()` after the document block, before the reply-to block.

---

### 3. Toolset Registration — `toolsets.py` ✅

Added `video-caption` as an optional toolset (NOT in `_HERMES_CORE_TOOLS`) so it doesn't affect platforms that don't have faster-whisper installed:

```python
"video-caption": {
    "description": "Bilingual video captioning — ...",
  "tools": ["video-caption"],
    "includes": []
}
```

Enable per-instance via `~/.hermes/config.yaml`:
```yaml
toolsets:
  - hermes-cli
  - video-caption
```

---

### 4. Caption Style Config — `hermes_cli/config.py` ✅

Added a `caption` section to `DEFAULT_CONFIG` with fully documented style defaults:

```python
"caption": {
    "style": {
        "font": "Arial",
        "font_size": 48,
        "primary_color": "&H00FFFFFF",  # white  (ASS: &HAABBGGRR)
        "outline_color": "&H00000000",  # black
        "outline_width": 3,
        "alignment": 2,                 # bottom-center
        "margin_bottom": 80,
        "max_line_length": 42,
    }
}
```

Users can override any field in `~/.hermes/config.yaml`. The tool reads this at runtime.

---

### 5. Teaching Caption Skill — `skills/video/phonetic_captions/SKILL.md` ✅

Renamed (1 May) from `bilingual_captions` to `phonetic_captions` for clarity and scalability.
Updated to surface dashboard link after generation and explain visual editor workflow.
Chat correction loop retained as fallback for mobile/no-dashboard users.

---

### 6. Dashboard Visual Caption Editor ✅

> **Added 1 May (evening)**: Replaces Telegram chat as primary editing surface.
> Key insight: user wants to replace CapCut's visual editing workflow — chat corrections are slow, expensive, and imprecise for visual media.

**Backend** (`hermes_cli/web_server.py` — 7 new endpoints):

| Endpoint | Purpose |
|---|---|
| `GET /api/caption/jobs` | List all jobs |
| `GET /api/caption/jobs/{id}` | Get full job (segments, style, paths) |
| `PUT /api/caption/jobs/{id}/segments` | Save edited segments |
| `PUT /api/caption/jobs/{id}/style` | Save style changes |
| `POST /api/caption/jobs/{id}/burn` | Re-burn via FFmpeg (no LLM, runs in `asyncio.to_thread`) |
| `GET /api/caption/jobs/{id}/video` | Stream video for in-browser player |
| `GET /api/caption/jobs/{id}/download` | Download final output |

**Job persistence** (`tools/video_caption.py` — `save_caption_job()`):
- After `caption` op: saves job to `~/.hermes/caption-jobs/{id}.json`
- Response includes `job_id` + `dashboard_url` (e.g. `http://localhost:9119/captions/abc123`)

**Frontend** (`web/src/pages/CaptionEditorPage.tsx` — new file):
- Two-column layout: video player left, segment editor + style panel right
- Per-segment: text input, phonetic input (VI only), EN/VI badge toggle
- Style panel: font, font size, text/outline color pickers, margin
- Re-burn button (calls burn endpoint, auto-reloads video on completion)
- Download button (fetch with auth header, creates blob URL)

**Routes** (`web/src/App.tsx`):
- `/captions` → `CaptionJobsPage` (job list)
- `/captions/:id` → `CaptionEditorPage` (visual editor)
- "Captions" nav entry added to sidebar

---

### 7. Hackathon Plans — `hackathon-docs/` ✅

- `PLAN_v1.md` — original Telegram-native plan (executed, backend pipeline complete)
- `PLAN.md` — v2 revised plan (dashboard editor, current direction)

---

## Pending Tasks / What's Left

### P0 — Must have before demo

| Task | Notes |
|---|---|
| Install dependencies | `pip install faster-whisper openai` in `.venv` |
| End-to-end smoke test | Send a real short video via Telegram, verify: path injection fires → transcription runs → translation fires → video returned |
| Set `NVIDIA_API_KEY` | Add to `~/.hermes/.env` to enable Kimi translation |
| Enable toolset | Add `video-caption` to `toolsets` in `~/.hermes/config.yaml`: `toolsets: ["hermes-cli", "video-caption"]` |
| Load skill | Run `hermes skills install skills/video/phonetic-captions` or add to active skills |

### P1 — Important for demo quality

| Task | Notes |
|---|---|
| Font selection | "Arial" is a safe fallback but "Montserrat Bold" looks much better for Shorts — install the font or pick a system font that renders well at 48pt |
| Style tuning | Run against a real Shorts video and adjust `font_size`, `margin_bottom`, `alignment` to taste |
| Correction memory test | After approving captions, verify Hermes saves the corrections and auto-applies them on a second video |
| Demo video 1 script | 10–20s clip, introduce yourself/channel in English — use it to show the correction loop |
| Demo video 2 script | Same topic but different take — show that Hermes auto-applied Video 1 corrections |

### P2 — Nice to have / polish

| Task | Notes |
|---|---|
| Whisper model upgrade | Switch to `large-v3` for better accuracy (needs ~3GB RAM, slower on CPU — test on target machine) |
| Vietnamese font rendering | ASS with Vietnamese diacritics should work out of the box, but test `ơ ư ạ ề ọ` characters display correctly in output |
| Style slash command | `/caption-style` shortcut to show/edit current caption style — not needed for MVP |
| Progress feedback | faster-whisper can take 10–30s on CPU for 20s clips — add a "transcribing…" acknowledgement message before calling the tool |
| Segment merge heuristic | Very short segments (< 1s) can be merged with adjacent ones for cleaner captions |
| CLI mode | Verify the same workflow works from Hermes CLI (not just Telegram), useful as a fallback demo |

---

## Known Issues / Risks

| Issue | Severity | Status |
|---|---|---|
| `ffmpeg` path was stale | Fixed | Current binary now resolves to `/opt/homebrew/bin/ffmpeg` |
| Kimi `reasoning_content` quirk | Handled | Fallback guard in `generate_phonetics()` |
| faster-whisper not yet installed | P0 | `pip install faster-whisper` |
| CPU transcription speed | Medium | 20s video takes ~15–30s on CPU — acceptable for demo, mention it in video |
| Telegram 50MB video limit | Low risk | 10–20s Shorts are typically 5–25MB — should be fine |
| No automated tests for new tool | Low risk | Manual E2E smoke test covers the hackathon window |

---

## Setup Checklist (run this before first demo test)

```bash
# 1. Install Python dependencies
source .venv/bin/activate
pip install faster-whisper openai

# 2. Install ffmpeg (macOS)
brew install ffmpeg

# 3. Verify ffmpeg is on PATH
ffmpeg -version

# 3a. Confirm Hermes can load the tool module
python - <<'PY'
from tools.video_caption import _DEFAULT_STYLE
print('video_caption import OK')
print(_DEFAULT_STYLE['font'])
PY

# 4. Add NVIDIA API key
echo "NVIDIA_API_KEY=nvapi-YOUR_KEY_HERE" >> ~/.hermes/.env

# 5. Enable the toolset in config
# Add to ~/.hermes/config.yaml:
#   toolsets:
#     - hermes-cli
#     - video-caption

# 6. Install the skill
hermes skills install skills/video/phonetic-captions

# 7. Quick smoke test from CLI
python -c "
from tools.video_caption import transcribe, translate_to_vietnamese
print('Import OK')
"

# 8. First real test
# Run captioning against a 10-20s local video with spoken English and
# verify the tool returns: transcription, English+Vietnamese pairs, ASS file,
# and a burned output video.
```

---

## Testing Start Plan

### Immediate checks

1. Confirm `ffmpeg -version` works from the shell. This is already green.
2. Use Python 3.11+ for the repo. The current shell Python is 3.8.11, which blocks importing `tools.registry` because the codebase uses newer typing syntax.
3. Install `faster-whisper` in a 3.11 virtualenv and verify `tools.video_caption` imports.
4. Add `NVIDIA_API_KEY` if you want Vietnamese translation in the first run.
5. Run the first smoke test on a short local clip before trying Telegram.

### First smoke test

Use a 10-20 second video with clear English speech and no complicated music bed. The goal is to prove the full loop, not caption quality yet:

1. Transcription completes.
2. Kimi translation returns Vietnamese.
3. ASS output is written.
4. FFmpeg burns the subtitles into a new video.

### Next test after that

Send the same clip through Telegram so we can verify the gateway path injection and the end-to-end agent flow, not just the local tool.

---

## Submission Checklist

- [ ] ffmpeg installed and working
- [ ] Python 3.11+ virtualenv active
- [ ] faster-whisper + openai installed
- [ ] NVIDIA_API_KEY set in `~/.hermes/.env`
- [ ] `video-caption` toolset enabled
- [ ] `phonetic-captions` skill loaded
- [ ] Demo video 1 recorded: show video → captions → corrections → final output
- [ ] Demo video 2 recorded: show memory auto-applied from Video 1
- [ ] Kimi proof: show NVIDIA_API_KEY in config / API call in logs
- [ ] Tweet posted tagging @NousResearch
- [ ] Discord submission in `#creative-hackathon-submissions`
