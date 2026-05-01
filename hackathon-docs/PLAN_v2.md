# Hermes Caption Agent — Plan v2 (Dashboard Visual Editor)

**Hackathon**: Hermes Agent Creative Hackathon  
**Prize pool**: $25k (Main $15k + Kimi Track $5k)  
**Deadline**: EOD Sunday May 3rd, 2026  
**Previous plan**: `PLAN_v1.md` (Telegram-native, executed — backend pipeline complete)

---

## Problem Statement (Unchanged)

Content creators making bilingual (English + Vietnamese) videos spend 30–45 minutes per video manually choosing font, size, color, position in CapCut, typing Vietnamese captions with no consistency, and fixing translation mistakes with no memory of prior corrections.

**User quote**: *"I was hoping that the tool can auto generate the caption for both English and Vietnamese with the same font, size and color and where it appears on the video with the same style that I'm doing in each video right now. I understand that the Vietnamese part will be tricky and that the auto caption will have many mistakes, which I can definitely fix it."*

---

## Revised Architecture

The key insight: **the user wants to replace CapCut's visual editing workflow**, not chat their way through corrections. The LLM should run exactly once per video; all editing should be fast, visual, and free.

```
Telegram (trigger only)
  → Agent: transcribe (Whisper) + generate phonetics (Kimi K2.5, once)
  → Saves job JSON to ~/.hermes/caption-jobs/{id}.json
  → Burns initial draft video (FFmpeg)
  → Returns: draft video + "Edit at http://localhost:9119/captions/{id}"
        ↓
Dashboard /captions/:id  (no LLM, no tokens, visual editing)
  ┌─────────────────────┬─────────────────────────────────────┐
  │   <video> player    │  Segment list (editable)            │
  │   (auto-reloads     │  ┌──────────────────────────────┐   │
  │    after re-burn)   │  │ EN  Today we're learning...  │   │
  │                     │  │ VI  không biết  [humm biet]  │   │
  │                     │  └──────────────────────────────┘   │
  │                     │                                     │
  │                     │  Style panel                        │
  │                     │  Font / Size / Color / Margin       │
  │                     │                                     │
  │                     │  [Re-burn]  [Download]              │
  └─────────────────────┴─────────────────────────────────────┘
```

**What doesn't change**: the backend pipeline (Whisper + Kimi + ASS + FFmpeg) is complete and reused as-is.

---

## Why Dashboard Instead of Telegram Chat

| Concern | Telegram Chat | Dashboard Editor |
|---|---|---|
| Caption text editing | Prompt per edit → LLM round-trip (~5s, tokens) | Direct text input → instant |
| Style changes (font/size/color) | Natural language → lossy → re-prompt | Form controls → exact values |
| Visual feedback | Download video to preview | Inline video player |
| Per-edit cost | Tokens + inference time | $0, 0ms latency |
| Scalability (other languages) | Works but slow | Works, same UI |

Telegram remains valuable as the **trigger surface** — creators already have it open, video upload is natural, and they don't need to start a browser to kick off processing.

---

## Implementation Phases

### Phase 1 — Job persistence in `tools/video_caption.py` (~1h)

**Goal**: After generating captions, persist job state to disk for the dashboard to read.

**Changes to `tools/video_caption.py`**:
1. Add `save_caption_job(job_id, video_path, segments, style, output_path)` — writes `{hermes_home}/caption-jobs/{id}.json`
2. Update `_handle_caption()`: after full `caption` operation, call `save_caption_job`, include `job_id` and `dashboard_url` in response JSON

**Job JSON schema**:
```json
{
  "id": "uuid4",
  "created_at": "ISO8601",
  "video_path": "/abs/path/input.mp4",
  "output_path": "/abs/path/output.mp4",
  "style": { "font": "Arial", "font_size": 48, ... },
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.3, "text": "Today we learn", "lang": "en", "phonetic": ""},
    {"id": 1, "start": 2.3, "end": 4.1, "text": "không biết", "lang": "vi", "phonetic": "[humm biet]"}
  ]
}
```

### Phase 2 — Backend API in `hermes_cli/web_server.py` (~2h)

**New endpoints** (all require `_require_token()` auth — follows existing pattern):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/caption/jobs` | List all jobs (id, created_at, video filename) |
| `GET` | `/api/caption/jobs/{id}` | Full job state (segments, style, paths) |
| `PUT` | `/api/caption/jobs/{id}/segments` | Save edited segments → write to job JSON |
| `PUT` | `/api/caption/jobs/{id}/style` | Save style changes → write to job JSON |
| `POST` | `/api/caption/jobs/{id}/burn` | Re-burn: `_build_ass_content()` + `burn()` from `video_caption.py`, update `output_path` in job JSON |
| `GET` | `/api/caption/jobs/{id}/video` | Stream video file for in-browser `<video>` player |
| `GET` | `/api/caption/jobs/{id}/download` | Download final output with `Content-Disposition: attachment` |

**Implementation note**: Import `_build_ass_content`, `burn` directly from `tools.video_caption` — no subprocess, no LLM.

### Phase 3 — Dashboard React page (~4-5h)

**New file**: `web/src/pages/CaptionEditorPage.tsx`

**Route**: `/captions/:id` — add to `BUILTIN_ROUTES_CORE` in `web/src/App.tsx`

**Layout (two-column)**:
```
Left (40%):
  - Native <video> tag with controls
  - src = /api/caption/jobs/{id}/video?t={burnTimestamp}  (cache-busting)
  - Auto-reloads after re-burn completes

Right (60%):
  - Segment list (scrollable):
      Each row: [EN/VI badge] [text input] [phonetic input, greyed if EN]
  - Style panel (collapsible):
      Font family (text), Font size (number), Primary color (color picker),
      Outline color (color picker), Margin bottom (slider)
  - Action bar (sticky bottom):
      [Re-burn] button — POST /api/caption/jobs/{id}/burn, poll until 200,
                         then reload video player
      [Download] button — GET /api/caption/jobs/{id}/download
```

**State management**: `useState` for `segments` and `style`; optimistic local edits; flush on Re-burn.

**Nav entry**: Add "Captions" tab to `BUILTIN_NAV_REST` in `App.tsx` with `Captions` label and appropriate icon.

Also add `/captions` route listing all jobs (links to `/captions/:id`).

### Phase 4 — Skill update (~30min)

**Update `skills/video/phonetic_captions/SKILL.md`**:
- After `caption` op completes, agent should present:
  - The draft video
  - Numbered segment summary (EN/VI + phonetic)
  - Dashboard link: "Open the caption editor to make visual edits: [link]"
  - Offer to make quick corrections via chat (for users on mobile without dashboard access)

---

## Files Changed / Created

| File | Action | Phase |
|---|---|---|
| `tools/video_caption.py` | Modify — add `save_caption_job()`, update `_handle_caption()` | 1 |
| `hermes_cli/web_server.py` | Modify — add 7 new `/api/caption/...` endpoints | 2 |
| `web/src/pages/CaptionEditorPage.tsx` | Create — visual editor | 3 |
| `web/src/App.tsx` | Modify — add routes + nav entry | 3 |
| `skills/video/phonetic_captions/SKILL.md` | Modify — surface dashboard link | 4 |

---

## Demo Story (Updated)

**Meet Linh — she makes bilingual cooking content.**

1. Opens Telegram, sends today's video (15s clip)
2. Hermes replies in ~30s: captioned draft video + link to `localhost:9119/captions/abc123`
3. Linh clicks the link — caption editor opens in her browser
4. She sees the segment table — fixes "không biết" phonetic, changes font size from 48 to 52
5. Clicks "Re-burn" — video updates in 5s (FFmpeg only, no LLM)
6. Downloads final video

**Second video (next week):** Hermes already has her style saved. Draft comes back with her font, size, and colour. Zero style changes needed.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Dashboard not running when Telegram link is sent | Include fallback note: "Or reply here to edit via chat" |
| Video CORS for `<video>` player | Serve via FastAPI with correct `Content-Type` header + range support |
| Re-burn blocking the FastAPI event loop | Run `burn()` in `asyncio.to_thread()` (FFmpeg subprocess) |
| Job JSON grows large for long videos | Cap at 500 segments; segment merging is a V2 feature |

---

## Kimi Track Strategy (Unchanged)

Submission video will show:
1. `NVIDIA_API_KEY` + `moonshotai/kimi-k2.5` config
2. Side-by-side: raw Whisper output vs Kimi-corrected Vietnamese + phonetics
3. Dashboard editor showing manual refinement on top of Kimi output

---

## Out of Scope (V2)

- Real-time caption preview (re-render on every keystroke)
- Drag-and-drop timeline editor
- Word-level Whisper segmentation splitting
- Multi-video batch management
- Mobile-responsive dashboard layout
