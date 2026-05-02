# Hermes Caption Agent — Plan v5 (3-Column Editor + Named Preset Library)

**Hackathon**: Hermes Agent Creative Hackathon  
**Prize pool**: $25k (Main $15k + Kimi Track $5k)  
**Deadline**: EOD Sunday May 3rd, 2026  
**Previous plans**:
- `PLAN_v1.md` — Telegram-native pipeline (executed)
- `PLAN_v2.md` — Dashboard core-edit approach (archived)
- `PLAN_v3.md` — Plugin architecture, editing + burn (executed)
- `PLAN_v4.md` — Hermes-integrated features: upload, NL edits, QA, style memory (executed)

---

## What Changed from v4

Plan v4 shipped a fully Hermes-integrated editor: upload, NL segment editing, QA review, and cross-session style learning. The style interaction was still file-based (load/save `.json`) and the UI had two columns — the third panel (Hermes AI features) was bolted onto the right column alongside the segment list and style fields.

Plan v5 makes **Hermes a first-class UI surface** with its own dedicated panel, and replaces the file-based style presets with a **server-side named preset library** that Hermes can create from natural language.

---

## New Capabilities

### 1. Three-Column Editor Layout

The editor is reorganised into three distinct panels:

```
┌─────────────────┬──────────────────────────┬─────────────────────┐
│  Col 1 (360px)  │  Col 2 (flex)            │  Col 3 (380px)      │
│                 │                          │                     │
│  Video player   │  Segments                │  ✦ Hermes           │
│                 │  [Re-burn] [Download]    │                     │
│                 │  ─────────────────────   │  ── Edit segments ──│
│                 │  segment cards           │  [instruction input]│
│                 │  (amber border if flagged│  [diff list]        │
│                 │                          │                     │
│                 │  ── Caption Style ──     │  ── QA ─────────────│
│                 │  font / size / color /   │  [Review all]       │
│                 │  outline / margin fields │  flag list          │
│                 │                          │                     │
│                 │                          │  ── Style presets ──│
│                 │                          │  [preset cards]     │
│                 │                          │  [Save current]     │
│                 │                          │  [Create with AI]   │
│                 │                          │  [Learned card]     │
└─────────────────┴──────────────────────────┴─────────────────────┘
```

**Responsive**: on screens narrower than 1280px (xl breakpoint) col 3 wraps below col 2 — no drawer needed.

**Col 1** — Video + actions:
- Video player (unchanged)
- Re-burn and Download buttons (moved from the old right column header)

**Col 2** — Data editing:
- Segment list with per-segment controls (text, phonetic, EN/VI toggle, split ✂)
- Amber left border on QA-flagged segments (visual anchor — details stay in col 3)
- Caption Style section: inline font / size / color / outline / margin fields

**Col 3 — ✦ Hermes panel** (new, branded):
- Three labelled sections:
  1. **Edit segments** — NL instruction input + AI patch diff list
  2. **QA** — "Review" button + flag list (clicking a flag scrolls to that segment)
  3. **Style presets** — named preset gallery + "Save current" + "Create with AI" + Learned card

---

### 2. Named Preset Library

Replaces the old file-based Load/Save buttons with a server-side named preset store.

**Storage**: `~/.hermes/caption-presets/{name}.json` (one file per preset, sorted by mtime).

**Preset gallery** in the Hermes panel:
- Horizontal-scroll row of named cards — click any card to apply that style instantly
- Each card has a ✕ delete button
- "Save current" button → inline name input → `PUT /presets/{name}` → card appears in gallery
- **Learned card** (amber border, ✦ icon): surfaced from the `GET /style/suggestion` MemoryStore analysis; not auto-saved; has its own "Save as preset" + "Apply" actions
- All-empty state: just "Save current" and the AI creation input

**New backend endpoints** (4):

| Endpoint | Purpose |
|---|---|
| `GET /presets` | List all named presets (`[{name, style}]`) |
| `PUT /presets/{name}` | Create or overwrite a named preset |
| `DELETE /presets/{name}` | Remove a named preset |
| `POST /presets/generate` | NL description → `AIAgent` → `CaptionStyle` (not saved automatically) |

---

### 3. AI Style Creation ("Create with AI")

Users describe a visual look in plain English; Hermes returns a `CaptionStyle` object for inspection and optional save.

**Flow:**
1. Expand "Create with AI" in the style presets section
2. Type a description, e.g. *"bold Impact font, yellow text, thick black outline — TikTok style"*
3. `POST /presets/generate` → `AIAgent` (1 iteration) returns `CaptionStyle` JSON
4. A preview card appears showing the 8 style fields with a name input and "Save" button
5. User names and saves → `PUT /presets/{name}` → card added to gallery
6. User can also "Apply" without saving (applies to local state only)

The agent never saves automatically — the user always names and approves.

---

### 4. Hermes Panel UX Notes

- Section labels use a muted separator style (not heavy headers) to keep the panel scannable
- QA flag list: each item shows `"Segment N: <issue>"` + one-line suggestion + "Fix →" button (pre-fills the segment NL input above with the QA suggestion)
- NL edit and style creation are **separate inputs** — one for segment operations, one for style generation — so there's no ambiguity about what will be affected
- "Review all" moves from the segment list header into the QA section of the Hermes panel; only the amber flag borders remain on segment cards in col 2

---

## Architecture (v5)

```
                          ┌────────────────────────────────────────────────┐
  Telegram / CLI          │  Dashboard /captions                           │
  (trigger only)          │                                                │
       │                  │  [+ New Job] ─── UploadModal                   │
       ▼                  │    ├── video file picker                       │
  Agent pipeline          │    ├── toggle: auto-pipeline                   │
  Whisper → Kimi          │    └── segments JSON (skip pipeline)           │
  save job JSON           │         │                                      │
  burn draft              │         ▼                                      │
  reply with link ────────┼──► JobListView (status badges)                 │
                          │         │                                      │
                          │         ▼                                      │
                          │  EditorView (3-column grid)                    │
                          │  ┌─────────┬──────────────────┬─────────────┐  │
                          │  │ Col 1   │ Col 2            │ Col 3       │  │
                          │  │ video   │ segments         │ ✦ Hermes    │  │
                          │  │ Re-burn │ amber←flagged    │             │  │
                          │  │ Download│ Caption Style    │ Edit segs   │  │
                          │  │         │  fields          │ QA          │  │
                          │  │         │                  │ Presets     │  │
                          │  └─────────┴──────────────────┴─────────────┘  │
                          └────────────────────────────────────────────────┘
                                         │
                          ┌──────────────▼──────────────────────┐
                          │  plugin_api.py (16 routes total)    │
                          │                                     │
                          │  — v4 routes (unchanged) ——         │
                          │  POST /upload                       │
                          │  GET  /jobs/{id}/status             │
                          │  POST /jobs/{id}/nl-edit            │
                          │  POST /jobs/{id}/qa                 │
                          │  GET  /style/suggestion             │
                          │  POST /jobs/{id}/burn               │
                          │  GET  /jobs                         │
                          │  GET  /jobs/{id}                    │
                          │  PUT  /jobs/{id}/segments           │
                          │  PUT  /jobs/{id}/style              │
                          │  GET  /jobs/{id}/video              │
                          │  GET  /jobs/{id}/download           │
                          │                                     │
                          │  — v5 routes (new) ——               │
                          │  GET    /presets                    │
                          │  PUT    /presets/{name}             │
                          │  DELETE /presets/{name}             │
                          │  POST   /presets/generate           │
                          └─────────────────────────────────────┘
                                         │
                          ┌──────────────▼──────────────────────┐
                          │  Hermes internals                   │
                          │  AIAgent   (nl-edit, qa, suggest,   │
                          │             generate style)         │
                          │  MemoryStore (style diff history)   │
                          │  tools.video_caption (pipeline)     │
                          └─────────────────────────────────────┘
```

---

## Files Changed (v5)

| File | Change |
|---|---|
| `plugins/phonetic-captions/dashboard/plugin_api.py` | +4 preset endpoints; `_presets_dir()`, `_safe_preset_name()` helpers; `_STYLE_GENERATE_SYSTEM_PROMPT`; `PresetPayload`, `GenerateStylePayload` Pydantic models |
| `plugins/phonetic-captions/dashboard/src/index.tsx` | New `HermesPanel` component; new `PresetGallery` component; `EditorView` refactored to 3-column grid; `NLEditPanel` inlined into `HermesPanel`; QA details moved to `HermesPanel`; remove file load/save buttons and old suggestion banner; add `CaptionPreset` type |
| `plugins/phonetic-captions/dashboard/dist/index.js` | Rebuilt bundle |

No other core files touched.

---

## Testing Flows

### Test D — Preset gallery basics

1. Open any job in the editor
2. In the Hermes panel → Style presets section → "Save current" → type name "default-white"
3. A card labelled "default-white" appears in the gallery
4. Change font size to 64 in the style fields
5. Click "default-white" card → font size resets to saved value
6. Click ✕ on the card → card disappears; backend confirms `DELETE /presets/default-white`

### Test E — Create style with AI

1. Hermes panel → Style presets → "Create with AI" (expand)
2. Type: *"Large bold yellow Impact text at the bottom, thick black outline"*
3. "Generating…" spinner → preview card appears with `font: Impact`, `primary_color: &H0000FFFF`, etc.
4. Enter name "tiktok-bold" → Save → card added to gallery
5. Click "tiktok-bold" card → style fields update; video Re-burn reflects new look

### Test F — Learned preset card

1. Burn 3+ jobs with non-default style (e.g. font_size 56, yellow primary color)
2. Open a new job; Hermes panel → Style presets → amber "Learned" card appears
3. Card shows: *"Based on N sessions: larger font (56), yellow text"* (example)
4. Click "Apply" → style fields update
5. Click "Save as preset" → name input → saves to gallery as named card

### Test G — QA in Hermes panel

1. Open any job → Hermes panel → QA section → "Review"
2. Flags appear in the Hermes panel (segment N: issue + suggestion)
3. Corresponding segment cards in col 2 gain amber left borders
4. Click a flag → segment list scrolls to that card
5. "Fix →" on a flag → segment NL input above is pre-filled; submit → diff appears

### Test H — 3-column responsive

1. Open editor at ≥ 1280px viewport → 3 columns visible side by side
2. Narrow browser to < 1280px → Hermes panel stacks below the segment list
3. Re-burn and Download buttons remain in col 1 at all widths

---

## Submission Checklist (EOD Sunday)

- [ ] 3-column layout renders correctly in Chrome
- [ ] Preset gallery: save, apply, delete all work
- [ ] AI style generation returns valid `CaptionStyle` JSON
- [ ] Learned card appears after ≥ 3 burns with non-default style
- [ ] QA flags appear in Hermes panel + amber borders on segment cards
- [ ] NL segment edits still work from Hermes panel
- [ ] Re-burn and Download accessible from col 1
- [ ] Hard refresh on `/captions/{id}` still works
- [ ] Upload modal still works (v4 regression check)
- [ ] End-to-end: Telegram video → agent → dashboard → edit → preset → re-burn → download
