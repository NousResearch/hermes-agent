# Hermes Caption Agent — Plan v4 (Hermes-Integrated Dashboard)

**Hackathon**: Hermes Agent Creative Hackathon  
**Prize pool**: $25k (Main $15k + Kimi Track $5k)  
**Deadline**: EOD Sunday May 3rd, 2026  
**Previous plans**:
- `PLAN_v1.md` — Telegram-native pipeline (executed)
- `PLAN_v2.md` — Dashboard core-edit approach (archived)
- `PLAN_v3.md` — Plugin architecture, editing + burn (executed)

---

## What Changed from v3

Plan v3 shipped a fully working caption editor plugin. The dashboard still required the Hermes agent tool (triggered via Telegram or CLI) to create jobs — the UI could only *edit* what the agent already made.

Plan v4 makes the dashboard **fully self-contained** and elevates it from a simple editor into a tool that actively uses Hermes' intelligence (LLM, memory) to assist the user at every stage.

---

## New Capabilities

### 1. File Upload — Dashboard-Initiated Jobs

Users can now create a caption job entirely from the dashboard, without going through the agent.

**Upload modal** ("+ New Job" button on the job list):
- Video file picker (`.mp4 .mov .avi .mkv .webm .m4v .ts .mts`)
- Toggle: "Auto-transcribe & generate phonetics" (default ON)
  - **ON**: uploads video → server runs Whisper + Kimi phonetics in a background thread → UI polls every 2s showing live status ("Transcribing audio…" → "Generating phonetics…") → navigates to editor when ready
  - **OFF**: second file picker appears for a segments JSON — skips pipeline entirely, job is immediately ready

**New job status lifecycle**: `pending → transcribing → generating_phonetics → ready | error`

Status is stored in the job JSON and exposed via `GET /jobs/{id}/status` for polling. Existing jobs without a `status` field default to `"ready"` (backward compat).

---

### 2. Natural Language Segment Editing

A text input panel docked below the segment list. Users describe what they want in plain English; Hermes proposes a structured diff; the user approves or rejects each change individually.

**Supported operations via NL:**
- Edit text, phonetics, or language classification on any segment
- Shift timing (`start`/`end` in seconds)
- Merge two or more segments into one
- Split a segment at a word boundary

**Flow:**
1. User types instruction, e.g. *"fix the diacritics in segment 4"* or *"merge segments 8 and 9"*
2. `POST /jobs/{id}/nl-edit` → `AIAgent` (1 iteration, `quiet_mode=True`) returns a JSON patch array
3. Frontend shows a before/after list with per-change checkboxes (all checked by default)
4. "Apply N changes" commits selected patches to local state and auto-saves via `PUT /segments`
5. "Dismiss" discards everything

The agent never writes to disk — it only proposes. The user is always in control.

**QA → NL link**: "Fix with AI" buttons on flagged segments pre-fill the NL panel with the QA suggestion, letting users review before submitting.

---

### 3. Segment QA — AI Quality Review

"Review all" button in the segments header. Sends all segments to the agent with structured QA instructions; highlights problems in the editor.

**What gets flagged:**
- Wrong language classification (Vietnamese text labelled EN or vice versa)
- Mangled Vietnamese diacritics (Whisper artifacts)
- Phonetic guide that doesn't phonetically match the Vietnamese text
- Very short segments (< 0.3 s) — likely stray words, candidate for merge
- Very long segments (> 8 s) — likely should be split
- Empty text field

**UI**: Flagged segments get an amber left border. Each flag shows the issue + a one-line suggestion + "Fix with AI" button that pre-fills the NL panel.

---

### 4. Cross-Session Style Memory (Hermes-Powered)

The style suggestion system uses Hermes' own `MemoryStore` to learn from past burns across sessions — not a simple cache of last-used values.

**How it works:**
1. **Passive accumulation**: On every successful burn, the diff between the used style and the defaults is written to `MemoryStore` as a compact entry: `"Caption style edit (job abc): {font_size: 56, primary_color: '&H0000FFFF'}"`. Zero UI, instant, uses the same memory infrastructure Hermes uses for conversation notes.

2. **On-demand analysis**: Once ≥ 3 diff entries exist, a "Suggest style" button appears in the style panel. Clicking it calls `GET /style/suggestion`, which runs an `AIAgent` pass over the accumulated diffs and returns a `CaptionStyle` object + a 1-sentence explanation of the observed pattern (e.g. *"Based on 10 sessions: larger font (56), yellow text, wider bottom margin"*).

3. **Inline Apply**: The suggestion appears as a dismissible banner above the style fields with an "Apply" button. Applying updates the local style state — no burn triggered, user can still tweak before committing.

**Why this is better than a "last used" cache**: the agent identifies *patterns across content types*, summarises its reasoning in plain language, and keeps the memory compact via periodic summarisation. It also surfaces the reasoning to the user rather than silently overriding their choices.

---

### 5. Style Preset File Load/Save

Simple stateless import/export for sharing style configs between machines or team members.

- **Load**: "Load" button → hidden file input → `FileReader` → validates required `CaptionStyle` keys → updates local style state. No backend call.
- **Save**: "Save" button → serialises current style → `Blob` → `<a>` click → downloads `caption-style.json`. No backend call.

---

### 6. Deep-Link on Hard Refresh Fix

Plugin routes (e.g. `/captions/abc123`) now survive a hard refresh (`Cmd+R`). Previously, the React Router `*` catch-all would fire before plugin manifests had loaded, immediately redirecting to `/sessions`. Fixed by suppressing the catch-all while `pluginsLoading` is true in `web/src/App.tsx`.

---

## Architecture (v4)

```
                          ┌──────────────────────────────────────┐
  Telegram / CLI          │  Dashboard /captions                 │
  (trigger only)          │                                      │
       │                  │  [+ New Job] ─── UploadModal         │
       ▼                  │    ├── video file picker             │
  Agent pipeline          │    ├── toggle: auto-pipeline         │
  Whisper → Kimi          │    └── segments JSON (skip pipeline) │
  save job JSON           │         │                            │
  burn draft              │         ▼                            │
  reply with link ────────┼──► JobListView (status badges)       │
                          │         │                            │
                          │         ▼                            │
                          │  EditorView                          │
                          │  ┌──────────────┬─────────────────┐  │
                          │  │ video player │ segments list   │  │
                          │  │              │ [Review all] ───┤  │
                          │  │ [Re-burn]    │ flagged ←amber  │  │
                          │  │ [Download]   │ [Fix with AI]   │  │
                          │  │              │                 │  │
                          │  │              │ NL Edit Panel   │  │
                          │  │              │ [instruction]   │  │
                          │  │              │ [patch diff]    │  │
                          │  │              │                 │  │
                          │  │              │ Style Panel     │  │
                          │  │              │ [Load] [Save]   │  │
                          │  │              │ [Suggest style] │  │
                          │  └──────────────┴─────────────────┘  │
                          └──────────────────────────────────────┘
                                         │
                          ┌──────────────▼──────────────────────┐
                          │  plugin_api.py (12 routes total)    │
                          │                                     │
                          │  POST /upload            ← new      │
                          │  GET  /jobs/{id}/status  ← new      │
                          │  POST /jobs/{id}/nl-edit ← new      │
                          │  POST /jobs/{id}/qa      ← new      │
                          │  GET  /style/suggestion  ← new      │
                          │  POST /jobs/{id}/burn    ← +memory  │
                          │  GET  /jobs                         │
                          │  GET  /jobs/{id}                    │
                          │  PUT  /jobs/{id}/segments           │
                          │  PUT  /jobs/{id}/style              │
                          │  GET  /jobs/{id}/video              │
                          │  GET  /jobs/{id}/download           │
                          └─────────────────────────────────────┘
                                         │
                          ┌──────────────▼──────────────────────┐
                          │  Hermes internals                   │
                          │  AIAgent   (nl-edit, qa, suggest)   │
                          │  MemoryStore (style diff history)   │
                          │  tools.video_caption (pipeline)     │
                          └─────────────────────────────────────┘
```

---

## Files Changed (v4)

| File | Change |
|---|---|
| `plugins/phonetic-captions/dashboard/plugin_api.py` | 5 new endpoints; `_run_pipeline`, `_call_agent`, `_update_job_status` helpers; `burn` writes to `MemoryStore` |
| `plugins/phonetic-captions/dashboard/src/index.tsx` | `UploadModal`, `NLEditPanel`, QA highlighting, style suggestion banner, style preset load/save, status badges on job cards |
| `plugins/phonetic-captions/dashboard/dist/index.js` | Rebuilt bundle (35.8 kB) |
| `web/src/App.tsx` | Catch-all redirect suppressed during `pluginsLoading` |
| `hermes_cli/web_dist/` | Rebuilt web dist |

---

## Testing Flows

### Test A — Upload with auto-pipeline

1. Open dashboard → Captions tab → "+ New Job"
2. Select a `.mp4` file; leave "Auto-transcribe" ON → "Create Job"
3. Modal shows "Transcribing audio…" → "Generating phonetics…"
4. Editor opens automatically with segments populated
5. **Verify**: segments have `lang` set; Vietnamese segments have `phonetic` field

### Test B — Upload with manual segments

1. "+ New Job" → select video → uncheck "Auto-transcribe" → upload a segments `.json` → "Create Job"
2. Editor opens immediately (no pipeline wait)
3. **Verify**: segments match the uploaded JSON exactly

### Test C — NL segment editing

1. Open any job with segments
2. Type *"fix the diacritics in segment 3"* → Enter
3. Diff list appears with proposed `text`/`phonetic` changes, all checked
4. Uncheck one change → "Apply N changes"
5. **Verify**: only checked changes appear in the list; `PUT /segments` fires (check network tab)
6. Type *"merge segments 4 and 5"* → apply
7. **Verify**: single merged segment, IDs renumbered

### Test D — Segment QA

1. Open a freshly transcribed job (likely has Whisper diacritics issues)
2. Click "Review all"
3. **Verify**: flagged segments get amber border; flag shows issue + suggestion
4. Click "Fix with AI" → NL panel pre-fills with suggestion text
5. Submit → apply patch → flags cleared on next "Review all"

### Test E — Style memory suggestion

1. Open 3+ different jobs; change style settings (e.g. font_size → 56, text color → yellow); Re-burn each
2. Open a new job → Style panel → "Suggest style" button visible
3. Click → amber banner with explanation (e.g. *"Based on N sessions: font_size 56, yellow text"*)
4. Click "Apply" → style fields update
5. **Verify**: banner is dismissible; suggestion persists across page reloads

### Test F — Style preset round-trip

1. Set custom style → "Save" → `caption-style.json` downloads
2. Open a different job → "Load" → select the file
3. **Verify**: style fields match saved values; no burn triggered

### Test G — Hard refresh deep-link

1. Navigate to an open job at `/captions/abc123`
2. Press `Cmd+Shift+R`
3. **Verify**: page reloads to the same job editor, not `/sessions`

---

## Known Limitations

| Item | Notes |
|---|---|
| NL split op | Split via NL returns `{op: "split", at_word_index}` but the frontend defers this to the interactive `SplitEditor` UI (word-chip picker); a note in the diff list directs the user to the ✂ button |
| Style suggestion threshold | Requires ≥ 3 burns with non-default styles before "Suggest" appears |
| NL/QA requires model config | `_call_agent` reads the configured model from `config.yaml`; errors gracefully with 502 if unset |
| Upload size | File picker has no explicit limit; validated server-side only by extension |
