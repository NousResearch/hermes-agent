# Hermes Caption Agent — Plan v3 (Dashboard Plugin)

**Hackathon**: Hermes Agent Creative Hackathon  
**Prize pool**: $25k (Main $15k + Kimi Track $5k)  
**Deadline**: EOD Sunday May 3rd, 2026  
**Previous plans**: 
- `PLAN_v1.md` (Telegram-native, executed — backend pipeline complete)
- `PLAN_v2.md` (Dashboard visual editor, core approach — now replaced by plugin model)

---

## Key Insight: Use the Plugin System

The Hermes dashboard was built to be extended without forking — **themes and plugins are first-class citizens**. Rather than edit core files (`web/src/App.tsx`, `hermes_cli/web_server.py`), the caption editor ships as a **self-contained dashboard plugin**.

**References:**
- [Extending the Dashboard](https://hermes-agent.nousresearch.com/docs/user-guide/features/extending-the-dashboard) — official guide
- [Combined theme + plugin demo](https://hermes-agent.nousresearch.com/docs/user-guide/features/extending-the-dashboard#combined-theme--plugin-demo) — working example (`strike-freedom-cockpit`)


---

## Problem Statement (Unchanged)

Content creators making bilingual (English + Vietnamese) videos spend 30–45 minutes per video manually choosing font, size, color, position in CapCut, typing Vietnamese captions with no consistency, and fixing translation mistakes with no memory of prior corrections.

**User quote**: *"I was hoping that the tool can auto generate the caption for both English and Vietnamese with the same font, size and color and where it appears on the video with the same style that I'm doing in each video right now."*

---

## Architecture (Plugin-Based)

```
Telegram (trigger)
  ↓
Agent: transcribe (Whisper) + generate phonetics (Kimi K2.5, once)
  ↓
Save job JSON to ~/.hermes/caption-jobs/{id}.json
  ↓
Burn initial draft video (FFmpeg)
  ↓
Return: draft video + link to /captions/{id}
  ↓
Plugin at ~/.hermes/plugins/phonetic-captions/dashboard/
  ├── manifest.json          (tab registration, icon, entry point)
  ├── plugin_api.py          (7 FastAPI routes at /api/plugins/phonetic-captions/*)
  ├── src/index.tsx          (React editor — IIFE, no build step for users)
  └── dist/index.js          (pre-built bundle)
  ↓
Dashboard /captions/{id}
  ┌─────────────────────┬─────────────────────────────────────┐
  │   <video> player    │  Segment list (editable)            │
  │   (auto-reloads     │  [EN/VI badge] [text] [phonetic]   │
  │    after re-burn)   │                                     │
  │                     │  Style panel                        │
  │                     │  Font / Size / Color / Margin       │
  │                     │                                     │
  │                     │  [Re-burn]  [Download]              │
  └─────────────────────┴─────────────────────────────────────┘
```

**Key differences from v2:**
- No `/web/src/` changes — plugin provides the UI
- Plugin API routes mounted at `/api/plugins/phonetic-captions/` (auth-free on localhost by design)
- Pre-built IIFE bundle — no user-facing build step
- Core dashboard remains unchanged — easier future maintenance

---

## Implementation Phases

### Phase 1 — Backend: plugin_api.py (1h)

**Location**: `plugins/phonetic-captions/dashboard/plugin_api.py`

Create FastAPI router with 7 endpoints (moved from Plan v2's `web_server.py` additions):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/jobs` | List all jobs |
| `GET` | `/jobs/{id}` | Get full job state |
| `PUT` | `/jobs/{id}/segments` | Save edited segments |
| `PUT` | `/jobs/{id}/style` | Save style changes |
| `POST` | `/jobs/{id}/burn` | Re-burn + update output_path |
| `GET` | `/jobs/{id}/video` | Stream video for `<video>` player |
| `GET` | `/jobs/{id}/download` | Download final output |

**Auth note**: Plugin routes bypass `_require_token()` — the framework explicitly skips `/api/plugins/*` prefix in the auth middleware. This is safe because dashboard only binds to `localhost` by default.

**Imports**: `_build_ass_content`, `burn` from `tools.video_caption` (direct, no subprocess).

**Helpers**: Copy `_caption_jobs_dir()`, `_load_caption_job()`, `_save_caption_job_data()` helpers into this file.

### Phase 2 — Plugin manifest and build setup (30 min)

**`plugins/phonetic-captions/dashboard/manifest.json`**:
```json
{
  "name": "phonetic-captions",
  "label": "Captions",
  "description": "Bilingual EN/VI phonetic caption editor",
  "icon": "FileText",
  "version": "1.0.0",
  "tab": {
    "path": "/captions",
    "position": "after:skills"
  },
  "entry": "dist/index.js",
  "api": "plugin_api.py"
}
```

**`plugins/phonetic-captions/dashboard/package.json`**:
```json
{
  "private": true,
  "scripts": { "build": "node build.mjs" },
  "devDependencies": {
    "esbuild": "^0.25",
    "lucide-react": "*",
    "typescript": "*"
  }
}
```

**`plugins/phonetic-captions/dashboard/build.mjs`** — esbuild config:
- Redirect `react` imports → `window.__HERMES_PLUGIN_SDK__.React`
- Redirect `react/jsx-runtime` → SDK shim for JSX transform
- Bundle `lucide-react` icons directly (~10KB)
- Format: IIFE
- Output: `dist/index.js`

### Phase 3 — React plugin UI (2h)

**`plugins/phonetic-captions/dashboard/src/index.tsx`**

Adapted from Plan v2's `CaptionEditorPage.tsx` with these changes:

1. **No react-router-dom** — use state-based navigation:
   - `useState<string | null>(null)` for `selectedJobId`
   - `useEffect` checks `window.location.pathname` on mount — if `/captions/<id>`, auto-select that job
   - Use `window.history.pushState()` to update URL for shareability

2. **SDK components** — replace `@nous-research/ui`:
   - `Button` → `SDK.components.Button`
   - Typography → plain `<div>` with Tailwind classes (`text-xl font-semibold`, etc.)
   - Use `SDK.React` for all `createElement` calls

3. **API paths** — update to plugin routes:
   - `/api/caption/jobs` → `/api/plugins/phonetic-captions/jobs`
   - `/api/caption/jobs/{id}/video` → `/api/plugins/phonetic-captions/jobs/{id}/video`
   - etc.

4. **SDK utilities**:
   - `SDK.fetchJSON` — handles auth automatically (no token plumbing needed)
   - `SDK.hooks.useState`, `SDK.hooks.useEffect`, etc.
   - `SDK.utils.cn` — Tailwind class merger

5. **Download**:
   ```js
   const res = await fetch(`/api/plugins/phonetic-captions/jobs/${id}/download`);
   // No auth header needed (plugin routes auth-free on localhost)
   ```

6. **Registration** — wrap in IIFE and register:
   ```js
   (function() {
     const SDK = window.__HERMES_PLUGIN_SDK__;
     const PLUGINS = window.__HERMES_PLUGINS__;
     if (!SDK || !PLUGINS) return; // guard for old dashboards
     
     // ... component definitions ...
     
     PLUGINS.register('phonetic-captions', CaptionApp);
   })();
   ```

7. **Spinner** — keep as local inline CSS component (no SDK equivalent)

8. **Icons** — import from `lucide-react` — bundled directly into IIFE

**Pre-build** `dist/index.js` and commit — no build step required for end users.

### Phase 4 — Verify core files unchanged (verify only)

No changes needed to:
- `hermes_cli/web_server.py` — caption endpoints stay out
- `web/src/App.tsx` — no caption routes added
- `web/src/pages/` — no new page added

The plugin system auto-discovers at `/api/dashboard/plugins` — no hardcoding needed.

### Phase 5 — Minor skill update (15 min)

**`skills/video/phonetic_captions/SKILL.md`**:
- After `caption` op, agent surfaces the dashboard link to the job ID
- Link format: `http://localhost:9119/captions/{job_id}` (or `https://` if exposed)
- Plugin reads this on mount and auto-opens the job

---

## Files Changed / Created

| File | Action | Phase |
|---|---|---|
| `plugins/phonetic-captions/dashboard/manifest.json` | Create | 2 |
| `plugins/phonetic-captions/dashboard/plugin_api.py` | Create | 1 |
| `plugins/phonetic-captions/dashboard/src/index.tsx` | Create | 3 |
| `plugins/phonetic-captions/dashboard/build.mjs` | Create | 2 |
| `plugins/phonetic-captions/dashboard/package.json` | Create | 2 |
| `plugins/phonetic-captions/dashboard/dist/index.js` | Create (pre-built) | 3 |
| `skills/video/phonetic_captions/SKILL.md` | Minor update | 5 |
| `web/src/App.tsx` | None | — |
| `hermes_cli/web_server.py` | None | — |

---

## Why This Is Better

| Aspect | Plan v2 (Core Edits) | Plan v3 (Plugin) |
|---|---|---|
| Core file changes | 3 files (App.tsx, web_server.py, new page) | 0 files |
| Build dependency | Yes — `npm run build` required | No — pre-built plugin |
| User install | Copy core code + rebuild | Drop `~/.hermes/plugins/` directory |
| Future maintenance | Merge conflicts if core changes | Zero interference |
| Extensibility | Hard to add features | Plugin slots, theme integration ready |
| Demo story | "We forked Hermes for this" | "We built a Hermes plugin" ✨ |

---

## Demo Story

**Meet Linh — she makes bilingual cooking content.**

1. Opens Telegram, sends today's video
2. Hermes replies: captioned draft + link to `/captions/abc123`
3. Linh clicks — caption editor opens
4. She edits segments, changes font size
5. Clicks "Re-burn" — video updates in 5s (FFmpeg only)
6. Downloads final video

**Why the plugin approach shines in a demo:**
- "This was built as a drop-in Hermes plugin — no fork, no build step"
- Shows the extensibility of the Hermes ecosystem
- Judges see best practices: uses official plugin SDK, follows auth patterns, etc.

---

## Kimi Track Strategy (Unchanged)

Submission video will show:
1. `NVIDIA_API_KEY` + `moonshotai/kimi-k2.5` config
2. Raw Whisper vs Kimi-corrected Vietnamese + phonetics
3. Plugin-based caption editor showing manual refinement on top

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Plugin SDK missing method | Guard: `if (!SDK.method) { fallback }` — works with old dashboards |
| Video streaming CORS | Serve via FastAPI with `Content-Type` + range support (built-in) |
| Re-burn blocks event loop | Run `burn()` in `asyncio.to_thread()` |
| Large job JSON | Cap at 500 segments; V2 feature: merging |

---

## Out of Scope (V2)

- Real-time caption preview
- Drag-and-drop timeline editor
- Multi-video batch management
- Mobile-responsive layout
