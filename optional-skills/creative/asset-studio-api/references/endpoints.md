# asset-studio API — Full Endpoint Reference

asset-studio has no global route prefix; each router sets its own. This file
catalogues the endpoints this skill treats as in-scope (the current v2 flow
pipeline, plus accounts/templates/characters/reel/batch), what's explicitly
out of scope, and the risk tier for every mutating call. `SKILL.md` only
carries the ~14 GET shortcuts an agent reaches for daily. For anything not
listed here, or to check whether a route's shape has changed, fetch the live
schema:

```bash
python3 scripts/asset_studio_api.py --port <port> spec --path <substring>
```

## Risk tiers

- **SAFE** — GET, read-only, call freely.
- **WRITE** — mutates state (a flow, a queue, a template file) but doesn't
  spend money. Requires `--confirm`; a quick heads-up to the user before
  running it is enough.
- **HIGH-RISK** — spends real money (Magnific credits, or a capped-but-real
  LLM call) or destroys stored data (any `DELETE`). Requires `--confirm` and
  prints a loud banner even with it. Get the user's *explicit, specific*
  approval first — "generate 3 running-pose variations for nerdysteps?", not
  a standing "sure go ahead".

## Scope: what this skill covers, and why

Verified by reading `agents/PRODUCT.md`, `agents/DESIGN.md`, and
`agents/README.md` in the asset-studio repo (the `agents` worktree — not
this skill's own `agents/` naming) plus the router/service source, not
guessed:

- **In scope, current:** `/api/v2/flows/*` (the carousel-authoring pipeline
  DESIGN.md names as "current"), `/api/accounts/*`, `/api/templates/*`
  (explicitly requested even though, see below, it's a different template
  system than the one v2 flows actually use), `/api/accounts/{handle}/
  characters/*` (generation + cutout library feeding the v2 builder), and
  `/api/reel/*` (a live, feature-flagged export path, gated behind
  `FEATURE_MAKE_REEL`).
- **In scope, current but separate feature area:** `/api/batch/*`'s
  **issue-#7 queue system** (`POST /api/batch`, `.../jobs`, `.../queue`,
  `.../order` (PUT), `.../start`, `.../status`, `.../summary`, `.../cancel`,
  `.../persist`, `GET /api/batch/jobs/{id}`). README.md documents this as a
  live, actively-developed feature (issues #7, #48, #51) for running a
  sequential queue of caption/image jobs — it is **not** superseded by v2
  flows, it's a different workflow (bulk-process N already-uploaded photos
  vs. generate-a-character-and-composite-into-a-template). Included with
  the same cost gating as character generation, because `POST .../start`
  spawns a headless `claude -p ... mcp__magnific` subprocess per queued job
  (confirmed by reading `services/batch/runner.py`).
- **Excluded, legacy v1 (per explicit instruction, confirmed in
  `agents/DESIGN.md`'s "Legacy (v1) — Slated for Removal" section and
  `agents/README.md`'s issue #94 entry, "Legacy Carousel Removal"):**
  `POST /api/run`, `POST /api/compose` (top-level), `POST /api/outline`,
  `POST /api/interesting-compose`, `POST /api/suggest`, `GET /api/job/{id}`,
  `GET /api/jobs`, `GET|POST /api/profile*`, and the entire `/api/run`-based
  half of `/api/batch/*`: `POST /api/batch/run`, `GET /api/batch/run/{id}`,
  `POST /api/batch/run/{id}/order`. Also excluded: `POST /api/batch/{id}/
  pick`, `POST /api/batch/{id}/more-options/{slide_idx}`, and `GET/POST
  /api/batch/{id}/export` — these three live under the `{batch_id}` path
  pattern but, per `routers/batch.py`, only operate on `RUN_BATCHES` state
  populated by the excluded `POST /api/batch/run` — they're unusable without
  the legacy creation call, so treat them as part of the excluded surface
  even though their paths don't say `/run/`.
- **Excluded, stale prototype:** `/api/slide-editor/*` — DESIGN.md/README.md
  (issue #91, superseded by issue #92's v2 flows) mark this as the
  predecessor `POST /api/slide-editor/flows` etc. Don't build new automation
  against it.
- **Out of scope, not evaluated:** `/api/magnific/auth/*` (Magnific's own
  OAuth flow — orthogonal to content authoring), the bare `/` UI route.
  Reachable via `call`/`spec` if ever needed; just not catalogued here.

### `/api/templates/*` vs. `/api/accounts/{handle}/templates` — don't confuse them

Two unrelated "template" systems share the word:

- **`/api/templates/*`** (`GET /api/templates`, `GET /api/templates/{name}`,
  `PUT /api/templates/{name}`, `POST /api/templates/{name}/assets`) — JSON
  files under `templates/<name>.json` (`default`, `cinematic`, `bold`
  ship in-repo) that drive the **v1** Pillow `compose()` pipeline
  (`POST /api/compose`, itself excluded as legacy). Included in this skill's
  scope per explicit instruction, but note it's not what v2 flows use.
- **`/api/accounts/{handle}/templates`** — the account's HTML layout packs
  (`cover`, `content`, `cta`, `quote`, `stat` for `nerdysteps`) that v2 flow
  slides actually select via `layout`. This is the one that matters for the
  Procedure in SKILL.md.

---

## V2 Flows — `/api/v2/flows/*` (current carousel pipeline)

| Risk | Method | Path | Purpose |
|---|---|---|---|
| WRITE | POST | `/api/v2/flows` | Create a flow (`account`, `slide_count` 1–5) |
| WRITE | POST | `/api/v2/flows/single` | Quick-create a 1-slide flow (`stat` layout if the account has it, else `content`) |
| SAFE | GET | `/api/v2/flows` | List all flows, queue view (shortcut: `flows`) |
| WRITE | POST | `/api/v2/flows/import` | LLM-drafted flow from pasted content (`crux_verdict`/`metric_week`) — subscription-funded `claude` CLI call, no per-call `$` tracked, still a real LLM invocation |
| SAFE | GET | `/api/v2/flows/{flow_id}` | Full flow state (shortcut: `flow`) |
| WRITE | POST | `/api/v2/flows/{flow_id}/slides` | Add a slide (max 5) |
| WRITE | DELETE | `/api/v2/flows/{flow_id}/slides/{i}` | Remove a slide (min 1) — DELETE, so HIGH-RISK by the blanket rule even though it only deletes flow-local state |
| WRITE | POST | `/api/v2/flows/{flow_id}/slides/{i}/images/{slot_key}` | Upload PNG/JPG ≤10MB to an image slot (e.g. the `stat` layout's chart) — **not** how character images are assigned, see caveat below |
| WRITE | DELETE | `/api/v2/flows/{flow_id}/slides/{i}/images/{slot_key}` | Clear an image slot |
| WRITE | PUT | `/api/v2/flows/{flow_id}/slides/reorder` | Reorder slides |
| WRITE | PUT | `/api/v2/flows/{flow_id}/slides/{i}/layout` | Switch a slide's layout |
| WRITE | PUT | `/api/v2/flows/{flow_id}/slides/{i}/texts` | Set slide text content (validated against the layout's slot manifest) |
| SAFE | GET | `/api/v2/flows/{flow_id}/slides/{i}/preview` | Render a slide preview PNG (shortcut: `slide_preview`) — **see Pitfalls: hung on the reference instance past 90s** |
| WRITE | PATCH | `/api/v2/flows/{flow_id}/status` | Transition `draft`/`ready`/`posted` |
| HIGH-RISK | POST | `/api/v2/flows/{flow_id}/caption` | Generate a Thai caption + hashtags — `claude -p`, `--max-budget-usd 0.05` (real, capped spend) |
| WRITE | PUT | `/api/v2/flows/{flow_id}/caption` | Save an operator-edited caption — free, don't confuse with the POST above |
| WRITE | POST | `/api/v2/flows/{flow_id}/export` | Render all slides → ZIP (or bare PNG for a 1-slide flow). Verified via source (`services/v2flows.py`, `services/render/html_renderer.py`): pure Pillow/Chromium render, **no Magnific or LLM call** — WRITE not HIGH-RISK, but shares the same Chromium render path as `slide_preview`, which hung on the live reference instance; budget for a long `--timeout` |

### Known gap — no API to assign a character cutout to a slide

`agents/README.md`'s "Character Picker (issue #90)" write-up describes
clicking a cutout to assign it to a slide's `character_path`. Reading the
actual code (`services/v2flows.py`, `routers/v2flows.py`) turned up **no
endpoint that sets `character_path`** — every v2 slide is created with it
hardcoded to `None`, and nothing in the v2 flows or characters routers ever
writes to it. A repo-wide grep also found no frontend JS wiring this up
(`asset_studio.html` has no `character_path` references at all, and there's
no separate v2-flow-builder HTML file in this checkout). Treat character
assignment as **not currently reachable through the API** — the closest
adjacent endpoint, `images/{slot_key}`, populates `image_paths` (used for
the `stat` layout's chart), not `character_path`. Don't invent a call for
this; if a user needs it, say plainly that this version of the API doesn't
expose it.

## Accounts — `/api/accounts/*`

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/accounts` | List accounts (shortcut: `accounts`) |
| SAFE | GET | `/api/accounts/{handle}/templates` | Account's HTML layout packs + slot manifests (shortcut: `account_templates`) |

## Characters — generation + cutout library (`/api/accounts/{handle}/characters/*`)

| Risk | Method | Path | Purpose |
|---|---|---|---|
| HIGH-RISK | POST | `/api/accounts/{handle}/characters/generate` | Queue N (2–4) character pose variations — real Magnific spend, no cost figure returned in the response (unlike batch/reel jobs) |
| SAFE | GET | `/api/accounts/{handle}/characters/jobs/{job_id}` | Poll a generation job (shortcut: `character_job`) |
| HIGH-RISK | POST | `/api/accounts/{handle}/characters/cutout` | Background-remove a raw variation into a transparent cutout — Magnific spend |
| SAFE | GET | `/api/accounts/{handle}/characters/cutouts` | List cutouts, optional `?action=` filter (shortcut: `cutouts`) |
| HIGH-RISK | DELETE | `/api/accounts/{handle}/characters/cutouts/{cutout_id}` | Delete a cutout (file + metadata) |

## Templates — `/api/templates/*` (v1 compose templates — see note above)

| Risk | Method | Path | Purpose |
|---|---|---|---|
| SAFE | GET | `/api/templates` | List template names (shortcut: `templates`) |
| SAFE | GET | `/api/templates/{name}` | Full template JSON (shortcut: `template`) |
| WRITE | PUT | `/api/templates/{name}` | Create/overwrite a template (schema-validated) |
| WRITE | POST | `/api/templates/{name}/assets` | Upload a character/preview/logo image asset for a template |

## Make Reel — `/api/reel/*` (feature-flagged: `FEATURE_MAKE_REEL`)

| Risk | Method | Path | Purpose |
|---|---|---|---|
| HIGH-RISK | POST | `/api/reel/start` | Image→video Magnific job for a slide (1080×1920 MP4, 3–5s) |
| SAFE | GET | `/api/reel/job/{job_id}` | Poll job status/output (shortcut: `reel_job`); returns 403 if the flag is off |
| HIGH-RISK | POST | `/api/reel/retry/{job_id}` | Re-submit a failed reel job — spends again |

## Batch Queue — `/api/batch/*` (issue #7 system; separate from the excluded `/api/batch/run/*`)

| Risk | Method | Path | Purpose |
|---|---|---|---|
| WRITE | POST | `/api/batch` | Create an empty batch queue |
| WRITE | POST | `/api/batch/{batch_id}/jobs` | Add a caption/image job to a pending batch |
| WRITE | POST | `/api/batch/jobs` | Create a single-job batch in one call |
| SAFE | GET | `/api/batch/{batch_id}/queue` | List jobs in order (shortcut: `batch_queue`) |
| WRITE | DELETE | `/api/batch/{batch_id}/jobs/{job_id}` | Remove a pending job |
| WRITE | PUT | `/api/batch/{batch_id}/order` | Reorder a pending queue |
| HIGH-RISK | POST | `/api/batch/{batch_id}/start` | Run the queue sequentially — spawns a paid `claude -p ... mcp__magnific` subprocess per job |
| SAFE | GET | `/api/batch/{batch_id}/status` | Progress: index, percent complete (shortcut: `batch_status`) |
| SAFE | GET | `/api/batch/jobs/{job_id}` | Single job detail (shortcut: `batch_job`) |
| SAFE | GET | `/api/batch/{batch_id}/summary` | Aggregate Magnific credits + agent cost (shortcut: `batch_summary`) |
| WRITE | POST | `/api/batch/{batch_id}/cancel` | Request cancellation of a running batch |
| WRITE | POST | `/api/batch/{batch_id}/persist` | Persist the pending queue to disk (`.batch_queues/`) for crash recovery |

Note: `GET /api/batch/{batch_id}` (bare, no suffix) is a dispatcher — it
checks the excluded `/run` system's in-memory store first, then falls
through to this queue system's status. Prefer the explicit
`GET /api/batch/{batch_id}/status` shortcut instead; it always means the
queue system, unambiguously.

## Not covered by this skill

`/api/run`, `/api/compose`, `/api/outline`, `/api/interesting-compose`,
`/api/suggest`, `/api/job/{id}`, `/api/jobs`, `/api/profile*`,
`/api/batch/run*`, `/api/batch/{id}/pick`, `/api/batch/{id}/more-options/*`,
`/api/batch/{id}/export`, `/api/slide-editor/*`, `/api/magnific/auth/*`, `/`.
All reachable via `call`/`spec` if a specific need comes up, but none of
them have named shortcuts, and the legacy ones shouldn't be built on.
