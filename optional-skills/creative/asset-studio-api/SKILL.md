---
name: asset-studio-api
description: asset-studio carousel API: Instagram/FB posts, flows, characters.
version: 1.0.0
author: zealchaiwut, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [asset-studio, carousel, instagram, facebook, magnific, image-generation, api]
    category: creative
    related_skills: [rest-graphql-debug]
    requires_toolsets: [terminal]
    config:
      - key: asset_studio_api.host
        description: Host asset-studio's API is reachable on (it binds to 127.0.0.1 only — no remote instances)
        default: "127.0.0.1"
        prompt: "asset-studio host (almost always 127.0.0.1)"
      - key: asset_studio_api.port
        description: Port of the specific asset-studio instance to talk to. No safe default — asset-studio runs as several parallel worktree checkouts (main/agents/coder/tester/uat), each potentially on its own port (that worktree's .env PORT, or 8000; README calls 7001 "the conventional dev port," but that's a convention, not a guarantee).
        default: ""
        prompt: "asset-studio port — check which worktree/instance the user means before guessing"
      - key: asset_studio_api.default_account
        description: Default account handle for flow/character calls when the user doesn't name one (e.g. "nerdysteps")
        default: ""
        prompt: "Default account handle (optional, saves asking every time)"
---

# Asset Studio API

Bridges Hermes to a locally running **asset-studio** — a single-operator,
local-only FastAPI app for authoring Instagram/Facebook carousel posts: AI
character generation (via the external Magnific API) → background removal →
compositing into brand-locked slide templates → PNG/MP4 export. It has
**no auth layer by design** (`agents/PRODUCT.md`: "local-first, no cloud
database, no auth layer") and binds to `127.0.0.1` only, so this client
needs no token — just the right host/port for the instance you mean.

This skill covers the current API surface: `/api/v2/flows/*` (the
carousel-authoring pipeline), `/api/accounts/*`, `/api/accounts/{handle}/
characters/*` (generation + cutout library), `/api/templates/*`, `/api/reel/*`
(feature-flagged image→video), and the issue-#7 `/api/batch/*` queue system.
It deliberately excludes the legacy v1 surface (`/api/run`, top-level
`/api/compose`, `/api/batch/run/*`) and the superseded `/api/slide-editor/*`
prototype — see `references/endpoints.md` for the full scope rationale, read
from the app's own docs and source rather than assumed.

Read **and** write — but every mutating call is confirm-gated, and calls
that spend real money (Magnific credits, or a capped LLM call) print a loud
warning even with `--confirm`.

## When to Use

- User wants to create, edit, or export an Instagram/Facebook carousel via
  asset-studio ("make a new post", "add a slide", "generate a character
  pose", "export this flow")
- User asks about the state of a flow, account, template, or job
  ("what flows are drafted", "is the character generation done yet",
  "what templates does nerdysteps have")
- Anything else touching the routes catalogued in `references/endpoints.md`
  — reachable via the generic client

## Prerequisites

- asset-studio running locally, reachable at `http://<host>:<port>`
- `python3` (stdlib only — no pip installs)
- Skill config (set via `hermes skills config`, or read from the
  `[Skill config]` block injected when this skill loads):
  - `asset_studio_api.host` — default `127.0.0.1`
  - `asset_studio_api.port` — **no default**. asset-studio can run several
    instances at once (one per git worktree — main/agents/coder/tester/uat),
    each on its own port. If the user hasn't said which one, ask, or check
    what's actually listening before guessing.
  - `asset_studio_api.default_account` — optional, saves asking every time
    (the seeded example account is `nerdysteps`)

## How to Run

All calls go through one client script, via `terminal`. **Never write ad hoc
`execute_code`/`curl`/`requests`/`subprocess` calls against this API, and
never search the asset-studio source tree for how to do something — this
script is the only sanctioned path, including for routes it has no named
shortcut for (use `call`, see below).**
`$HERMES_HOME/skills/creative/asset-studio-api/scripts/` is where this
skill lives once installed:

```bash
python3 $HERMES_HOME/skills/creative/asset-studio-api/scripts/asset_studio_api.py \
  --port <asset_studio_api.port> <subcommand> [args]
```

Every subcommand prints JSON with a `status` field. List responses are
capped at 15 items so one call can't blow the context budget. Binary
responses (slide preview PNGs, flow/batch export ZIPs) are detected by
Content-Type and never dumped as garbled text — pass `--out <path>` to save
them; without it you get size/content-type only. There are no SSE/streaming
routes anywhere in this API (checked the live OpenAPI schema and grepped the
source; the only `text/event-stream` code is asset-studio's own outbound
client talking to Magnific, not anything it exposes) — unlike this skill's
structural sibling `commander-api`, there's no `stream()` subcommand here.

Slide preview and flow/batch export render through headless Chromium
(Playwright) and can be slow, or — confirmed against this skill's reference
instance — can hang outright past 90 seconds with no response. Pass
`--timeout <seconds>` (default 30) to wait longer, and don't loop retrying
the same render; report to the user that it's stuck if it times out twice.

## Quick Reference

| Subcommand | Endpoint | Purpose |
|---|---|---|
| `accounts` | `GET /api/accounts` | List registered accounts |
| `account_templates <handle>` | `GET /api/accounts/{handle}/templates` | Account's layout packs + slot manifests |
| `flows` | `GET /api/v2/flows` | List all flows (queue view) |
| `flow <flow_id>` | `GET /api/v2/flows/{flow_id}` | Full flow state |
| `slide_preview <flow_id> <i>` | `GET .../slides/{i}/preview` | Render a slide preview PNG (binary — use `--out`) |
| `templates` | `GET /api/templates` | v1 compose templates (default/cinematic/bold) — not the v2 layout packs |
| `template <name>` | `GET /api/templates/{name}` | Full v1 template JSON |
| `character_job <handle> <job_id>` | `GET .../characters/jobs/{job_id}` | Poll a character-generation job |
| `cutouts <handle> [--action]` | `GET .../characters/cutouts` | List cutouts, optional action filter |
| `reel_job <job_id>` | `GET /api/reel/job/{job_id}` | Poll a Make Reel job (403 if feature-flagged off) |
| `batch_status <batch_id>` | `GET /api/batch/{id}/status` | Batch-queue progress |
| `batch_queue <batch_id>` | `GET /api/batch/{id}/queue` | List jobs in a batch queue |
| `batch_job <job_id>` | `GET /api/batch/jobs/{job_id}` | Single batch-queue job detail |
| `batch_summary <batch_id>` | `GET /api/batch/{id}/summary` | Magnific credits + cost rollup |
| `spec [--path <substr>]` | `GET /openapi.json` | Live schema — source of truth if this doc drifts |
| `call <METHOD> <path> [--json '<body>'] [--confirm] [--out <path>]` | any route | Escape hatch — see `references/endpoints.md` |

## Procedure

### Content Creation — the actual end-to-end flow

This is what asset-studio is *for*: authoring one carousel post. Walk it in
this order; don't skip the cost-gated steps without the user's explicit,
specific approval.

1. **Resolve the account.** Use what the user named, `asset_studio_api.
   default_account`, or `call GET /api/accounts` (shortcut `accounts`) if
   neither is set — there's currently one seeded account, `nerdysteps`.
2. **Create the flow.** `call POST /api/v2/flows --json '{"account": "<handle>", "slide_count": <1-5>}' --confirm`
   (WRITE — mention what you're about to do, not full sign-off needed for a
   plain draft create). For a single-image post, use
   `call POST /api/v2/flows/single --json '{"account": "<handle>"}' --confirm`
   instead. Read back the `flow_id` from the response.
3. **Shape the slides.** Add/remove slides
   (`POST .../slides` / `DELETE .../slides/{i}`, both `--confirm`) until the
   count matches what the user wants (1–5), then set each slide's layout —
   `call PUT /api/v2/flows/{flow_id}/slides/{i}/layout --json '{"layout": "<cover|content|cta|quote|stat>"}' --confirm`
   — using layout names from `account_templates <handle>` for that account
   (don't guess a layout name; read the manifest).
4. **Character images — known gap, read this before promising anything.**
   `agents/README.md` describes clicking a generated cutout to seat it in a
   slide's character slot, but reading the actual v2 flows code found **no
   API endpoint that sets a slide's `character_path`** (see
   `references/endpoints.md`'s "Known gap" note — verified by grepping both
   the router/service code and the frontend HTML). You *can* still run
   character generation and inspect the results:
   - `call POST /api/accounts/{handle}/characters/generate --json '{"action_key": "<preset-key>", "variations": <2-4>}' --confirm`
     — **HIGH-RISK, real Magnific spend.** Get explicit approval for this
     specific generation first; the `accounts` shortcut's response already
     includes each account's `action_presets` list — read the real preset
     keys from there rather than guessing one.
   - Poll `character_job <handle> <job_id>` until done.
   - `call POST /api/accounts/{handle}/characters/cutout --json '{"raw_image_id": "<id>"}' --confirm`
     — **HIGH-RISK, real Magnific spend** — turns one raw variation into a
     transparent cutout.
   - `cutouts <handle>` to see the library.
   - But there's currently no confirmed way to attach that cutout to a
     specific flow slide via this API. Say so plainly if the user expects
     it to "just work" — don't fabricate a call.
5. **Text.** `call PUT /api/v2/flows/{flow_id}/slides/{i}/texts --json '{"texts": {"<slot_key>": "<value>", ...}}' --confirm`
   per slide, using slot keys/limits from that slide's layout manifest
   (`account_templates` again) — the endpoint 422s with per-slot errors if
   you exceed `max_chars`, so check the manifest rather than guessing.
6. **Chart/image slots** (the `stat` layout, or similar): the underlying
   route (`POST /api/v2/flows/{flow_id}/slides/{i}/images/{slot_key}`) is a
   **multipart file upload**, which `call --json` cannot send (see
   Pitfalls). Point the user to the app UI for this one step.
7. **Caption.** `call POST /api/v2/flows/{flow_id}/caption --confirm`
   — **HIGH-RISK**: a real but small (`--max-budget-usd 0.05`) LLM spend.
   Confirm with the user first. To save an operator-edited caption instead
   (free, no LLM call): `call PUT /api/v2/flows/{flow_id}/caption --json '{"caption": "<text>"}' --confirm`.
8. **Preview.** `slide_preview <flow_id> <i> --out <path>` per slide before
   export. Renders through headless Chromium and can be slow — pass a
   generous `--timeout` and don't loop-retry a hung render (see Pitfalls).
9. **Export.** `call POST /api/v2/flows/{flow_id}/export --confirm --out <path>`
   (optionally `?ratios=4:5,1:1,9:16` via the path). WRITE, not HIGH-RISK —
   verified via source that export is a pure Pillow/Chromium render with no
   Magnific/LLM call — but still needs explicit confirmation structurally,
   and shares the same slow/hang-prone render path as preview.
10. **Status.** Once posted, `call PATCH /api/v2/flows/{flow_id}/status --json '{"status": "posted"}' --confirm`.

### Read Flows — status checks, no gating needed

- **"What flows exist / what's drafted"**: `flows`, narrate `status`,
  `title`, `slide_count`, `layout_types`, timestamps as returned — don't
  invent a summary field that isn't there.
- **"What's in this flow"**: `flow <flow_id>`.
- **"What accounts/layouts are set up"**: `accounts`, then
  `account_templates <handle>`.
- **"Is this job done yet"**: `character_job <handle> <job_id>` for
  character generation, `reel_job <job_id>` for Make Reel,
  `batch_status <batch_id>` / `batch_job <job_id>` for the batch queue.
  Relay `status` verbatim (`queued`/`running`/`done`/`failed`) — don't
  re-derive a verdict.
- **"What's this going to cost / did cost"**: `batch_summary <batch_id>`
  for the batch queue's rollup. Character generation and cutout calls don't
  return a cost figure in their own response (verified — no `total_cost_usd`
  or credits field anywhere in `services/characters*.py`); tell the user
  that plainly rather than estimating one.

### Bulk Photo Processing — the batch queue (separate feature)

If the user wants to run one operation (upscale, restyle) across several
already-uploaded photos rather than build a character-driven carousel, that's
the issue-#7 batch queue, not v2 flows:

1. `call POST /api/batch --confirm` (WRITE) → `batch_id`.
2. `call POST /api/batch/{batch_id}/jobs --json '{"caption": "...", "filename": "..."}' --confirm`
   per photo (WRITE).
3. Review with `batch_queue <batch_id>` before starting.
4. `call POST /api/batch/{batch_id}/start --confirm` — **HIGH-RISK**: this
   spawns a paid Magnific job per queued item. Get explicit approval,
   listing how many jobs are about to run.
5. Poll `batch_status <batch_id>` / `batch_job <job_id>`, then
   `batch_summary <batch_id>` for the final cost rollup.

## Pitfalls

- Never pass `--confirm` on a mutating call without the user having approved
  *that specific action* in chat — a standing "sure, go ahead" earlier in
  the conversation doesn't cover a different flow, account, or export later.
- Character generation (`.../characters/generate`), cutout creation
  (`.../characters/cutout`), reel start/retry, batch `.../start`, and
  caption generation (`POST .../caption`) all spend real money — get
  explicit approval naming the specific thing before running them, the same
  standard as any other HIGH-RISK call.
- `PUT .../caption` (save an edited caption) is free and looks similar to
  `POST .../caption` (generate, costs money) — don't conflate them when
  relaying to the user what a call will do.
- There is no API endpoint to assign a generated character cutout to a v2
  flow slide (`character_path` is never set anywhere in the current code).
  Don't promise this works; say it's not currently reachable via the API.
- Don't assume a port. asset-studio commonly runs several instances at
  once across worktrees — confirm which one the user means rather than
  defaulting to 8000 or 7001.
- Slide preview and flow/batch export can hang for well over a minute on a
  slow or stuck Chromium render (verified against the reference instance —
  a `slide_preview` call sat for 90s with zero response on two different
  flows). Pass a generous `--timeout`, and if it still times out, tell the
  user the render appears stuck rather than silently retrying in a loop.
- `/api/templates/*` and `/api/accounts/{handle}/templates` are two
  unrelated systems that both use the word "template" — the first is the
  legacy v1 compose pipeline's JSON templates, the second is what v2 flow
  slides actually pick a `layout` from. Don't mix them up when telling a
  user which layouts are available for their carousel.
- Don't build new automation against `/api/run`, `/api/compose` (top-level),
  `/api/batch/run/*`, or `/api/slide-editor/*` — all documented as legacy or
  superseded in the asset-studio repo itself. Use the `/api/v2/flows/*`
  equivalents.
- `call`'s `--json` only sends a JSON body. A handful of routes take
  multipart file uploads instead (slide image-slot upload, template asset
  upload, and the legacy batch-run image endpoints) — this script can't
  drive those. Tell the user to use the app UI for a file-upload step
  rather than trying to fake it through `--json`.

## Verification

- `python3 scripts/asset_studio_api.py --port <port> accounts` returns
  `status: 200` with real account data (no connection error) when
  asset-studio is running.
- `call POST ...` without `--confirm` always exits non-zero and refuses —
  confirming the safety gate is structural, not just documented.
- A HIGH-RISK path (character generate/cutout, reel start/retry, batch
  start, caption generate, or any `DELETE`) prints the loud warning banner
  even with `--confirm` passed; a free-adjacent call sharing a substring
  (e.g. `PUT .../caption`) does **not** print it — the gate is method-aware,
  not just a path substring match.
- Every named GET shortcut, run live against a running instance, returns a
  clean `200` with real data (or a clean `404`/`403` business-logic error
  for a made-up id) — never a `422` from a malformed/undocumented query
  param. If one doesn't, the shortcut's param list is wrong; fix it, don't
  work around it in the calling code.
- Status narration (flow/job/batch state) always uses the API's own
  `status` field verbatim, never an invented summary.
