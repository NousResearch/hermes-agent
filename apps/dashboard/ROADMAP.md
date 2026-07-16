# HERMES//HUB — feature & upgrade roadmap

Detailed, buildable proposals for the next phases of the hub. Every item here
respects the hard constraint: **zero-dependency Python stdlib server + zero-build
vanilla ES-module frontend**, optional `anthropic` SDK only. Nothing below needs
a framework, bundler, or Docker.

Each proposal states: the problem, the design, the exact files/APIs touched, a
step-by-step build plan, tests, and a rough effort (S = <½ day, M = ~1 day,
L = multi-day). Items are ordered by value-to-effort within each tier.

### Status (updated as items ship)
- ✅ **1.2 Model-augmented reflection** — shipped (`assistant.reflect_candidates`).
- ✅ **1.3 Evolution audit/rollback** — shipped (`evolve.rollback`/`history`).
- ✅ **2.1 News search** — shipped (client-side filter).
- ✅ **2.3 Routing overrides UI** — shipped (`data/routing.json` + panel).
- ✅ **2.4 Structured tasks** — shipped (due + priority; recurrence still open).
- ✅ **3.3 Accent presets** — shipped (cyan/amber/green/magenta).
- ⬜ **1.1 Web Push**, **2.2 Plan preview**, **3.1 Palette execution**,
  **3.2 Per-widget intervals**, **3.4 Backup download/upload** — still open.

---

## Tier 1 — highest value, ship first

### 1.1 Web Push notifications (real push, not just in-app toasts) — **M**

**Problem.** Automations (daily briefing, market/worldstate alerts) currently
surface only as in-app toasts + the `Notification` API, which fire only while a
tab is open. A morning-briefing automation is useless if the dashboard isn't on
screen. The whole automations engine is capped by this.

**Design.** Standard Web Push (VAPID) — fully doable stdlib-only:
- Generate a VAPID keypair once (ECDSA P-256). The signing (ES256 JWT + AES128-GCM
  payload encryption per RFC 8291) is implementable with `hashlib`, `hmac`,
  `secrets`, and the `cryptography`-free path is hard for ECDH… **caveat:** raw
  P-256 ECDH is *not* in the stdlib. Two honest options:
  - **(a) Payload-less push** (recommended, still stdlib-only): send an empty
    push "tickle"; the service worker wakes and calls `/api/notifications` to pull
    the actual content. ES256 JWT signing for VAPID *is* stdlib-only via a small
    P-256 signer using `int` math over the NIST curve (~120 lines) — or accept the
    one narrow dependency below.
  - **(b)** Allow `pywebpush` as a second optional dependency, mirroring how
    `anthropic` is optional. Keep the feature degrading to in-app toasts when
    absent. This is the pragmatic choice and matches the existing "optional SDK"
    precedent.

  Recommend **(a)**: keeps the zero-dependency promise. Payloads stay server-side
  and are pulled over the authenticated `/api/notifications` channel that already
  exists.

**Files / APIs.**
- `server.py`: new `push.py` module — VAPID keypair persisted in
  `data/vapid.json`; `POST /api/push/subscribe` (store subscription in
  `data/push_subs.json`), `POST /api/push/unsubscribe`, and an internal
  `push.notify(subscription, ttl)` that sends the tickle. Wire
  `Automations.tick()` → after appending a notification, fan out a push tickle to
  every stored subscription.
- `public/sw.js`: add a `push` event listener → `registration.showNotification`,
  and a `notificationclick` handler that focuses/opens the hub.
- `public/js/notifications.js`: `Notification.requestPermission()` +
  `pushManager.subscribe({ applicationServerKey })`; a Settings toggle.
- `GET /api/push/vapid-public` so the client can fetch the applicationServerKey.

**Build plan.**
1. `push.py`: keypair gen + persistence; ES256 signer; subscription store.
2. Endpoints + auth (reuse the existing bearer/token gate).
3. SW push/notificationclick handlers.
4. Settings UI: "Enable push notifications" with permission + subscribe flow.
5. Hook into `automations.tick()`.

**Tests.** Unit: VAPID JWT verifies against a known public key; subscription
store add/dedupe/remove; tick() fans out to subscribers (mock the sender). E2E:
grant notifications, assert a subscription POST is made.

**Effort M.** The ES256 signer is the only fiddly part; everything else is small.

---

### 1.2 Model-augmented reflection (Phase 6 enhancement) — **M**

**Problem.** `evolve.py` reflection is purely deterministic heuristics. The
handoff flags this: let Claude write richer self-improvement proposals on top of
the heuristics, still gated behind the same approval inbox.

**Design.** Add `assistant.reflect(observations, telemetry_summary, memory)` that,
in **claude mode only**, asks the deep tier for up to N candidate proposals as
strict JSON (`kind` ∈ the existing allowed set, `title`, `body`, `rationale`).
Feed it the telemetry summary + recent tool outcomes + current memory/agent-notes.
`evolve.Reflection._observe()` merges model proposals with heuristic ones,
de-duplicating by normalized title.

**Guardrails (unchanged trust boundary).**
- Model proposals enter the **same** approval inbox; the auto-apply policy is
  untouched — only `memory_prune` auto-applies, `prompt_addendum` still needs a
  click. A model can *suggest* a prompt addendum but never self-applies it.
- Validate every model proposal server-side against a JSON schema; drop anything
  whose `kind` isn't in the allowlist. The model cannot invent new apply actions.
- Deep-tier budget already rate-caps how often this runs.

**Files.** `assistant.py` (new `reflect`), `evolve.py` (`_observe` merge),
`router.py` (a `reflect` task → deep tier, already tierable).

**Build plan.** (1) strict JSON tool/output for candidate proposals; (2) schema
validation + kind allowlist; (3) merge/dedupe in `_observe`; (4) claude-mode gate;
(5) telemetry event `kind:"reflect-model"` for the System widget.

**Tests.** Unit: schema rejects bad `kind`; dedupe against heuristic titles;
local-mode path produces zero model proposals (no SDK). Confirm the auto-apply
boundary still holds with a model-authored `prompt_addendum` (must stay pending).

**Effort M.**

---

### 1.3 Applied-evolution audit/history view — **S**

**Problem.** Phase 6 snapshots the hub before every apply (rollback exists) but
there's no way to *see* what was applied/dismissed over time, or to roll back from
the UI. Trust in self-evolution needs visibility.

**Design.** Append every apply/dismiss/rollback to `data/evolution_log.jsonl`
(bounded, like telemetry). New `GET /api/evolve/history` returns recent entries.
System widget grows an "EVOLUTION" expandable row → timestamp, kind, title,
outcome, and a "roll back" button for applied entries that still have a snapshot.

**Files.** `evolve.py` (log on apply/dismiss + `history()` + `rollback(id)`),
`server.py` (route), `public/js/evolve.js` + `widgets/system.js` (history panel).

**Tests.** Unit: apply writes a log line; rollback restores the snapshot and logs
it; history is newest-first and bounded.

**Effort S.** Mostly plumbing over machinery that already exists.

---

## Tier 2 — strong quality-of-life

### 2.1 News search box + in-widget filtering — **S**

**Problem.** Handoff lists a "news search box". The news widget shows a topic's
feed but you can't search within/across topics.

**Design.** A search input in the news widget header filters the merged item list
client-side (title/summary/source match, debounced). Optional server support:
`GET /api/news?q=` filters server-side across the selected topic before the TTL
cache so large feeds stay cheap. Keep it client-side first (zero API change).

**Files.** `public/js/widgets/news.js`, small CSS. Optional `server.py` `news()`
`q` param + cache-key suffix.

**Tests.** E2E: type a query, assert item count shrinks and matches highlight.

**Effort S.**

---

### 2.2 Agent multi-step plan preview — **M**

**Problem.** Handoff idea: before the agent executes a chain of tool calls, show
the user the plan. Today confirm-tier tools pause individually; there's no
holistic "here's what I'm about to do" for a multi-action turn.

**Design.** When a chat turn returns ≥2 `tool_use` blocks, the agent widget
renders a compact plan card (numbered actions with their tiers) and a single
"Run all / Run auto-only / Cancel" control, instead of prompting per confirm-tool.
Auto-tier steps run immediately; confirm-tier steps stay individually gated within
"Run all". Pure UX layer over the existing loop in `widgets/agent.js` +
`actions.js`; no server change.

**Files.** `public/js/widgets/agent.js` (plan card + batched execution),
`actions.js` (`describeAction` already exists — reuse for the preview lines), CSS.

**Tests.** E2E in local mode: issue a command that yields multiple actions (e.g.
"add task A and add task B"), assert the plan card lists both and "Run all"
executes them.

**Effort M.**

---

### 2.3 Routing-table overrides UI — **S**

**Problem.** Handoff idea. Tiers are env-configurable
(`HERMES_HUB_MODEL_FAST/_CORE/_DEEP`) but there's no UI; a user can't see or nudge
routing without restarting.

**Design.** A read-only routing table already ships in `assistant.status()` and
the System widget. Add a Settings panel that shows the current FAST/CORE/DEEP
model ids + the deep-budget state, with per-tier text overrides written to
`data/routing.json` (server reads it at startup and on a new
`POST /api/assistant/routing`). Overrides layer under env pins (env still wins, to
preserve the documented precedence).

**Files.** `router.py` (load `routing.json`, precedence env > file > default),
`server.py` (GET/POST route), `public/js/sources.js` or a new settings panel.

**Tests.** Unit: precedence env > file > default; bad model id rejected; deep
budget unaffected by an override.

**Effort S.**

---

### 2.4 Tasks upgrade: due dates, recurrence, priority — **M**

**Problem.** Tasks are flat text items. The briefing engine already *reasons*
about automatability; giving tasks structure (due date, priority, recurrence)
makes briefings and calendar integration far stronger.

**Design.** Extend the task item shape (`store.js` defaults + migration):
`{ id, text, done, due?, priority?, repeat? }`. Backward compatible — missing
fields default. Calendar widget overlays due tasks on their date. `add_task` tool
schema gains optional `due`/`priority`. Local command grammar learns
"…by Friday" / "…!high".

**Files.** `store.js` (shape + migration), `widgets/tasks.js`, `widgets/calendar.js`
(due overlay), `assistant.py` (tool schema + `parse_local_command`),
`actions.js` (add_task).

**Tests.** Unit: migration preserves old tasks; local parser extracts due/priority.
E2E: add a task with a due date, assert it appears on the calendar day.

**Effort M.**

---

## Tier 3 — depth & polish

### 3.1 Command palette → command *execution* (not just navigation) — **M**

**Problem.** Ctrl/⌘-K searches data and jumps to it. It could also *run* agent
commands ("add task…", "brief me") without opening the agent widget.

**Design.** Palette entries gain an "actions" section: typing a recognized command
routes through `parse_local_command` (local) or a one-shot chat turn (claude) and
shows the result inline. Reuses the entire existing command stack.

**Files.** `public/js/main.js` (palette), thin reuse of `actions.js`/`api.js`.

**Effort M.**

---

### 3.2 Per-widget refresh intervals + "last updated" affordance — **S**

**Problem.** Refresh cadences are hardcoded per widget (`ctx.every(...)`). Power
users may want faster markets or slower news.

**Design.** A small settings panel maps widget-type → interval, persisted in
state; `ctx.every` reads the override. Each widget header gains a subtle
"updated Xm ago" using the existing `timeAgo`.

**Effort S.**

---

### 3.3 Theming: accent picker + a second "amber/CRT" preset — **S**

**Problem.** The design system is token-driven (`--accent` etc.) but locked to
cyan. The tokens make alternate presets trivial.

**Design.** Settings accent picker writing `--accent`/`--accent-dim`/`--accent-ink`
to a `:root` style override persisted in state. Ship 3 curated presets (cyan,
amber, phosphor-green) that stay within the intelligence-agency aesthetic. The new
visual-polish glow layer already keys off `--accent`, so presets glow correctly
for free.

**Effort S.**

---

### 3.4 Backups: download/upload + scheduled off-box copy — **S**

**Problem.** Server-side backups exist (`/api/backup*`) but live on the same disk.
A container reset (noted as a real hazard) loses them unless pushed.

**Design.** Add `GET /api/backup/<name>` (download the JSON, name-validated like
restore) and `POST /api/backup/import` (upload a snapshot). Optionally a `backup`
automation action already exists — document a pattern to write into a synced
folder. Pure additive endpoints.

**Effort S.**

---

## Security & hardening backlog (from the audit)

These are smaller than features but worth scheduling:

1. **CSRF for state-changing POSTs when a token is set** — **S.** With a token the
   API is bearer-auth'd (not cookie-auth'd), so classic CSRF doesn't apply; but if
   a future feature adds cookie auth, add a same-site check / CSRF token. Track it.
2. **Rate-limit the reader/geocode endpoints** — **S.** A local tool, but the
   reader makes outbound fetches; a simple per-process token bucket in `TTLCache`
   style bounds abuse if the port is ever exposed.
3. **Structured error responses** — **S.** `_handle_api` leaks `str(exc)` on 500.
   Fine for localhost; when `--token` is set (i.e. exposed), return a generic
   message and log detail server-side.
4. **Subresource integrity / no external origins** — already clean (no CDN, no
   external fonts). Keep it that way; add a note to `JARVIS.md`.
5. **Optional: sandbox the viewer iframe** — **S.** Cross-origin isolation already
   protects the parent, so this is defense-in-depth for the embedded content only.
   `sandbox="allow-scripts allow-forms allow-popups allow-same-origin"` — verify
   it doesn't break common embeds before shipping.

---

## Sequencing recommendation

1. **Web Push (1.1)** — unlocks the automations engine's real value.
2. **Model-augmented reflection (1.2) + audit view (1.3)** — completes the Phase 6
   vision with the trust surface it needs.
3. **News search (2.1) + routing UI (2.3) + accent presets (3.3)** — a fast, high-
   visibility polish sprint.
4. **Tasks upgrade (2.4) + plan preview (2.2)** — the biggest UX step-up.
5. Fold the security backlog items in as each area is touched.

Everything above is stdlib-only (or reuses the already-optional `anthropic` SDK),
keeps the approval/permission trust boundaries intact, and extends existing
machinery rather than replacing it.
