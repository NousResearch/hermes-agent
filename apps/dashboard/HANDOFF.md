# HERMES//HUB — session handoff

Paste this into a new session to continue. It captures everything needed to
pick up cleanly.

## 1. What this is
A self-hosted, zero-dependency personal dashboard + "Jarvis" AI agent, living
in `apps/dashboard/` of the repo. Design language: dark-first, minimalist,
"intelligence-agency" aesthetic (mono labels, hairlines, cyan accent). It began
as an all-in-one widget board and grew into a full agent platform. The complete
Jarvis architecture is documented in `apps/dashboard/JARVIS.md`.

**Hard constraint / ethos:** zero-dependency **Python stdlib** server + zero-build
**vanilla ES-module** frontend. The only optional dependency is the official
`anthropic` SDK (unlocks the live agent). Do NOT add frameworks (no FastAPI,
React, bundlers, Docker-required flows, etc.). Everything must keep working with
just `python3 server.py`.

## 2. Repo, branch, git rules
- Repo: `DrZM007/hermes-agent` (aka `drzm007/hermes-agent`).
- **Work only on branch `claude/all-in-one-dashboard-xqh6ct`.** Never push elsewhere.
- Push with `git push -u origin claude/all-in-one-dashboard-xqh6ct`.
- Do NOT open a PR unless explicitly asked.
- Commit footer convention used so far (keep consistent):
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_<this-session-id>
  ```
- Commit + push after each self-contained bundle (a stop-hook nags about
  uncommitted changes).

## 3. How the user runs it (their setup)
- The user is on **Windows**, runs Python via `python3` (Python 3.14). **Docker
  does NOT work on their laptop** — use the no-Docker paths.
- Their repo clone lives in their home dir (`C:\Users\ziyaad.moolla\hermes-agent`).
- Daily start: `cd $HOME\hermes-agent\apps\dashboard` then `python3 server.py`,
  open `http://127.0.0.1:8787`. They also have a `start-dashboard.bat` on the Desktop.
- To get updates: `git pull origin claude/all-in-one-dashboard-xqh6ct`.
- The dashboard only runs on THEIR machine — servers spun up in the cloud
  session are not reachable from their laptop.
- Always-on without Docker: `apps/dashboard/deploy/serve.sh` (nohup runner),
  `deploy/hermes-hub.service` (systemd user unit), `deploy/com.hermeshub.hub.plist`
  (macOS launchd).

## 4. Architecture (backend = Python, frontend = JS)
Server relays a Claude conversation; tool calls execute **client-side** against
the browser's localStorage (nothing personal stored server-side except synced
state in SQLite). Agent loop lives in `public/js/widgets/agent.js` +
`public/js/actions.js`. SSE streaming for live replies.

### Jarvis layers — ALL 6 PHASES COMPLETE
1. **Model router** (`router.py`) — cost-aware tiers FAST=Haiku / CORE=Sonnet /
   DEEP=Opus; picks cheapest viable per task, escalates hard/security/finance
   turns to deep, deep-tier rate-capped. Env overrides
   `HERMES_HUB_MODEL_FAST/_CORE/_DEEP`; `HERMES_HUB_MODEL` pins one model.
2. **Permission gate** (`assistant.TOOL_TIERS`) — auto/confirm/blocked. Confirm
   tools (add_app, open_url, create/delete_automation) pop an approval card in
   the Agent widget; unknown/sensitive tools blocked. Mirrored client-side in
   `actions.js`.
3. **Telemetry + System widget** (`telemetry.py`, `public/js/widgets/system.js`)
   — bounded `data/telemetry.jsonl` of routing + tool outcomes; widget shows
   engine/tiers, deep budget, permission split, recent tool calls.
4. **Kill switch** — one toggle freezes all autonomy; `automations.tick()`
   checks a persisted `frozen` flag before firing. `GET/POST /api/killswitch`.
5. **Advisor escalation** — when the core model self-reports low confidence
   (`needs_escalation`), `advise()` consults the deep tier (guidance only, no
   tools), core finishes confidently. Bounded by deep budget. In both
   `_chat_claude` and `chat_stream`.
6. **Bounded self-evolution** (`evolve.py`) — reflection over telemetry+memory
   queues proposals in an approval inbox (⚙ → Agent proposals…, or System widget
   PROPOSALS row). **Policy: only `memory_prune` auto-applies**; `prompt_addendum`
   (learned guidelines → `data/agent_notes.md`, injected into system prompt)
   needs a click. Every apply snapshots the hub first (rollback). `reflect`
   automation action runs it nightly.

### Other major features (all shipped)
- Widgets: clock, worldstate (State-of-the-World index), agent, weather
  (multi-city + AQI + sunrise/sunset), launcher, news (custom RSS sources),
  reading list, tasks, notes, calendar (+ICS subscriptions), markets
  (watchlist), focus timer (Pomodoro), system.
- In-app viewer (reader + embed), summarize-everywhere (∑ buttons), voice
  (push-to-talk + speak replies), command palette (Ctrl/⌘-K) that also searches
  your own data and jumps to it.
- Cross-device sync (SQLite `data/hub.db`, optimistic concurrency), bearer-token
  auth + lock screen, PWA (manifest + service worker, currently **hub-v10**).
- Automations engine (`automations.py`): daily/market/worldstate triggers →
  notify/briefing/backup/reflect actions; 20s daemon thread.
- Server-side backups (`/api/backup`, `/api/backups`, `/api/backup/restore`);
  snapshots include memory **and** agent_notes.

### Post-phase feature additions (all shipped, tested)
- **News search** — filter box in the news widget; client-side, no refetch.
- **Accent presets** — settings swatches switch the UI accent (cyan default /
  amber / green / magenta); sets `--accent*` inline, persisted in state.
- **Structured tasks** — optional due date + priority via inline tokens
  (`!high`/`!low`, `@YYYY-MM-DD`/`@today`/`@tomorrow`); priority rail + overdue
  due chip; open tasks sort by priority then due; due tasks overlay the
  calendar; `add_task` tool + local parser learned the tokens.
- **Evolution rollback + audit** (Phase 6) — applied proposals show a one-click
  "Roll back" that restores the pre-apply snapshot; `evolve.history()` +
  `GET /api/evolve/history`; `rollback` op on `/api/evolve/proposal`.
- **Model-augmented reflection** (Phase 6) — in claude mode the deep tier
  proposes richer `prompt_addendum` guidelines into the same approval inbox
  (validated, capped, never auto-applies). `assistant.reflect_candidates()`.
- **Routing overrides UI** — "Model routing…" panel edits per-tier models,
  persisted in `data/routing.json`; precedence env > file > default (env-pinned
  tiers shown locked). `GET/POST /api/assistant/routing`.
- **SSRF hardening** — the reader resolves the host and rejects non-global
  addresses and re-validates every redirect hop (`host_is_blocked`); all
  upstream fetches capped at 8 MiB.

## 5. File map (`apps/dashboard/`)
```
server.py         stdlib HTTP server: static + JSON API + live/sample fallback
assistant.py      AI layer: Claude (optional SDK) or local rule-based; routing,
                  permission tiers, advisor escalation, summarize, briefing
router.py         cost-aware model routing (Phase 1)
telemetry.py      bounded routing + tool-call telemetry (Phase 3)
automations.py    standing-rules engine + kill switch (Phase 4)
evolve.py         self-evolution / reflection engine (Phase 6)
ics.py            minimal RFC 5545 calendar parser
sample_data.json  bundled offline data
Dockerfile        OPTIONAL container (user can't use Docker); prefer deploy/
compose.yaml      OPTIONAL
deploy/           serve.sh + systemd/launchd units (no-Docker always-on)
JARVIS.md         merged agent architecture + phase status
HANDOFF.md        this file
ROADMAP.md        detailed future-feature build plans
README.md         full user docs
public/           zero-build frontend
  js/main.js        layout, palette (data search), settings menu
  js/store.js       localStorage state + defaults + sync merge
  js/api.js         API client (+ SSE reader)
  js/actions.js     executes agent tool calls; TOOL_TIERS mirror
  js/evolve.js      Agent-proposals inbox panel (+ rollback/history)
  js/routing.js     Model-routing overrides panel
  js/sources.js / calendars.js   settings panels
  js/widgets/*.js   one module per widget
tests/test_server.py   140 unit tests
tests/e2e.mjs          109-check Playwright suite
.github/workflows/dashboard.yml  CI
```

## 6. Runtime data files (under `--data-dir`, default `data/`)
`hub.db` (synced state), `memory.md` (agent facts), `agent_notes.md` (learned
guidelines), `feeds.json`, `calendars.json`, `automations.json` (rules + frozen
flag + notifications), `telemetry.jsonl`, `proposals.json`, `routing.json`
(per-tier model overrides), `backups/*.json`.

## 7. Environment variables
- `HERMES_HUB_TOKEN` — access code (required when exposed beyond localhost).
- `HERMES_HUB_API_KEY` (or `ANTHROPIC_API_KEY`) — enables live Claude agent.
- `HERMES_HUB_MODEL` — pin one model for all tiers (back-compat).
- `HERMES_HUB_MODEL_FAST/_CORE/_DEEP` — override individual tiers.

## 8. Testing + verification standard
- **Unit:** `cd apps/dashboard && python3 -m unittest discover -s tests` (140 tests).
- **E2E:** Playwright against a running server. Standard to ship anything =
  full unit suite passes **plus 3 consecutive green e2e runs**.
- E2E needs `playwright-core` installed somewhere and Chromium at
  `/opt/pw-browsers/chromium`. Run pattern (two servers — open + token-locked):
  ```bash
  # install playwright-core into a scratch dir once: npm install playwright-core
  python3 apps/dashboard/server.py --offline --port 8787 --data-dir <fresh1> &
  python3 apps/dashboard/server.py --offline --port 8788 --token e2e-access-code --data-dir <fresh2> &
  PW_CORE_DIR=<scratch-with-node_modules> AUTH_URL=http://127.0.0.1:8788 \
    AUTH_TOKEN=e2e-access-code node apps/dashboard/tests/e2e.mjs http://127.0.0.1:8787 <shotsdir>
  ```
  Expected tail: `ALL E2E CHECKS PASSED`.

## 9. Environment / session hazards (IMPORTANT for the agent)
- **Backend changes require restarting the e2e servers** (server.py imports are
  loaded once). Frontend files are served fresh from disk (no restart needed).
- **`pkill`/`pgrep` can match your own shell → exit 144 "suicide".** Use a
  bracket pattern that won't match the literal command, e.g.
  `pgrep -f "port 87[89]" | xargs -r kill`. Never pattern-match a string present
  in your own command line.
- **Background bash cwd resets** — use absolute paths when starting servers.
- **Never delete a SQLite data dir while a server is using it** → "readonly
  database" 500s. Stop server → wipe → start.
- **Container restarts can roll the workspace back** to an old snapshot. Recover
  with `git fetch origin claude/all-in-one-dashboard-xqh6ct &&
  git reset --hard origin/claude/all-in-one-dashboard-xqh6ct`. Everything is
  pushed after each bundle, so nothing is lost.
- E2E resets client state + clears automations/calendars/killswitch at startup
  for idempotency; keep new e2e sections idempotent (clean up what you create).
- Model identity: this session runs as `claude-fable-5`; when asked which model,
  say the configured id. Do NOT put the model id in commits/PRs/code.

## 10. User preferences
- Replies: **normal English, terse and direct, no unnecessary words.** (There is
  NO "caveman skill" — that was a misunderstanding; write plainly.)
- The user tests on their own Windows laptop; give Windows-exact commands when
  troubleshooting. Docker is not an option for them.
- Confirm genuinely irreversible/self-modifying decisions before building (e.g.
  the Phase 6 auto-apply boundary was confirmed via a question).

## 11. Current status
- All 6 Jarvis phases + the post-phase additions, PLUS the big all-in-one
  expansion (see ROADMAP2.md): dashboard **pages/tabs** (Main/Markets/Feeds/
  Sports), a **crypto suite** (detail drawer, TA indicators, portfolio, global
  bar + Fear & Greed, trending), **stocks/indices/FX** (Stooq), **live sports
  scores + standings** (ESPN), a **socials hub** (HN/Lobsters/Reddit), a
  **gaming** widget (Epic free games + Steam deals) + gaming news topic, an
  **At-a-Glance hero**, **follow-any-search** news (Google News), and a **richer
  reader** (image/byline/reading time). Shared infra: `chart.js` (SVG line/
  candle/donut), `detail.js` (⤢ expand-window), `indicators.py`.
- New widget files: `glance, markets(detail), scores, socials, gaming, stocks`;
  new backend endpoints under `/api/crypto/*, /api/stocks*, /api/scores,
  /api/standings, /api/social, /api/gaming/*`; new data files none (all cached +
  sample-backed). PWA cache at **hub-v17**.
- Test counts: **187 unit / ~165 e2e checks**, all green (3/3 consecutive e2e),
  all pushed to `claude/all-in-one-dashboard-xqh6ct`.
- Detailed plans live in `ROADMAP.md` (phase-1 ideas) and `ROADMAP2.md` (the
  all-in-one expansion, with a shipped/pending status banner).

## 12. Open / future ideas (not yet built — see ROADMAP.md for full plans)
- **Web Push notifications** (Tier 1) — real push beyond in-app toasts; the
  recommended path is a payload-less VAPID "tickle" so it stays stdlib-only.
- **Agent multi-step plan preview** (Tier 2) — show a plan card before running
  a multi-tool turn, with Run-all / auto-only / cancel.
- **Command palette execution** (Tier 3) — run agent commands from ⌘-K, not
  just navigate.
- Smaller: per-widget refresh intervals; backup download/upload + off-box copy;
  task recurrence (the due/priority groundwork is now in place).
