# HERMES//HUB — all-in-one personal dashboard

A self-contained personal command center: your most-used apps, news of every
kind, a state-of-the-world situation board, weather, markets, task lists,
notes, a calendar — and a built-in AI agent that can act on all of it.
Minimalist, dark-first, intelligence-agency aesthetic.

Everything you click opens **inside** the dashboard: articles get a
server-side reader view, apps open in an embedded viewer (with a one-click
external fallback for sites that refuse framing).

## Quick start

```bash
cd apps/dashboard
python3 server.py            # → http://127.0.0.1:8787
```

That's it — no build step, no dependencies. Python 3.10+ and a browser.

```bash
python3 server.py --offline  # demo mode: bundled sample data, no network
python3 server.py --port 9000 --host 0.0.0.0 --token my-secret-code
```

If live feeds are unreachable the server automatically falls back to bundled
sample data and the UI marks affected widgets with a `DEMO DATA` chip.

## Use it from your phone (Jarvis anywhere)

The dashboard is a full PWA with cross-device sync: your lists, notes,
events, apps and agent history live in a small SQLite database next to
`server.py` (`data/hub.db`) and every device converges on it — edits flush
within ~1.5 s, on tab close/background via `sendBeacon`, and remote changes
are picked up by a 25 s poll (plus on tab focus).

**The host machine must be on** for remote access — the server, your data
and the agent all run there. A laptop works if it doesn't sleep; a Raspberry
Pi or any always-on box is the comfortable long-term home (the server is
dependency-free Python).

1. **Lock it.** Any time the server is reachable beyond localhost, start it
   with `--token <code>` (or `HERMES_HUB_TOKEN`). Phones/browsers get a lock
   screen; the API refuses everything (except `/api/health`) without the
   code. The static shell is public by design — the data lives behind the API.
2. **Reach it.** Pick one:
   - **Tailscale (recommended):** install on host + phone, sign in to the
     same tailnet, then open `http://<host-tailscale-ip>:8787` from anywhere.
     No ports opened to the internet, traffic is WireGuard-encrypted.
   - **Same Wi-Fi only:** `--host 0.0.0.0` and open `http://<laptop-ip>:8787`.
   - **Cloudflare Tunnel / Caddy:** for a public HTTPS hostname; keep the
     token on and prefer adding real TLS in front.
3. **Install it.** Open the URL on the phone → browser menu →
   *Add to Home Screen*. You get an app icon, standalone window, and an
   offline shell that shows the last-fetched data with no signal.

Note: iOS treats plain-HTTP sites as insecure for some PWA features; over
Tailscale it works, but for the full experience put HTTPS in front (Caddy
with a local CA, or Cloudflare Tunnel).

### Keep it always-on (no Docker required)

Because the server is dependency-free Python, the simplest way to keep it
running on your laptop or an always-on box is a background process or your OS's
own service manager — no container runtime needed.

**Simplest — background runner** (any Unix, no root, no install):

```bash
cd apps/dashboard
HERMES_HUB_TOKEN=my-secret-code ./deploy/serve.sh start   # → http://localhost:8787
./deploy/serve.sh status   # running?
./deploy/serve.sh logs     # follow the log
./deploy/serve.sh stop
```

**Start on boot / survive logout:**

- **Linux (systemd):** `deploy/hermes-hub.service` — a *user* unit (no root).
  Copy it to `~/.config/systemd/user/`, edit the path + token, then
  `systemctl --user enable --now hermes-hub` and
  `loginctl enable-linger "$USER"`. Logs: `journalctl --user -u hermes-hub -f`.
- **macOS (launchd):** `deploy/com.hermeshub.hub.plist` — copy to
  `~/Library/LaunchAgents/`, edit the paths + token, then
  `launchctl load ~/Library/LaunchAgents/com.hermeshub.hub.plist`.

Each template documents its own install/stop steps at the top.

**Optional — Docker** (only if you already run it, e.g. a NAS/VPS): a
`Dockerfile` + `compose.yaml` ship too — `HERMES_HUB_TOKEN=… docker compose up -d`.
State persists in the `hub-data` volume. Not required for laptop use.

## The agent (optional AI mode)

The **◆ Agent** widget is a Jarvis-style operator for the dashboard. It ships
with two engines:

| Mode | When | What you get |
|---|---|---|
| **LOCAL** | default, zero setup | command parser (`add task…`, `complete…`, `open GitHub`, `add event tomorrow: …`, `show tech news`, `brief me`), extractive summaries, rule-based daily briefing with automation suggestions |
| **CLAUDE** | `pip install anthropic` + `ANTHROPIC_API_KEY` (or `ant auth login`), then restart | full conversational agent that plans, chats, and calls dashboard tools; model-written summaries and briefings |

In both modes the agent can **act** (add/complete tasks, events, notes,
launcher apps, open URLs in the viewer, switch news topics — executed in your
browser against your local data) and **research** (read the news, fetch and
read an article, check the weather, markets and world state — served from the
dashboard's own feeds). It also has **long-term memory**: say
"remember: …" and the fact persists in `data/memory.md` across sessions
("what do you remember" reads it back; in Claude mode it is injected into the
agent's context automatically).

**Cost-aware routing (Claude mode).** The agent doesn't burn the biggest model
on every turn. A router (`router.py`) picks the cheapest viable tier per task —
Haiku for summaries, Sonnet for chat, Opus only for hard/ambiguous/security- or
finance-sensitive turns — and rate-caps the expensive tier. Override any tier
with `HERMES_HUB_MODEL_FAST` / `_CORE` / `_DEEP`, or pin one model for
everything with `HERMES_HUB_MODEL`.

**Advisor escalation.** If the core model's own answer signals it's stuck, the
deep tier is consulted as a *scoped advisor* (guidance only, no tools) and the
core model produces a confident answer with that guidance — you'll see a brief
"escalating to a deeper model…" note when it happens. It's bounded by the same
deep-tier budget, so it never runs away, and each escalation shows in the
System widget.

**Permission gate.** Every tool call is classified server-side (Jarvis Layer F,
not agent-editable). Read-only and reversible edits (tasks, notes, events,
research) run instantly; outward-facing or unattended-effect actions (adding a
launcher app, opening a URL, arming or deleting an automation) pop an
**approval card** in the Agent widget and wait for your click; anything
unclassified is refused. See `JARVIS.md` for the full agent architecture and
roadmap.

**System widget.** A live status panel shows the active engine and model tiers,
the deep-tier budget, the permission split, and a feed of recent tool calls
(name · tier · outcome, including anything you denied). It reads from bounded
server telemetry (`data/telemetry.jsonl`) — routing decisions plus tool
outcomes, tool *names* only, never their arguments or results.

**Kill switch.** The System widget has one **freeze** toggle that halts all
autonomous behaviour — no automation fires while it's engaged (the freeze is
checked before every evaluation, server-side, so it holds even with no browser
open). It persists across restarts and shows a red banner when active.

**Self-evolution (bounded).** The agent periodically reflects on its own
telemetry and memory and proposes small, reversible improvements — de-duping
memory, or learning a guideline like "stop offering X, the user keeps declining
it." Review them in ⚙ → *Agent proposals…* (or the System widget's PROPOSALS
row). **Only memory cleanup applies automatically**; prompt/behaviour changes
wait for your click. Applied changes touch data files only (never code) and the
hub is snapshotted first, so anything can be rolled back. A `reflect` automation
action can run it nightly.

Every widget (and every news story, and the article viewer) has a **∑**
button that summarizes that piece of data with the active engine. In
Claude mode, replies stream in live (SSE) and type out as they arrive.

### Automations — set-and-forget Jarvis

Standing orders run **server-side** (they fire even with no browser open) and
surface as in-app toasts plus system notifications (tap **🔔 Alerts** in the
Agent widget once to allow them). Create them by just telling the agent:

- "every morning at 7:30 brief me" → a daily auto-generated briefing
- "alert me if BTC moves 5%" → edge-triggered 24 h-change alert
- "alert me when the world reaches elevated" → situation-board level alert
- "list automations" / "delete automation 2" to manage them

Rules live in `data/automations.json`; triggers are `daily`, `market` and
`worldstate`; actions are `notify` or `briefing`. The engine evaluates every
20 s and keeps the last 100 notifications for clients to poll
(`/api/notifications?after=<id>`).

## Features

- **State of the World** — per-domain stability index (geopolitics, economy,
  tech/cyber, climate, health, markets) computed by transparent keyword
  analysis of current headlines, with expandable explanations and the exact
  headline "signals" behind each score. Informational, not authoritative.
- **News** — curated RSS/Atom sources across 7 default topics, deduped and
  merged; read stories in-app via the reader. Fully customizable: ⚙ →
  *News sources…* to add any feed (sites, YouTube channels via
  `/feeds/videos.xml?channel_id=…`, subreddits via `.rss`, podcasts) or
  create whole new topics that appear as tabs. Config lives server-side
  (`data/feeds.json`) so every device sees the same tabs.
- **Apps launcher** — tiles for your most-used sites, opening in the in-app
  viewer; add/edit/remove in edit mode.
- **Weather** — Open-Meteo (no key needed), 24 h temperature chart with hover
  tooltip, 7-day outlook, air-quality (US AQI) chip, sunrise/sunset, and up to
  5 saved cities as tabs (city search to add, ✕ in edit mode to remove).
- **Calendar feeds** — subscribe to read-only ICS calendars (⚙ → *Calendar
  feeds…*): Google/Apple/Outlook private iCal addresses, webcal links, any
  .ics URL. Events (including daily/weekly/monthly/yearly recurrence) merge
  into the Calendar widget, upcoming list, briefings and the agent's context.
  Config is server-side (`data/calendars.json`), refreshed every 15 min.
- **Markets** — editable watchlist (up to 15 assets; "+ watch asset" with any
  CoinGecko id, remove in edit mode) with price, 24 h change and 7-day
  sparkline (CoinGecko, no key needed). The watchlist syncs across devices.
- **Voice** — 🎙 push-to-talk on the agent (browsers with SpeechRecognition,
  e.g. Chrome) and a 🔊 toggle to have replies read aloud.
- **Reading list** — tap 🔖 on any story to save it; opened stories are
  tracked and dimmed everywhere, with one-tap "clear read". Synced across
  devices like everything else.
- **Lists, Notes, Calendar** — multiple task lists with progress, autosaving
  notes, month calendar with events. All stored in `localStorage`.
- **Focus timer** — a minimalist Pomodoro (25 / 50 focus, 5 break) with a
  reload- and sync-proof countdown (only the target end-time is stored, so
  every device shows the same remaining time) and a daily session log.
- **Universal search** — engine picker + bang prefixes (`g` `ddg` `yt` `w`
  `gh`), `/` focuses search, `Ctrl/⌘-K` opens the command palette. The palette
  also searches your own data — tasks, notes, events, reading list and apps —
  and jumps to the match (pulsing the widget it lives in).
- **Layout** — drag-and-drop reordering, per-widget resize, add/remove
  widgets, masonry packing, dark/light/auto theme, export/import backup.

## Layout of this app

```
server.py          stdlib HTTP server: static files + JSON API + live/sample fallback
assistant.py       AI layer: Claude (optional SDK) or local rule-based engines
automations.py     standing rules engine (daily/market/worldstate → notify/briefing)
ics.py             minimal RFC 5545 parser for subscribed calendars
sample_data.json   bundled offline data (news, weather, markets, geocode)
deploy/            always-on service without Docker: serve.sh runner +
                   systemd (Linux) and launchd (macOS) unit templates
Dockerfile         optional container build (compose.yaml); not needed for laptops
public/            zero-build frontend (ES modules, design-system CSS)
  js/widgets/      one module per widget (clock, worldstate, agent, …)
  js/viewer.js     in-app reader/embed overlay
  js/actions.js    executes agent tool calls against local state
tests/
  router.py        cost-aware model routing (Jarvis Layer I)
  telemetry.py     bounded routing + tool-call telemetry (Jarvis Phase 3)
  evolve.py        bounded self-evolution / reflection engine (Jarvis Phase 6)
  test_server.py   140 unit tests (feeds+sources, worldstate, reader, assistant,
                   sync, auth, automations, memory, watchlist, SSE, ICS,
                   backups, model router, permission tiers, telemetry,
                   kill switch, advisor escalation, self-evolution, HTTP)
  e2e.mjs          109-check Playwright suite (needs playwright-core + Chromium)
                   — also runs in CI (.github/workflows/dashboard.yml)
```

## Tests

```bash
cd apps/dashboard
python3 -m unittest discover -s tests -v

# end-to-end (server must be running; needs playwright-core somewhere)
node tests/e2e.mjs http://127.0.0.1:8787 /tmp/shots
```

## Privacy notes

- Personal data (tasks, notes, events, apps, agent chat log) lives in your
  browser's `localStorage` and, for cross-device sync, in `data/hub.db` on
  the machine running the server — never anywhere else. Use ⚙ → Export for
  backups.
- The reader endpoint refuses to fetch private/internal addresses.
- In CLAUDE mode, your message plus a snapshot of dashboard context (task
  texts, event titles, note titles, headlines) is sent to the Anthropic API
  for that request.
