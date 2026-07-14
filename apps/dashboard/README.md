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

## The agent (optional AI mode)

The **◆ Agent** widget is a Jarvis-style operator for the dashboard. It ships
with two engines:

| Mode | When | What you get |
|---|---|---|
| **LOCAL** | default, zero setup | command parser (`add task…`, `complete…`, `open GitHub`, `add event tomorrow: …`, `show tech news`, `brief me`), extractive summaries, rule-based daily briefing with automation suggestions |
| **CLAUDE** | `pip install anthropic` + `ANTHROPIC_API_KEY` (or `ant auth login`), then restart | full conversational agent (Claude Opus 4.8 by default, override with `HERMES_HUB_MODEL`) that plans, chats, and calls dashboard tools; model-written summaries and briefings |

In both modes the agent can **act** (add/complete tasks, events, notes,
launcher apps, open URLs in the viewer, switch news topics — executed in your
browser against your local data) and **research** (read the news, fetch and
read an article, check the weather, markets and world state — served from the
dashboard's own feeds). It also has **long-term memory**: say
"remember: …" and the fact persists in `data/memory.md` across sessions
("what do you remember" reads it back; in Claude mode it is injected into the
agent's context automatically).

Every widget (and every news story, and the article viewer) has a **∑**
button that summarizes that piece of data with the active engine.

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
  tooltip, 7-day outlook, city search.
- **Markets** — editable watchlist (up to 15 assets; "+ watch asset" with any
  CoinGecko id, remove in edit mode) with price, 24 h change and 7-day
  sparkline (CoinGecko, no key needed). The watchlist syncs across devices.
- **Voice** — 🎙 push-to-talk on the agent (browsers with SpeechRecognition,
  e.g. Chrome) and a 🔊 toggle to have replies read aloud.
- **Lists, Notes, Calendar** — multiple task lists with progress, autosaving
  notes, month calendar with events. All stored in `localStorage`.
- **Universal search** — engine picker + bang prefixes (`g` `ddg` `yt` `w`
  `gh`), `/` focuses search, `Ctrl/⌘-K` opens the command palette.
- **Layout** — drag-and-drop reordering, per-widget resize, add/remove
  widgets, masonry packing, dark/light/auto theme, export/import backup.

## Layout of this app

```
server.py          stdlib HTTP server: static files + JSON API + live/sample fallback
assistant.py       AI layer: Claude (optional SDK) or local rule-based engines
automations.py     standing rules engine (daily/market/worldstate → notify/briefing)
sample_data.json   bundled offline data (news, weather, markets, geocode)
public/            zero-build frontend (ES modules, design-system CSS)
  js/widgets/      one module per widget (clock, worldstate, agent, …)
  js/viewer.js     in-app reader/embed overlay
  js/actions.js    executes agent tool calls against local state
tests/
  test_server.py   73 unit tests (feeds+sources, worldstate, reader, assistant,
                   sync, auth, automations, memory, watchlist, HTTP)
  e2e.mjs          74-check Playwright suite (needs playwright-core + Chromium)
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
