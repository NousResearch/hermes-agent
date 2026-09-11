# google_meet plugin

Let the hermes agent join a Google Meet call, transcribe it, optionally speak
in it, and do the followup work afterwards.

## What ships

| Version | What | Status |
|---|---|---|
| v1 | Transcribe-only: Playwright joins Meet, scrapes captions to transcript file | ✓ ships by default |
| v2 | Realtime duplex audio: bot speaks in-call via OpenAI Realtime + BlackHole/PulseAudio null-sink | ✓ opt in with `mode='realtime'` |
| v3 | Remote node host: run the bot on a different machine than the gateway | ✓ opt in with `node='<name>'` |

## Architecture

```
┌─ gateway (Linux box, where hermes runs) ────────────────────────────┐
│                                                                      │
│   agent → meet_join(url, mode='realtime', node='my-mac')             │
│         │                                                            │
│         └─ NodeClient ─── ws ────┐                                   │
│                                  │                                   │
└──────────────────────────────────┼───────────────────────────────────┘
                                   │ wss (token auth)
                                   ▼
┌─ node host (user's Mac, signed-in Chrome lives here) ───────────────┐
│                                                                      │
│   NodeServer (from `hermes meet node run`)                           │
│     │                                                                │
│     ├─ start_bot → process_manager.start() → spawns meet_bot         │
│     │                                                                │
│     └─ meet_bot (Playwright)                                         │
│        ├─ Chromium → meet.google.com                                 │
│        ├─ caption scraper → transcript.txt                           │
│        └─ (realtime mode only) RealtimeSpeaker thread                │
│             ↓                                                        │
│           OpenAI Realtime WS → speaker.pcm                           │
│             ↓                                                        │
│           paplay → null-sink ← Chrome fake mic                       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

Without v3: the whole right column runs on the gateway machine.
Without v2: the "realtime" path is skipped; transcribe runs alone.

## Files

| Path | Purpose |
|---|---|
| `plugin.yaml` | manifest |
| `__init__.py` | `register(ctx)` — registers 5 tools + `on_session_end` hook + `hermes meet` CLI |
| `meet_bot.py` | Playwright bot subprocess (standalone, `python -m plugins.google_meet.meet_bot`) |
| `process_manager.py` | local bot lifecycle + `enqueue_say` |
| `tools.py` | agent-facing tools + node-routing helper |
| `cli.py` | `hermes meet setup / auth / join / status / transcript / say / stop / node ...` |
| `audio_bridge.py` | v2: PulseAudio null-sink (Linux) + BlackHole probe (macOS) |
| `realtime/openai_client.py` | v2: `RealtimeSession` + `RealtimeSpeaker` (file-queue → OpenAI Realtime WS → PCM) |
| `node/protocol.py` | v3: message envelope + validation |
| `node/registry.py` | v3: `$HERMES_HOME/workspace/meetings/nodes.json` |
| `node/server.py` | v3: `NodeServer` (runs on host machine) |
| `node/client.py` | v3: `NodeClient` (used by tool handlers + CLI on gateway) |
| `node/cli.py` | v3: `hermes meet node {run,list,approve,remove,status,ping}` |
| `SKILL.md` | agent usage guide |

## Local quick start

```bash
hermes plugins enable google_meet
hermes meet install                                      # pip + Chromium
hermes meet setup                                        # preflight
hermes meet auth                                         # optional
hermes meet join https://meet.google.com/abc-defg-hij    # transcribe
```

## Realtime mode

Linux (preferred, most automated):
```bash
hermes meet install --realtime                     # installs pulseaudio-utils
echo 'OPENAI_API_KEY=sk-...' >> ~/.hermes/.env
hermes meet join https://meet.google.com/abc-defg-hij --mode realtime
# then from the agent or CLI:
hermes meet say "Good morning everyone, I'm the note-taker bot."
```

macOS:
```bash
hermes meet install --realtime     # runs: brew install blackhole-2ch ffmpeg
# then — manually! — open System Settings → Sound → Input → BlackHole 2ch
echo 'OPENAI_API_KEY=sk-...' >> ~/.hermes/.env
hermes meet join https://meet.google.com/abc-defg-hij --mode realtime
```

On macOS, hermes will **not** switch your system audio input automatically — the
user has to do it. This is deliberate: switching default input on a whim would
be a surprising side effect.

## Remote node host

On the node machine (e.g. user's Mac with a signed-in Chrome):
```bash
pip install playwright websockets
python -m playwright install chromium
hermes plugins enable google_meet
hermes meet node run --display-name my-mac --host 0.0.0.0 --port 18789
# prints the bearer token on first run; copy it
```

On the gateway:
```bash
hermes meet node approve my-mac ws://<mac-ip>:18789 <token>
hermes meet node ping my-mac
# now any meet_* tool call accepts node='my-mac' (or 'auto')
```

## Safety

- URL gate: only `https://meet.google.com/abc-defg-hij`, `/new`, `/lookup/<id>`.
- No calendar scanning, no auto-dial, no auto-consent announcement.
- Node server uses bearer-token auth; no key exchange, no TLS termination
  built in — run it on a LAN or behind a reverse proxy you trust.
- One active meeting per (gateway, node) pair. A second `meet_join` leaves the first.
- `meet_say` refuses unless the active meeting was started with `mode='realtime'`.

## Out of scope

- **Calendar scanning** — deliberately not implemented. Join URLs must be explicit.
- **Multi-tenant node sharing** — a node serves one gateway at a time.
- **Windows** — audio bridging isn't tested; `register()` no-ops on Windows.
- **System audio input switching on macOS** — user responsibility, not the bot's.
