# google_meet plugin

Let the hermes agent join a Google Meet call, transcribe it, optionally speak
in it, and do the followup work afterwards.

## What ships

| Version | What | Status |
|---|---|---|
| v1 | Transcribe-only: Playwright joins Meet, scrapes captions to transcript file | ✓ ships by default |
| v2 | Realtime speech out: bot speaks in-call via OpenAI Realtime + BlackHole/PulseAudio null-sink; input stays caption-derived | ✓ opt in with `mode='realtime'` |
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
| `__init__.py` | `register(ctx)` — registers 5 tools + session-finalize cleanup hook + `hermes meet` CLI |
| `meet_bot.py` | Playwright bot subprocess (standalone, `python -m plugins.google_meet.meet_bot`) |
| `process_manager.py` | local bot lifecycle + `enqueue_say` |
| `queue_io.py` | shared locked JSONL queue; durable message IDs and removal without losing producer appends |
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

Caption updates revise their canonical transcript row rather than appending every
partial hypothesis. Rows with unresolved speakers remain visible; separate rows
are not merged merely because their text is similar.

## Local quick start

```bash
hermes plugins enable google_meet
hermes meet install                                      # pip + Chromium
hermes meet setup                                        # preflight
hermes meet auth                                         # optional saved Google state
hermes meet join https://meet.google.com/abc-defg-hij    # transcribe as guest
# or explicitly reuse saved Google auth:
hermes meet join --use-auth-state https://meet.google.com/abc-defg-hij
```

## Configuration

Google Meet behavior is profile-aware and belongs in `config.yaml` on the
machine that runs the bot (the gateway for local calls, or the node host for
remote calls):

```yaml
google_meet:
  debug_status: false
  xvfb: auto                 # Linux only: auto | force | disabled
  proxy:
    server: ""
    bypass: null             # null = pinned media default; "" = no bypass
  realtime_ready_timeout: 15
  stall_after: 90
```

The proxy's WebRTC policy always disables direct UDP when a proxy is set so
media cannot silently bypass it. Keep credentials such as `OPENAI_API_KEY` in
`.env`; the internal `HERMES_MEET_*` child variables are not user settings.

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

Realtime mode is **speak-only**: `speaker.pcm` is streamed into the virtual mic by a
stdin-fed `paplay` / `ffmpeg` pump that follows the file as Realtime appends audio.
Incoming speech is still the caption scrape — meeting audio is never sent to the
Realtime session, so there is no barge-in on raw audio and no STT billing. The bot
enables its Meet microphone only after verifying the realtime session and audio
pump. `meet_say` additionally requires an in-call bot with `localMicrophoneOn=true`,
`realtimeAudioPumpStatus="ready"`, and a live PCM pump process. If that process or
its PCM stream fails after admission, readiness is revoked and the bot leaves with
`leaveReason="realtime_audio_route_failed"` instead of queuing silent speech.
Transcription-only mode keeps both microphone and camera off and refuses to join
if their state cannot be verified.

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

`--use-auth-state` / `use_auth_state=true` is local-only on the gateway. Remote
nodes must manage Google auth on the node host; the gateway will reject
`use_auth_state` when `node` is set instead of silently starting as guest.

## Safety

- URL gate: only `https://meet.google.com/abc-defg-hij`, `/new`, `/lookup/<id>`.
- No calendar scanning, no auto-dial, no auto-consent announcement.
- Node server uses bearer-token auth; no key exchange, no TLS termination
  built in — run it on a LAN or behind a reverse proxy you trust.
- Guest mode is the default. Saved Google auth from `hermes meet auth` is reused
  only with `--use-auth-state` / `use_auth_state=true`, because it changes the
  identity and meeting permissions used to join.
- Hermes session-finalize cleanup leaves active calls by default, even with a
  duration set. Use `--persist-after-session` / `persist_after_session=true`
  only when the user explicitly wants a detached bot.
- One active meeting per (gateway, node) pair. A second `meet_join` leaves the first.
- `meet_say` refuses unless the active meeting was started with `mode='realtime'`
  and the bot is in-call with the realtime audio pump and Meet microphone ready.

## Out of scope

- **Calendar scanning** — deliberately not implemented. Join URLs must be explicit.
- **Multi-tenant node sharing** — a node serves one gateway at a time.
- **Windows** — audio bridging isn't tested; `register()` no-ops on Windows.
- **System audio input switching on macOS** — user responsibility, not the bot's.
- **Meeting-audio ingestion into Realtime** — input is caption-derived; true bidirectional audio is a separate feature.
