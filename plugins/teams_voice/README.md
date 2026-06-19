# teams_voice — Microsoft Teams real-time voice/video (CVI) bridge driver

The **Python driver** half of the Conversational Video Interface (CVI) for
Microsoft Teams in Hermes. It is the port of the openclaw TypeScript `voice-call`
msteams provider (upstream PRs openclaw/openclaw #91438 + #92081).

> **Two processes, one bridge.** Teams real-time call media
> (`Microsoft.Skype.Bots.Media`) is **Windows/.NET-only**, so the avatar tile and
> RTP media are rendered by a separate **C# media worker** (`AzureBot` /
> `OpenClawBridge`). This plugin is the cross-platform *brain*: it hosts the
> WebSocket the worker dials into, runs dialogue + perception, and sends the
> avatar **driver cues**. The worker renders; this plugin drives.

```
Hermes (this plugin) ──HMAC WebSocket──▶ AzureBot C# media worker ──▶ Teams call
  • bridge_server.py (WS server)            • renders NV12 avatar tile
  • dialogue (realtime / streaming)         • samples inbound A/V, forwards DTMF
  • perception (vision ring)                • recording-status compliance gate
  • emits expression / visemes / image
```

The worker is the **WebSocket client**; this plugin is the **server**
(binds `127.0.0.1:8443` by default). Chat-plane features (messages, message
actions, meeting-recap posting) are handled by the existing
`plugins/platforms/teams` adapter (the `microsoft-teams-apps` SDK), **not** here.

## Status — implemented

| Layer | File | Status |
|---|---|---|
| Bridge WS server (HMAC, replay guard, lifecycle, ping/pong, conn caps) | `bridge_server.py` | ✅ |
| Wire protocol (mirrors worker `Protocol.cs`) | `protocol.py` | ✅ |
| Config (config.yaml + env, allowlist, vision cap, recap, …) | `config.py` | ✅ |
| **Realtime mode** (OpenAI/Azure speech-to-speech) | `realtime/openai_client.py`, `handlers.py` | ✅ |
| **Streaming mode** (STT→agent→TTS; needs `ffmpeg` on PATH) | `streaming_audio.py`, `handlers.py` | ✅ |
| Echo guard · group gate · verbal interrupts (EN/AR) · DTMF · bilingual | `echo_guard.py`, `group_call_gate.py`, `verbal_interrupts.py` | ✅ + tests |
| Vision (`look_at_screen` live/history) · ambient push · budget cap | `vision_store.py`, `vision_budget.py` | ✅ + tests |
| Tools: consult, agent_task, show_to_caller, call_me_back, post_meeting_minutes | `realtime_tools.py`, `handlers.py` | ✅ |
| Visemes (estimator; Latin) | `viseme_estimate.py` | ✅ + tests · Arabic + real ElevenLabs alignment = follow-up |
| Meeting recap/minutes (text; posts via Bot Framework REST) | `meeting.py` | ✅ · DOCX attach = follow-up |
| CLI: `hermes teams-voice serve --handler {logging,echo,realtime,streaming}` | `cli.py` | ✅ |

Validated on a live Teams call (realtime): connect, recording gate, expression
cues, avatar rendered, audio out, clean teardown. The Windows .NET media worker
renders the avatar tile and is paired over the HMAC bridge.

## Wire contract (fixed by the worker — do not drift)

* **Handshake:** `HMAC-SHA256(sharedSecret, "{timestampMs}.{callId}")`, lowercase
  hex, sent as `X-OpenClawTeamsBridge-Timestamp` / `-Signature` headers on the WS
  upgrade. ±60 s window; accepted `(callId, ts, sig)` tuples are single-use.
* **Path:** `/voice/msteams/stream/{callId}`.
* **Audio:** PCM 16 kHz, 16-bit, mono, little-endian; 20 ms / 640-byte frames, base64.
* **Messages** (camelCase JSON, additive): inbound `session.start` / `session.end`
  / `recording.status` / `audio.frame` / `video.frame` / `participants` / `dtmf`
  / `ping`; outbound `audio.frame` / `expression` / `speech.marks` /
  `display.image` / `assistant.cancel` / `pong`.

The `sharedSecret` here **must equal** the worker's `OpenClawSharedSecret`.

## Configure

Two sources are supported (per the Hermes docs); **config.yaml takes precedence, `.env`
is the fallback**. The recommended pattern keeps **secrets in `.env`** and references
them from config.yaml with `${VAR}` (the loader expands them), so config lives in one
declarative file without copying secrets around.

**config.yaml** (`%LOCALAPPDATA%\hermes\config.yaml`):

```yaml
plugins:
  enabled:
    - teams_voice
  entries:
    teams_voice:
      config:
        shared_secret: ${TEAMS_VOICE_SHARED_SECRET}   # secret stays in .env
        host: 127.0.0.1
        port: 8443
        realtime:
          backend: azure
          azure_endpoint: https://pcfcaoai2.cognitiveservices.azure.com
          azure_deployment: gpt-realtime
          azure_api_version: 2025-04-01-preview
          voice: cedar
          api_key: ${AZURE_FOUNDRY_API_KEY}           # secret stays in .env
          vad_threshold: 0.5
          prefix_padding_ms: 300
          silence_duration_ms: 500
```

**`.env`** (`%LOCALAPPDATA%\hermes\.env`) — the secret store (used directly, or referenced above):

```bash
TEAMS_VOICE_SHARED_SECRET=...        # must equal the worker's OpenClawSharedSecret
AZURE_FOUNDRY_API_KEY=...            # realtime key (also used by the gateway)
# fully env-only is fine too:
TEAMS_VOICE_HOST=127.0.0.1
TEAMS_VOICE_PORT=8443
TEAMS_VOICE_REALTIME_BACKEND=azure
TEAMS_VOICE_AZURE_ENDPOINT=https://pcfcaoai2.cognitiveservices.azure.com
TEAMS_VOICE_AZURE_DEPLOYMENT=gpt-realtime
TEAMS_VOICE_AZURE_API_VERSION=2025-04-01-preview
TEAMS_VOICE_REALTIME_VOICE=cedar
```

Each config.yaml key has a matching env var (e.g. `realtime.azure_endpoint` ↔
`TEAMS_VOICE_AZURE_ENDPOINT`); config.yaml wins where both are set.

## Run

```bash
hermes teams-voice status      # show config + readiness
hermes teams-voice serve       # run the bridge server (foreground)
# or, standalone:
python -m plugins.teams_voice.bridge_server
```

Point a worker instance's `OpenClawWsBaseUrl` at this server
(`ws://<host>:8443/voice/msteams/stream`) with a matching shared secret. One
worker identity per gateway — the worker's multi-identity config (`BridgeInstanceSettings`)
lets one host serve both the openclaw and Hermes gateways.

## Test

```bash
pytest plugins/teams_voice/tests/ -v
```

## Roadmap (next increments)

1. **Realtime client** (`realtime/openai_client.py`): OpenAI/Azure realtime over
   WS, 16k↔24k resampling, `expression`/`speech.marks` emission, barge-in.
2. **Dialogue handler**: a `CallSessionHandler` that owns the recording gate,
   echo guard, group-gate enforcement, and delegates real work to `run_agent`.
3. **Perception**: a 16-frame vision ring + `look_at_screen` / ambient push.
4. **Avatar tools**: `show_to_caller` → `display.image`.
5. **Outbound**: "call me back" via the worker's authenticated place-call endpoint.
6. **Arabic visemes** + bilingual parity.

See `C:\AzureBot\docs\CVI-STUDY-91438-92081.md` for the full feature/architecture study.
