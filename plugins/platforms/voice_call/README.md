# voice_call — phone calls for Hermes

Gateway platform plugin that lets Hermes make and receive real phone calls
through carrier APIs: **Telnyx (default)**, Twilio, Plivo, or a
credential-free **mock** provider for development. Ported from OpenClaw's
voice-call extension.

The plugin registers a gateway platform (so the carrier webhook server
starts and stops with `hermes gateway run` — inbound calls need an
always-on listener), the `voice_call` model tool, the `hermes voicecall`
CLI, and the `/voicecall` slash command.

User documentation: `website/docs/user-guide/features/voice-call.md`.

## Dev quickstart (mock provider, no credentials)

```yaml
# ~/.hermes/config.yaml
gateway:
  platforms:
    voice_call:
      enabled: true
      extra:
        provider: mock
        from_number: "+15555550000"
        inbound_policy: allowlist
        allow_from: ["+15555550001"]
```

```bash
hermes gateway run                 # starts the webhook server on 127.0.0.1:3334
hermes voicecall status --json
hermes voicecall call --to +15555550001 --message "Hello from Hermes"
hermes voicecall status
```

Simulate an inbound call + caller speech (mock provider accepts
pre-normalized events):

```bash
curl -X POST http://127.0.0.1:3334/voice/webhook \
  -H 'content-type: application/json' \
  -d '{"event":{"type":"call.initiated","direction":"inbound","provider_call_id":"c1","from":"+15555550001","to":"+15555550000"}}'
curl -X POST http://127.0.0.1:3334/voice/webhook \
  -H 'content-type: application/json' \
  -d '{"event":{"type":"call.answered","provider_call_id":"c1"}}'
curl -X POST http://127.0.0.1:3334/voice/webhook \
  -H 'content-type: application/json' \
  -d '{"event":{"type":"call.speech","provider_call_id":"c1","text":"what time is it?"}}'
```

The transcript becomes a normal gateway agent turn; the reply is "spoken"
back through the provider (visible in `hermes voicecall tail`).

## Telnyx live smoke checklist

1. `~/.hermes/.env`: `TELNYX_API_KEY`, `TELNYX_CONNECTION_ID`,
   `TELNYX_PUBLIC_KEY` (from the Telnyx portal's webhook signing section).
2. Config: `provider: telnyx`, your `from_number`, and either `public_url`
   (your reverse proxy) or `tunnel: {provider: ngrok}` + `NGROK_AUTHTOKEN`.
3. `hermes voicecall doctor` — everything should read `present` / `ok`.
4. `hermes gateway run`, then
   `hermes voicecall call --to +1... --message "Hello from Hermes" --mode notify`.

## Architecture

```
adapter.py    thin BasePlatformAdapter shell (connect/disconnect/send)
runtime.py    singleton owning provider + store + manager + webhook server
manager.py    call state machine, timers, transcript waiters
store.py      JSONL persistence + boot restore-and-verify
webhook.py    aiohttp server: signature verify → replay cache → policy → manager
responder.py  caller speech → MessageEvent → gateway agent → spoken reply
providers/    base ABC, mock, telnyx, twilio, plivo
realtime/     speech-to-speech bridge (OpenAI Realtime / Gemini Live)
```

Tests: `scripts/run_tests.sh tests/plugins/platforms/voice_call
tests/gateway/test_voice_call_platform.py -q`
