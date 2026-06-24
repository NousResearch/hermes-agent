# Voice-Native WhatsApp Stack

This document describes the local Hermes + `voice` integration path for
WhatsApp voice notes and WhatsApp Calling/WebRTC experiments.

The target shape is:

- `voice` is the source of truth for speech audio formats.
- Hermes asks `voice` for WhatsApp-ready Ogg/Opus files for voice notes.
- Hermes sends native Opus voice notes without a redundant ffmpeg conversion.
- Hermes uses a local WebRTC sidecar for WhatsApp Calling SDP and PCM frame
  exchange.
- `voice` keeps the local media contract stable through
  `voice stream-contract`.

## Components

| Component | Role |
| --- | --- |
| `voice say --format ogg-opus` | Writes completed WhatsApp-ready Ogg/Opus voice-note files. |
| `voice stream` | Emits raw 48 kHz mono 20 ms PCM frames or streamed Ogg/Opus files. |
| `voice stream-transcribe` | Transcribes files or decoded PCM frames from the sidecar. |
| `voice-webrtc-sidecar.service` | Runs the Python `aiortc` sidecar from the `voice` checkout. |
| `hermes-gateway.service` drop-in | Points the running gateway at the checkout and sidecar stream command. |
| `scripts/install_voice_local_stack.py` | Produces or applies the local systemd/config wiring. |
| `scripts/verify_voice_local_stack.py` | Proves the isolated and optional live local voice stack. |
| `scripts/verify_voice_live_gateway.py` | Proves the installed gateway process and sidecar are in sync. |
| `scripts/verify_voice_whatsapp_cloud_webhook.py` | Proves local Cloud webhook health, verify-token, signed POST handling, and inbound voice-note media caching without live Meta calls. |

## Install Plan

Install and start the `voice` daemon before applying the Hermes wiring. On
Linux, `voice daemon install` registers and starts the `voiced.service`
systemd user unit. The sidecar unit generated below orders after that daemon
and pulls it in with `Wants=voiced.service`.

```bash
voice daemon install
voice daemon status
```

Run the installer in dry-run mode first. The output is JSON with the files it
would write and the verifier commands to run afterward.

```bash
python3 scripts/install_voice_local_stack.py \
  --hermes-home ~/.hermes \
  --live-hermes-root ~/.hermes/hermes-agent-voice-stack \
  --voice-repo ~/code/src/github.com/rgbkrk/voice \
  --voice-bin ~/.local/bin/voice \
  --voice-daemon-service voiced.service \
  --webrtc-python-bin /tmp/voice-webrtc-venv/bin/python \
  --configure-tts \
  --configure-stt
```

Use `--apply` to write the systemd user unit, gateway drop-in, and optional
Hermes TTS/STT config updates. Use `--restart-hermes` when the gateway should
reload the new drop-in immediately.

```bash
python3 scripts/install_voice_local_stack.py \
  --apply \
  --restart-hermes \
  --hermes-home ~/.hermes \
  --live-hermes-root ~/.hermes/hermes-agent-voice-stack \
  --voice-repo ~/code/src/github.com/rgbkrk/voice \
  --voice-bin ~/.local/bin/voice \
  --voice-daemon-service voiced.service \
  --webrtc-python-bin /tmp/voice-webrtc-venv/bin/python \
  --configure-tts \
  --configure-stt
```

The generated TTS provider asks `voice` for native Opus:

```yaml
tts:
  provider: kokoro
  providers:
    kokoro:
      type: command
      command: voice say --format ogg-opus --input-file {input_path} --output {output_path} --voice {voice} --speed {speed}
      output_format: ogg
      voice_compatible: true
```

The generated STT provider routes files through `voice stream-transcribe`:

```yaml
stt:
  enabled: true
  provider: voice
  providers:
    voice:
      type: command
      command: voice stream-transcribe --quiet {input_path}
      format: txt
```

## Verification

The installer prints seven verifier commands under `verify_commands`:

| Key | Use |
| --- | --- |
| `local_stack` | Isolated local checks without touching the running gateway. |
| `live_gateway` | Running-gateway checks only. |
| `live_gateway_cloud_only` | Running-gateway checks only, skipping local Baileys bridge health. |
| `live_gateway_cloud_ready` | Same as `live_gateway_cloud_only`, plus non-contacting WhatsApp Cloud credential, recipient-authorization, local webhook health, verify-token handshake, and signed POST readiness. |
| `local_stack_live_gateway` | Full local stack plus running-gateway checks. |
| `local_stack_live_gateway_cloud_only` | Same as `local_stack_live_gateway`, but skips local Baileys bridge health. |
| `local_stack_live_gateway_cloud_ready` | Same as `local_stack_live_gateway_cloud_only`, plus the WhatsApp Cloud readiness gate. |

For a local Cloud-API-focused deployment, run the generated
`local_stack_live_gateway_cloud_only` command. It should include:

```bash
python3 scripts/verify_voice_local_stack.py \
  --voice-bin ~/.local/bin/voice \
  --voice-repo ~/code/src/github.com/rgbkrk/voice \
  --webrtc-python-bin /tmp/voice-webrtc-venv/bin/python \
  --live-hermes-root ~/.hermes/hermes-agent-voice-stack \
  --run-live-gateway \
  --calling-sidecar-url http://127.0.0.1:8787 \
  --live-gateway-python-bin ~/.hermes/hermes-agent/venv/bin/python \
  --live-gateway-hermes-home ~/.hermes \
  --run-live-gateway-stt-smoke \
  --run-live-gateway-calling-live-sidecar-smoke \
  --live-gateway-sidecar-service voice-webrtc-sidecar.service \
  --live-gateway-voice-daemon-service voiced.service \
  --skip-live-gateway-bridge-health
```

Before pointing Meta webhooks at the host, run the generated
`local_stack_live_gateway_cloud_ready` command. It adds
`--require-live-gateway-whatsapp-cloud-readiness`, which verifies
`WHATSAPP_CLOUD_PHONE_NUMBER_ID`, `WHATSAPP_CLOUD_ACCESS_TOKEN`,
`WHATSAPP_CLOUD_APP_SECRET`, `WHATSAPP_CLOUD_VERIFY_TOKEN`, and either
`WHATSAPP_CLOUD_ALLOWED_USERS` or `WHATSAPP_CLOUD_ALLOW_ALL_USERS`, then probes
the running local WhatsApp Cloud `/health` endpoint and local Meta subscription
challenge handshake. It also sends a signed synthetic delivery-receipt POST to
prove inbound HMAC verification accepts Meta-shaped webhook delivery without
dispatching an agent message or printing secret values. If setup is incomplete,
the verifier reports every missing or malformed Cloud setting it can detect in
one redacted failure.

That aggregate verifier proves:

- `voice` can produce WhatsApp-ready Ogg/Opus.
- `voice stream-contract` matches the checked-in sidecar contract.
- Hermes command TTS returns `voice_compatible: true` Opus voice-note media.
- Hermes command STT transcribes through `voice stream-transcribe`.
- `voice stream` returns raw 48 kHz mono 20 ms PCM frames.
- The local Baileys bridge keeps native `.ogg`/`.opus` media as voice notes
  when the bridge check is not skipped.
- The WhatsApp Cloud adapter uploads `.ogg`/Opus with
  `audio/ogg; codecs=opus`.
- The WhatsApp Cloud webhook path answers local health and verify-token
  handshakes, accepts signed status-only POST payloads without dispatching
  agent messages, and accepts a signed synthetic audio webhook that caches an
  Opus/Ogg voice note for downstream STT dispatch.
- The synthetic WhatsApp Calling control plane gates accept on sidecar
  readiness.
- A real local sidecar can answer an `aiortc` SDP offer and report
  `ready_for_accept: true`.
- The deployed Hermes checkout can handle a synthetic WhatsApp Calling
  `connect` webhook, POST the offer to the configured local sidecar, send
  `pre_accept` and `accept`, move PCM in both directions, and close the
  sidecar session on `terminate`.
- The installed `hermes-gateway.service` process imports from the expected
  checkout and points at the expected sidecar URL.
- The installed `voiced.service` daemon is active and runs the expected
  `voice daemon start` command for live-call stream TTS.
- The installed `voice-webrtc-sidecar.service` runs the expected `voice` binary
  and `voice` checkout.

## Readiness Boundary

The local WebRTC sidecar exposes `state.ready_for_accept` and per-check
`state.readiness` booleans. Hermes should send WhatsApp Graph `pre_accept` and
`accept` only after the sidecar reports `ready_for_accept: true`.

This local proof does not contact Meta's Graph API, receive a real WhatsApp
Calling webhook, or prove media against a real WhatsApp client. It proves the
local Hermes/voice boundary up to the point where an external WhatsApp
Calling event can be handed to the sidecar and accepted safely.

## Troubleshooting

- If `--run-sidecar-offer-smoke` fails, inspect the sidecar `/health` and
  `/contract` endpoints first.
- If Opus checks fail, make sure `ffmpeg` and `ffprobe` are installed and on
  `PATH`.
- If the live verifier imports old code, confirm the gateway systemd drop-in
  sets `PYTHONPATH` to the intended checkout.
- If the sidecar service check fails, confirm `VOICE_BIN`, `WorkingDirectory`,
  `--host`, and `--port` in `systemctl --user show voice-webrtc-sidecar.service`.
