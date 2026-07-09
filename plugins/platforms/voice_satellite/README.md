# Voice satellite platform plugin

This plugin connects Hermes Agent to [Wyoming](https://github.com/rhasspy/wyoming)
voice satellites — small always-on devices (Raspberry Pi + ReSpeaker, an old
laptop, anything that runs [wyoming-satellite][ws]) placed around your home.
Say the wake word, ask a question, and the full agent — memory, skills, tools,
automations — answers out loud through the satellite's speaker.

Wake-word detection runs **on the satellite**; Hermes does STT with your
configured `stt` provider, runs the agent turn, and speaks the reply with your
configured `tts` provider. Each satellite is one gateway session, so background
deliveries (cron results, completed tasks) are announced through the speaker
too.

[ws]: https://github.com/rhasspy/wyoming-satellite

## Architecture

Like Discord and Slack, this is a **persistent-connection** channel — no
public URL, no webhook. The direction is inverted from a typical server,
though: the satellite is the TCP listener (default port 10700) and **Hermes
dials out** to every satellite listed in config, reconnecting with backoff if
the link drops. Satellites never need to know where Hermes lives.

```
Satellite (wyoming-satellite, on-device wake word)
   ▲ persistent TCP :10700 — Hermes dials out    [Wyoming: JSONL events + PCM]
   ▼
hermes gateway → plugins/platforms/voice_satellite/
   run-pipeline + audio-chunks → RMS endpointing → stt provider
   → VOICE MessageEvent → agent turn
   → tts provider → ffmpeg → s16le PCM → satellite speaker
```

## Requirements

- The `wyoming` Python package: `pip install 'hermes-agent[satellite]'`
  (or let the lazy-dependency prompt install it on first use).
- `ffmpeg` on PATH — TTS output is transcoded to raw PCM for the satellite.
- A satellite running [wyoming-satellite][ws] with a wake-word service
  (e.g. `--wake-uri` pointing at openWakeWord). See its docs for the
  hardware-side install.

## Quick start: first smoke test

What you need: any Linux box with a microphone and speaker (a laptop works;
no Pi required), plus Hermes with an `stt:` and `tts:` provider already
configured in `~/.hermes/config.yaml` — the round trip uses both.

On the satellite machine:

```bash
# 1. The wake-word service (separate process, its own Wyoming server)
git clone https://github.com/rhasspy/wyoming-openwakeword.git
cd wyoming-openwakeword && script/setup
script/run --uri 'tcp://127.0.0.1:10400' &

# 2. The satellite itself
git clone https://github.com/rhasspy/wyoming-satellite.git
cd wyoming-satellite && script/setup
script/run \
  --name kitchen \
  --uri 'tcp://0.0.0.0:10700' \
  --mic-command 'arecord -r 16000 -c 1 -f S16_LE -t raw' \
  --snd-command 'aplay -r 22050 -c 1 -f S16_LE -t raw' \
  --wake-uri 'tcp://127.0.0.1:10400' \
  --wake-word-name 'ok_nabu'
```

Notes:
- There is no "Hey Hermes" wake model; test with a stock openWakeWord model
  such as `ok_nabu` ("Okay Nabu"). Custom wake words can be trained later.
- The `aplay -r 22050` rate must match `tts_sample_rate` below (both default
  to 22050).
- Wrong mic or speaker? List devices with `arecord -L` / `aplay -L` and pass
  `-D plughw:...` inside the respective command.

On the Hermes machine: add the `voice_satellite:` block below with the
satellite's address, then `hermes gateway start`.

Success looks like: the gateway log shows the Wyoming handshake for
`kitchen` at startup; saying "Okay Nabu" logs `listening`; your question
logs `heard: <transcript>`; the reply plays through the satellite speaker.
If the handshake never appears, test reachability first:
`python -c "import socket; socket.create_connection(('<satellite>', 10700), timeout=5)"`.

## Configuration

In `~/.hermes/config.yaml`:

```yaml
voice_satellite:
  enabled: true
  satellites:
    - name: kitchen
      host: 192.168.1.40   # satellite IP or hostname
      port: 10700
  listen_timeout_seconds: 30.0
  tts_sample_rate: 22050
  endpointing:
    silence_threshold: 200      # RMS below this = silence (0-32767)
    silence_duration: 1.2       # trailing silence that ends an utterance
    min_speech_seconds: 0.5     # shorter = discard as noise
    max_utterance_seconds: 20.0 # hard cap per utterance
```

STT and TTS provider choice stays in the existing `stt:` / `tts:` sections.
For the lowest perceived latency run local providers (`faster-whisper` +
`piper`) on a machine with spare CPU; cloud providers can beat local models
on small servers.

Start with `hermes gateway start` and watch the logs for the Wyoming
`describe`/`info` handshake — that confirms the link is up. Boot order does
not matter: if a satellite is offline at startup, Hermes keeps retrying and
connects whenever it appears.

## Security model

Wyoming has **no authentication or encryption**. Listing a satellite in
`satellites:` is the trust grant: anything reachable at that address is
treated as an authorized speaker for your agent. This is safe on a private
network and only there.

- **Never port-forward 10700 to the internet.** An exposed satellite port
  lets anyone stream audio to your speakers and feed transcripts into your
  agent.
- Keep satellites and the Hermes host on the same LAN, VLAN, or VPN
  (see below), and firewall the port from everything else.

## Remote Hermes (VPS / Docker) over Tailscale

Hermes does not need to run in the same building as the satellites — all
audio hardware lives on the satellite; the server side only processes PCM
bytes. The one problem to solve is reachability: satellites sit on your home
LAN behind NAT, so a VPS cannot dial `192.168.x.x:10700` directly. A
[Tailscale](https://tailscale.com) (or WireGuard) mesh fixes this while
preserving the private-network security model.

1. **Join both sides to the tailnet** — the VPS and each satellite host.
   On the satellite, find its address with `tailscale ip -4`.
2. **Bind the satellite to the tailnet interface.** Simplest is all
   interfaces: `--uri 'tcp://0.0.0.0:10700'` (then firewall the LAN side if
   you want tailnet-only access).
3. **Point config at the tailnet address:**

   ```yaml
   voice_satellite:
     enabled: true
     satellites:
       - name: kitchen
         host: 100.101.102.103   # tailscale ip -4 on the satellite
         port: 10700
   ```

4. **Docker: usually nothing to do.** With Tailscale on the VPS *host*,
   containers on the default bridge network reach tailnet IPs out of the
   box (outbound traffic NATs through the host, which routes
   `100.64.0.0/10` via `tailscale0`). Verify from inside the container:

   ```bash
   docker exec <hermes> python -c \
     "import socket; socket.create_connection(('100.101.102.103', 10700), timeout=5); print('ok')"
   ```

   Caveats:
   - **MagicDNS names may not resolve in the container** (bridge containers
     use Docker's DNS, not the host's Tailscale resolver). If DNS fails, use
     the raw `100.x` IP in config — either works.
   - If Tailscale runs **as a container** instead of on the host, share its
     network namespace (`network_mode: service:tailscale`) or use
     `network_mode: host` for the Hermes container.

5. **Optional hardening with tailnet ACLs** — make the trust model
   enforceable by allowing only the Hermes server to reach satellite ports:

   ```json
   {"action": "accept", "src": ["tag:hermes-server"], "dst": ["tag:voice-satellite:10700"]}
   ```

Bandwidth is negligible (~32 KB/s of 16 kHz PCM while speaking, idle
otherwise) and the reconnect loop absorbs flaky WAN links. Expect a little
extra turn latency from the round trip; if the VPS is small, benchmark cloud
STT/TTS against local models before assuming local is faster.

## Limitations (v1)

- One turn at a time: wake word → utterance → spoken reply. The follow-up
  window (mic re-opens after a reply), spoken "working on it" acks for slow
  turns, and barge-in are planned follow-ups.
- Wake-word detection must run on the satellite (`--wake-uri`); server-side
  wake is not supported.
- Satellites are dialed out to from config; Wyoming server/zeroconf mode
  (satellites discovering Hermes) is not supported.
- ESPHome-native devices (Home Assistant Voice PE, ESP32-S3-BOX) need
  Wyoming-capable firmware; stock firmware speaks the ESPHome protocol,
  which this plugin does not implement.
