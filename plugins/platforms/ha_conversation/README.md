# Home Assistant Conversation Agent platform plugin

This plugin registers Hermes as a **conversation agent** inside Home
Assistant's Assist pipeline, using the Wyoming `handle` protocol. HA keeps
owning wake word, speech-to-text, and text-to-speech on its existing voice
hardware (ESPHome devices like the Voice PE, S3-BOX, the HA companion app's
Assist); Hermes only receives the transcript and produces the reply text.

This **complements, not replaces**, the `voice_satellite` platform:
`voice_satellite` makes Hermes *behave like* a satellite/wake-word device
talking directly to HA's Assist pipeline (Hermes-initiated), while
`ha_conversation` makes Hermes the *conversation agent* HA calls into
(HA-initiated, per utterance). It also composes with the `homeassistant`
event-monitor/toolset — this plugin only answers utterances; entity control
during a turn goes through the same `homeassistant` toolset the rest of
Hermes uses.

## Architecture

```
ESPHome device (Voice PE, S3-BOX, HA app Assist)
   │ ESPHome native API (HA's side)
   ▼
Home Assistant — Assist pipeline (wake, STT, TTS)
   │ Wyoming handle: per-utterance TCP connection FROM HA
   ▼
hermes gateway → plugins/platforms/ha_conversation/  (this plugin, port 10600)
   Transcript → shared "home" session → agent turn → Handled(reply)
```

Each utterance opens a **fresh TCP connection from HA**, HA sends a single
`Transcript` event, and the plugin answers with exactly one `Handled` (or
`NotHandled`) event before the connection closes — one utterance per
connection, by design of the Wyoming `handle` protocol.

## Quick start

1. Enable the block in `~/.hermes/config.yaml` (uncomment/paste from
   `cli-config.yaml.example`):

   ```yaml
   ha_conversation:
     enabled: true
     bind_host: 127.0.0.1     # use your tailnet/LAN address if HA is remote
     port: 10600
     ack_after_seconds: 8.0
     announce_mode: "off"     # off | last_active | default_device | broadcast
     announce_entity: ""      # e.g. assist_satellite.kitchen (default_device mode)
     max_transcript_chars: 2000
   ```

2. In Home Assistant: **Settings → Devices & Services → Add Integration →
   Wyoming Protocol** → host = the machine running Hermes, port = `10600`
   (or whatever you set `port` to).
3. **Settings → Voice assistants →** your pipeline **→ Conversation agent =
   "hermes"**.
4. Speak via any Assist device (Voice PE, S3-BOX, the mobile app's Assist,
   the HA dashboard mic). The reply comes back through HA's existing TTS.

If HA runs on a different machine than Hermes, set `bind_host` to a
LAN/tailnet address Hermes is reachable on (the default `127.0.0.1` only
accepts connections from the same host) and make sure the port is reachable
through any firewall between them.

## One shared conversation

Every room/device talks into the **same `home` gateway session** —
there is no per-satellite or per-speaker session. That's deliberate: it's
what lets you start a thought in the kitchen and finish it in the living
room. The privacy tradeoff is explicit: anything said to *any* Assist device
in the house lands in the same conversation history and context Hermes
already has for you, and one voice interrupting mid-answer from another room
shares that same turn.

## Slow turns: ack and announce modes

Some turns (tool calls, long generations) take longer than HA's pipeline
would like to wait. If a reply isn't ready after `ack_after_seconds`
(default `8.0`), the plugin answers the open connection early with a short
acknowledgement so HA doesn't time out or show an error, then keeps working
in the background and delivers the real answer per `announce_mode`. The ack
text itself is honest about where the eventual answer will surface:

| `announce_mode`   | Ack text says…                                                      | Where the late reply actually goes                                  |
|--------------------|---------------------------------------------------------------------|-----------------------------------------------------------------------|
| `off` (default)    | "I'm still working on that. I'll keep the answer in our conversation." | Nowhere spoken — stays in the session transcript only.                |
| `last_active`      | "I'm still working on that. I'll announce the answer when it's ready." | The satellite that most recently sent a transcript.                   |
| `default_device`   | "I'm still working on that. I'll announce the answer when it's ready." | `announce_entity` (an `assist_satellite.*` entity you configure).     |
| `broadcast`        | "I'm still working on that. I'll announce the answer when it's ready." | All Assist satellites (`assist_satellite.announce` target `all`).     |

Announcements are delivered via Home Assistant's own
`assist_satellite.announce` service call (through the `homeassistant`
toolset's REST helpers), so HA renders the TTS — this plugin never
synthesizes audio itself.

A late reply that answers a **specific question** always targets the room
that asked it, regardless of `announce_mode`'s general routing: if Room A
asked something slow and Room B asks something else in the meantime, Room
A's eventual answer only ever plays in Room A (or nowhere, under `off`), never
in Room B.

One declared design corner: this per-room targeting only applies to a slow
*voice* turn's own late reply. A background delivery that isn't tied to any
open voice turn (e.g. a cron job finishing) arriving **while a voice turn is
actively being processed** merges straight into that turn's spoken reply
instead of following `announce_mode` — the shared `home` session has no way
to distinguish "more from the same turn" from "an unrelated delivery" once a
turn window is open.

## Home control

`supports_home_control` is advertised to HA automatically — not a config
key — whenever a `HASS_TOKEN` (Long-Lived Access Token) is configured
(the same credentials the `homeassistant` platform and toolset use). The HA
instance URL comes from `HASS_URL`, defaulting to `http://homeassistant.local:8123`;
set `HASS_URL` explicitly if your instance lives elsewhere, since announce/control calls
will otherwise target that default hostname. When a token is present,
device-control intents spoken during a conversation turn ("turn off the
kitchen lights") execute through Hermes's existing `homeassistant` toolset
inside that turn, the same way they would from text chat.

## Security model

The Wyoming `handle` protocol carries **no per-speaker identity** — HA
tells this plugin what was said, not who said it. Practically:

- Anyone within microphone range of an Assist device, or anything on the
  network that can reach the bound port, is talking to your agent with its
  full toolset and conversation history, exactly as if they'd typed to it.
- The default bind is `127.0.0.1` (loopback only); exposing the port to a
  LAN or tailnet is an explicit, opt-in choice you make via `bind_host`.
- There is **no per-utterance confirmation for destructive actions** in this
  first version — anything the agent's normal approval policy allows from
  text chat, it will also do from a spoken command reaching this port.

Treat the port the same way you'd treat physical access to your voice
hardware: don't expose it further than you'd want a guest (or a stranger on
your Wi-Fi) to be able to talk to Hermes.

## Limitations

- **No streaming replies.** The Wyoming `handle` protocol answers with one
  `Handled` event per connection — there is no partial/incremental delivery.
- **No Wyoming intent-service mode.** This plugin implements the `handle`
  (conversation-agent) role only, not a Home Assistant intent-recognition
  service.
- **English-only ack text.** The acknowledgement strings are not localized.
- **One utterance in flight per turn slot, with a queue cap.** Utterances
  are serialized behind a single turn lock; up to **8** utterances may be
  in flight at once — one actively being processed under the lock, the
  rest queued behind it — beyond that, new requests get an immediate "I'm
  handling several requests right now — try again in a moment." instead of
  queuing further.
