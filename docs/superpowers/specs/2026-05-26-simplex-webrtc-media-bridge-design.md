# SimpleX WebRTC Media Bridge Design

Date: 2026-05-26
Status: approved design direction, pending written-spec review

## Goal

Implement a real private SimpleX-native WebRTC media bridge for Hermes, while preserving the existing Tailnet browser `/call` fallback. The first implementation must prove native SimpleX compatibility before layering on full speech-to-text, local inference, text-to-speech, and speech-to-speech behavior.

Native support means Hermes can answer an authorized SimpleX app call, complete SimpleX call signaling, establish a compatible WebRTC media endpoint, exchange controlled audio, report state transitions, and end cleanly. It is not enough to detect a call and send a browser fallback link.

## Scope

In scope:

- Incoming SimpleX native calls from authorized direct contacts.
- Private-only call handling.
- SimpleX call signaling through the existing `simplex-chat` daemon WebSocket.
- A WebRTC media endpoint controlled by Hermes.
- Loud user-facing failure messages and structured redacted logs.
- Tailscale-only browser fallback for `/call`.
- Hexagonal boundaries around signaling, media, voice runtime, session state, and platform delivery.
- Test-driven vertical slices.

Out of scope for the first native milestone:

- Group calls.
- Public call URLs.
- Tailscale Funnel.
- Multi-user rooms.
- Full speech-to-speech before SimpleX-native WebRTC compatibility is proven.
- Replacing the existing messaging gateway session model.
- Sending raw audio, tokens, URLs, or encryption keys to logs.

## Research Findings

SimpleX native calls are scriptable at the signaling layer. The official SimpleX source exposes hidden commands including:

- `/_call get`
- `/_call invite @<contactId> <CallType JSON>`
- `/_call reject @<contactId>`
- `/_call offer @<contactId> <WebRTCCallOffer JSON>`
- `/_call answer @<contactId> <WebRTCSession JSON>`
- `/_call extra @<contactId> <WebRTCExtraInfo JSON>`
- `/_call end @<contactId>`
- `/_call status @<contactId> connecting|connected|disconnected|failed`

The upstream call types show that `WebRTCSession` stores compressed JSON strings for SDP and ICE candidate arrays. The SimpleX WebRTC package uses browser APIs, `RTCPeerConnection`, `getUserMedia`, compressed SDP/ICE serialization, and optional encoded insertable streams for AES-GCM media encryption.

The SimpleX CLI/WebSocket API does not expose raw microphone, speaker, RTP, Opus, or PCM streams. It can move signaling messages, but Hermes must own or host a compatible WebRTC media endpoint.

The useful OpenClaw material is architectural, not a drop-in SimpleX bridge:

- Realtime session lifecycle.
- Audio source/sink boundaries.
- PCM format conversion and pacing.
- Interruption/barge-in shape.
- Loud failure codes.
- State cleanup.

## Recommended Approach

Use a browser/WebRTC sidecar as the first native SimpleX media endpoint, with all SimpleX-specific protocol behavior isolated behind ports.

This sidecar should reuse or closely drive the upstream SimpleX WebRTC browser behavior rather than recreating it in Python. This reduces risk around compressed SDP/ICE payloads and insertable-stream AES-GCM behavior.

Python-native `aiortc` remains a possible later adapter, but it should not be the first implementation because it would require Hermes to recreate the least-documented part of SimpleX calling.

LiveKit remains the browser fallback path, not the native SimpleX bridge. It can also become the browser media-room provider for `/call browser`, but it does not answer SimpleX app calls by itself.

## Architecture

Use hexagonal architecture inside a vertical-slice delivery plan.

The domain/application core owns call state and decisions. Adapters own external details. Ports make those boundaries explicit.

```text
gateway command / SimpleX event
        |
        v
Call application service
        |
        +-- SimplexCallSignalingPort
        |       -> SimpleX daemon WebSocket adapter
        |
        +-- WebRTCMediaPort
        |       -> browser sidecar adapter
        |       -> possible future aiortc adapter
        |
        +-- VoiceRuntimePort
        |       -> compatibility test runtime
        |       -> local STT -> Hermes -> local TTS runtime
        |
        +-- BrowserFallbackPort
        |       -> Tailnet browser room provider
        |
        +-- CallSessionStore
                -> in-memory MVP, optional persistent state later
```

### Application Core

Suggested package:

```text
gateway/calls/
  __init__.py
  manager.py
  models.py
  tokens.py
  browser_room.py
  native/
    __init__.py
    application.py
    ports.py
    simplex_signaling.py
    media.py
    voice_runtime.py
```

The existing `gateway/calls` package already contains browser call state, token, and Tailnet URL logic. The native bridge should extend that package without putting SimpleX-specific logic into `gateway/run.py`.

Core responsibilities:

- Enforce private-only calls.
- Enforce authorization before call setup.
- Allow one active call per `(platform, chat_id, user_id)` for MVP.
- Own call state transitions.
- Own TTLs, ring timeouts, and cleanup.
- Map internal failures to user-facing messages.
- Emit redacted structured logs.

### Ports

`SimplexCallSignalingPort`

- `list_pending_invitations()`
- `reject(contact_id, reason_code)`
- `send_offer(contact_id, offer)`
- `send_answer(contact_id, answer)`
- `send_extra_ice(contact_id, extra)`
- `send_status(contact_id, status)`
- `end(contact_id)`

`WebRTCMediaPort`

- `start_session(call_id, media, encryption)`
- `accept_offer_or_create_offer(...)`
- `apply_answer(...)`
- `add_remote_ice(...)`
- `read_audio_frames()`
- `write_audio_frames(frames)`
- `stop_session(call_id)`

`VoiceRuntimePort`

- `start(call_id)`
- `on_audio_frame(frame)`
- `on_remote_speech_start()`
- `on_remote_speech_end()`
- `emit_audio_frames()`
- `stop(call_id)`

The first implementation of `VoiceRuntimePort` should be a compatibility runtime that can exchange a known tone, silence, or loopback frames. The full voice runtime comes after native WebRTC is verified.

`BrowserFallbackPort`

- `create_room_url(source)`
- `end_room(call_id)`

`CallSessionStore`

- `create(session)`
- `get(route_key)`
- `transition(call_id, from_state, to_state, reason_code=None)`
- `end(call_id)`
- `expire(now)`

## State Machine

Primary native path:

```text
idle
  -> invitation_received
  -> authorized
  -> media_starting
  -> offer_sent
  -> connecting
  -> connected
  -> active
  -> ending
  -> ended
```

Failure paths:

```text
unauthorized -> rejected
unsupported_encryption -> failed
simplex_ws_disconnected -> failed
media_start_failed -> failed
signaling_timeout -> failed
connection_timeout -> failed
remote_ended -> ended
local_end_requested -> ended
```

Browser fallback path:

```text
native_unavailable
  -> browser_room_created
  -> browser_link_sent
  -> browser_waiting
  -> browser_active
  -> ended
```

`/call native` must not silently fall back to browser. `/call` may use browser fallback according to existing product decision, but logs must show why native was not used.

## Command UX

Existing command contract remains:

- `/call`: reliable browser fallback by default across private channels, with later SimpleX-native auto mode only when stable.
- `/call browser`: explicit Tailnet browser fallback.
- `/call native`: explicit SimpleX-native call only.
- `/call status`: current call state.
- `/call end`: terminate the active call.

Incoming SimpleX app calls are the primary native-call experience. Until the native bridge is enabled and healthy, Hermes rejects authorized native calls with a clear fallback message and rejects unauthorized calls without disclosing details.

## Vertical Slices

### Slice 1: Native Signaling Domain

Write tests first for:

- call invitation normalization;
- authorization gate;
- direct-chat-only enforcement;
- command payloads for `reject`, `offer`, `answer`, `extra`, `status`, and `end`;
- state transitions;
- loud failure messages.

Deliver:

- native call application service;
- signaling port and fake adapter;
- SimpleX daemon adapter methods for signaling only;
- no media claims yet.

### Slice 2: Browser Sidecar Compatibility Harness

Write tests first for:

- sidecar process lifecycle;
- JSON-RPC or WebSocket control protocol;
- timeout behavior;
- media capability errors;
- redacted logging.

Deliver:

- sidecar adapter behind `WebRTCMediaPort`;
- controlled fake media runtime;
- no full STT/TTS yet.

### Slice 3: SimpleX Native Compatibility Proof

Write tests first for:

- incoming invitation starts media sidecar;
- offer/answer/extra ICE commands flow through the signaling port;
- connected status maps into call state;
- failed sidecar setup rejects or ends the call loudly;
- no token, SDP, ICE, or encryption key leaks in logs.

Manual verification:

- Call Hermes from the SimpleX mobile app.
- Hermes answers through the native bridge.
- The call reaches connected or produces a concrete failure code.
- A controlled audio path is verified.
- `/call status` reflects the native session.
- `/call end` ends the native session cleanly.

This is the first point where Hermes may claim native SimpleX call compatibility.

### Slice 4: Local STT/TTS Voice Runtime

Write tests first for:

- inbound frame buffering;
- turn boundary detection;
- STT failure handling;
- Hermes agent invocation boundary;
- TTS output frame pacing;
- interruption and cancellation.

Deliver:

- local STT -> Hermes agent -> local TTS runtime;
- clear provider config;
- all non-local providers disabled by default for private voice calls.

### Slice 5: Streaming And Message UI Polish

Write tests first for:

- SimpleX live message creation with `live=on`;
- message update/finalize behavior;
- message ID extraction from correlated responses;
- markdown formatting compatibility;
- fallback when live update fails.

Deliver:

- SimpleX live text streaming for call transcripts and response summaries;
- markdown rendering aligned with SimpleX expectations.

## Privacy And Security

Rules:

- Native calls are private-only.
- SimpleX daemon WebSocket remains loopback-only.
- Browser fallback remains Tailnet-only unless a future explicit public exposure flag is enabled.
- Tokens are short-lived and scoped to platform, chat, user, call, nonce, and expiry.
- Store only token hashes where storage is needed.
- Never log room URLs, JWTs, SDP payloads, ICE payloads, AES keys, raw audio, transcripts marked private, or bearer links.
- Logs may include call ID, platform, redacted chat/user IDs, state, and reason code.
- If SimpleX media encryption is negotiated and unsupported, fail loudly with `call_simplex_native_encryption_unsupported`.
- Hermes necessarily sees decrypted audio locally when it transcribes and responds.

## Loud Failure Model

Required reason codes:

- `call_private_chat_required`
- `call_auth_denied`
- `call_already_active`
- `call_simplex_ws_disconnected`
- `call_simplex_native_unavailable`
- `call_simplex_native_timeout`
- `call_simplex_native_encryption_unsupported`
- `call_simplex_native_signaling_failed`
- `call_simplex_native_media_failed`
- `call_sidecar_start_failed`
- `call_sidecar_protocol_failed`
- `call_media_bridge_failed`
- `call_browser_provider_missing`
- `call_public_url_missing`
- `call_public_exposure_disabled`
- `call_token_expired`
- `call_join_auth_failed`

Every failure must produce:

- a structured log entry with reason code;
- a user-visible message when safe;
- a call state transition to `failed`, `ended`, or `rejected`.

## Dependency Strategy

Avoid adding a WebRTC Python dependency in the first slice.

If a sidecar package is needed, pin Node dependencies intentionally and keep the sidecar isolated under a clear package directory. Do not add large or cloud-specific packages to core dependencies.

If Python dependencies are added later, follow the repository dependency policy: exact pins in this checkout and lock regeneration when required.

## Test Strategy

Follow test-driven development:

1. Write a failing test for one behavior.
2. Verify the failure is the expected missing behavior.
3. Implement the smallest passing code.
4. Run the targeted tests.
5. Refactor while green.

Test layers:

- domain unit tests for state machine and failure mapping;
- fake signaling adapter tests;
- fake WebRTC media adapter tests;
- SimpleX adapter command serialization tests;
- gateway command tests;
- log-redaction tests;
- manual SimpleX mobile verification.

The implementation must not claim completion without manual verification of a SimpleX app call or a clearly documented blocker.

## Acceptance Criteria

First native compatibility milestone is accepted when:

- authorized incoming SimpleX native calls are detected;
- unauthorized incoming calls are rejected or ignored without disclosure;
- Hermes sends valid SimpleX call signaling commands;
- the WebRTC sidecar starts and reaches a concrete connected or failed state;
- controlled audio exchange or loopback is verified;
- `/call status` reflects native call state;
- `/call end` terminates the native call cleanly;
- failures are loud and logged with reason codes;
- logs redact secrets and media payloads;
- targeted tests pass.

Full voice milestone is accepted later when:

- local STT transcribes incoming call audio;
- Hermes runs the normal agent turn;
- local TTS is streamed back into the call;
- interruptions stop in-flight speech;
- no cloud STT/TTS provider is used unless explicitly configured;
- manual mobile SimpleX call supports at least one spoken round trip.

## References

- SimpleX audio/video calls: https://simplex.chat/docs/guide/audio-video-calls.html
- SimpleX chat protocol: https://simplex.chat/docs/protocol/simplex-chat.html
- SimpleX call commands: https://github.com/simplex-chat/simplex-chat/blob/stable/src/Simplex/Chat/Library/Commands.hs
- SimpleX call types: https://github.com/simplex-chat/simplex-chat/blob/stable/src/Simplex/Chat/Call.hs
- SimpleX WebRTC package: https://github.com/simplex-chat/simplex-chat/blob/stable/packages/simplex-chat-webrtc/src/call.ts
- LiveKit Agents overview: https://docs.livekit.io/agents/
- Existing Hermes call manager: `gateway/calls/manager.py`
- Existing Hermes SimpleX adapter: `plugins/platforms/simplex/adapter.py`
- Existing Hermes STT/TTS tools: `tools/transcription_tools.py`, `tools/tts_tool.py`
