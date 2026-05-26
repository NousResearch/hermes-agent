# Private SimpleX Call Flow Implementation Design

> **For Hermes:** Use test-driven-development for implementation. Use subagent-driven-development only after this design is converted into small reviewed implementation tasks.

**Goal:** Add private voice calls to Hermes with a reliable browser/WebRTC room as the first shipped command path, while preparing a SimpleX-native call bridge for incoming SimpleX app calls and explicit native SimpleX commands.

**Architecture:** Keep SimpleX as the private control channel for native calls, but do not assume the SimpleX daemon exposes live audio. `simplex-chat` exposes call signaling commands; Hermes must own the WebRTC media endpoint to actually hear and speak. For the first milestone, `/call` creates a private browser room link and sends it back in the requesting channel.

**Tech Stack:** Hermes gateway, platform adapters, SimpleX WebSocket daemon, browser WebRTC room service, local STT/TTS, local model inference, pytest. LiveKit is the likely browser-room backend, but the design should isolate it behind a small provider interface.

---

## 1. Product Decisions

### Accepted

- `/call` is reserved for the reliable browser/WebRTC fallback path.
- `/call` works across authorized private messaging channels, including Telegram and SimpleX.
- When `/call` is entered in Telegram, Hermes sends the browser room link back in Telegram.
- When `/call` is entered in SimpleX, Hermes sends the browser room link back in SimpleX.
- Calls are private-only for MVP. Group/channel invocations fail loudly and do not mint links.
- `/call browser` is an alias for `/call`.
- `/call native` is explicit SimpleX-native only.
- Incoming calls made from the SimpleX app to the Hermes SimpleX contact are the primary SimpleX-native experience.
- Native SimpleX calling is not treated as implemented until Hermes has a WebRTC media endpoint compatible with SimpleX call signaling.
- Silent failures are not acceptable. User-facing failures and redacted structured logs are required.
- OpenClaw realtime voice/session pieces are reference material for the media runtime, not a direct SimpleX bridge already present locally.

### Deferred

- Group call links.
- Multi-user rooms.
- Native SimpleX call initiation from non-SimpleX channels, except a later explicit configured mapping such as `/call me`.
- Public unauthenticated call links.
- Reusing dashboard auth for guest/mobile call access.
- Guaranteeing SimpleX-native call compatibility before a proof-of-concept WebRTC endpoint validates signaling, encryption, media capture, and audio playback.

---

## 2. Research Findings

### SimpleX Native Calls

SimpleX call signaling is scriptable, but live audio is not exposed by `simplex-chat` as raw PCM/RTP. The official source exposes hidden call commands:

- `/_call invite @<contactId> <CallType JSON>`
- `/_call reject @<contactId>`
- `/_call offer @<contactId> <WebRTCCallOffer JSON>`
- `/_call answer @<contactId> <WebRTCSession JSON>`
- `/_call extra @<contactId> <WebRTCExtraInfo JSON>`
- `/_call end @<contactId>`
- `/_call status @<contactId> connecting|connected|disconnected|failed`
- `/_call get`

Reference sources:

- SimpleX command parser: <https://github.com/simplex-chat/simplex-chat/blob/stable/src/Simplex/Chat/Library/Commands.hs>
- SimpleX call types: <https://github.com/simplex-chat/simplex-chat/blob/stable/src/Simplex/Chat/Call.hs>
- SimpleX WebRTC engine: <https://github.com/simplex-chat/simplex-chat/blob/stable/packages/simplex-chat-webrtc/src/call.ts>

Important native-call implications:

- SimpleX calls use WebRTC for the media plane.
- SimpleX signaling travels through chat messages over SMP.
- SimpleX adds an AES-GCM insertable-stream encryption layer on top of WebRTC SRTP when both peers support it.
- The desktop app runs WebRTC in a browser page served from a localhost NanoWSD server.
- Hermes can send and receive SimpleX call signaling, but it must implement or host the WebRTC media endpoint itself.

### Current Hermes State

Current Hermes SimpleX support handles text, files, and native SimpleX voice notes, not live calls.

Relevant files:

- `/Users/bryanmurphy/PROJECTS/Hermes-Agent/plugins/platforms/simplex/adapter.py`
- `/Users/bryanmurphy/PROJECTS/Hermes-Agent/tests/gateway/test_simplex_plugin.py`
- `/Users/bryanmurphy/PROJECTS/Hermes-Agent/gateway/run.py`
- `/Users/bryanmurphy/PROJECTS/Hermes-Agent/hermes_cli/commands.py`

### OpenClaw Material To Cherry-Pick

No local OpenClaw SimpleX-native bridge was found. Useful OpenClaw code is still valuable for the realtime voice runtime:

- `/Users/bryanmurphy/openclaw/src/talk/session-runtime.ts`
- `/Users/bryanmurphy/openclaw/src/talk/provider-types.ts`
- `/Users/bryanmurphy/openclaw/src/talk/audio-codec.ts`
- `/Users/bryanmurphy/openclaw/extensions/voice-call/src/media-stream.ts`
- `/Users/bryanmurphy/openclaw/extensions/voice-call/src/webhook/realtime-handler.ts`
- `/Users/bryanmurphy/openclaw/extensions/google-meet/src/realtime.ts`

Cherry-pick concepts:

- Call/session lifecycle events.
- Audio source/sink abstractions.
- Barge-in and cancellation behavior.
- Realtime transcript accumulation.
- Output audio lifecycle.
- Loud failure codes and stale call cleanup.
- Audio codec conversion and pacing patterns.

---

## 3. Target UX

### 3.1 `/call` in Telegram DM

User sends:

```text
/call
```

Hermes:

1. verifies the Telegram user is authorized;
2. verifies the message is in a private chat;
3. creates a private browser/WebRTC room;
4. creates a short-lived room token scoped to the call;
5. sends the room link back in Telegram;
6. starts the media bridge when the user joins.

Example response:

```text
Private call room ready:
https://<private-host>/call/<opaque-token>

This link expires in 10 minutes.
```

### 3.2 `/call` in SimpleX DM

User sends:

```text
/call
```

Hermes:

1. verifies the SimpleX contact is authorized by the existing SimpleX allowlist;
2. verifies the chat is direct/private;
3. creates the same browser/WebRTC room;
4. sends the short-lived link back in SimpleX.

This command does not initiate a SimpleX-native call. Native SimpleX is explicit or incoming-only.

### 3.3 `/call` in a Group or Channel

Hermes does not mint a link. It replies loudly:

```text
Calls are private-only. DM me /call to create a private room.
```

The gateway logs a redacted reason code such as `call_private_chat_required`.

### 3.4 `/call browser`

Alias for `/call`. It exists for clarity and future-proofing.

### 3.5 `/call native`

SimpleX-only command. It must fail loudly unless all native-call prerequisites are met:

- requester is an authorized SimpleX direct contact;
- SimpleX daemon is connected;
- SimpleX call feature is available for the contact;
- Hermes native WebRTC endpoint is enabled;
- media bridge is healthy.

If unavailable:

```text
SimpleX-native calls are not available: native WebRTC bridge is not enabled.
```

No browser fallback is used for `/call native`.

### 3.6 Incoming SimpleX App Call

The user calls Hermes from the SimpleX app.

Hermes target behavior:

1. receive `CallInvitation` from the SimpleX daemon;
2. verify the contact is authorized and direct/private;
3. accept only if native WebRTC bridge is enabled and healthy;
4. establish SimpleX-compatible WebRTC signaling;
5. bridge remote audio into STT and local inference;
6. stream TTS audio back over the WebRTC connection;
7. end loudly and cleanly on timeout, disconnect, or media failure.

Until the native bridge exists, Hermes should reject or ignore native call invitations with an explicit log reason, not silently fail.

---

## 4. Command Contract

Add a gateway-visible command family:

- `/call`
- `/call browser`
- `/call native`
- `/call status`
- `/call end`

Optional later command:

- `/call me`

`/call me` should not be included in MVP unless there is a configured cross-platform identity map from Telegram user to SimpleX contact.

Command behavior matrix:

| Command | Telegram DM | SimpleX DM | Group/channel |
|---|---|---|---|
| `/call` | Browser room link in Telegram | Browser room link in SimpleX | Loud private-only failure |
| `/call browser` | Browser room link in Telegram | Browser room link in SimpleX | Loud private-only failure |
| `/call native` | Loud unsupported unless mapped later | SimpleX-native only | Loud private-only failure |
| `/call status` | Current call state | Current call state | Loud private-only failure |
| `/call end` | End current call | End current call | Loud private-only failure |

---

## 5. Core Architecture

### 5.1 Gateway Call Manager

Add a small gateway call manager rather than embedding call state directly in `gateway/run.py`.

Suggested module:

```text
gateway/calls/
  __init__.py
  manager.py
  models.py
  tokens.py
  browser_room.py
  native.py
```

Core objects:

```python
CallSession:
    call_id: str
    platform: str
    chat_id: str
    user_id: str
    mode: Literal["browser", "simplex_native"]
    state: CallState
    room_id: str | None
    token_hash: str | None
    created_at: datetime
    expires_at: datetime
    ended_at: datetime | None
    last_error_code: str | None
```

`CallManager` responsibilities:

- auth gate integration;
- private-chat enforcement;
- one active call per `(platform, chat_id, user_id)` for MVP;
- room creation;
- token minting and hashing;
- call state transitions;
- TTL cleanup;
- redacted structured logging;
- user-facing error mapping.

### 5.2 Browser Room Provider

Use a provider interface so LiveKit can be swapped if needed:

```python
class BrowserRoomProvider(Protocol):
    async def create_room(self, session: CallSession) -> BrowserRoom: ...
    async def end_room(self, room_id: str) -> None: ...
```

Recommended provider: self-hosted LiveKit, privately exposed over Tailscale or another HTTPS endpoint reachable by mobile.

Do not reuse dashboard auth. Call rooms need short-lived guest/mobile credentials scoped only to one room.

### 5.3 Media Bridge

The browser room media bridge connects:

```text
mobile browser audio -> WebRTC room -> Hermes media bridge -> STT -> local model -> streaming TTS -> WebRTC room -> mobile browser
```

The bridge should expose:

- inbound PCM frames;
- outbound PCM/Opus frames;
- VAD or turn boundary events;
- interruption/barge-in signals;
- transcript events;
- final response events;
- loud media failure events.

Initial local runtime:

- STT: configured local STT provider, with faster-whisper or whisper.cpp as practical candidates.
- TTS: configured Hermes TTS provider.
- LLM: current Hermes local/provider routing.

### 5.4 SimpleX Native Bridge

Native SimpleX calling is a separate adapter capability:

```python
class NativeCallAdapter(Protocol):
    async def supports_native_call(self, event) -> CapabilityResult: ...
    async def accept_native_call(self, invitation) -> NativeCallResult: ...
    async def start_native_call(self, event, call_id: str) -> NativeCallResult: ...
    async def end_native_call(self, call_id: str) -> None: ...
```

The SimpleX implementation must:

- observe incoming call invitation events;
- send `/_call offer`, `/_call answer`, `/_call extra`, and `/_call end` through the daemon;
- host or embed a WebRTC endpoint compatible with SimpleX compressed SDP/ICE payloads;
- implement SimpleX AES-GCM insertable-stream compatibility if the contact negotiated encryption;
- bridge decrypted audio to the same Hermes media runtime used by browser rooms.

---

## 6. Privacy And Security

MVP rules:

- Calls are private-only.
- Browser links are sent back only in the channel where `/call` was authorized.
- Browser room links must be short-lived.
- Tokens bind to `platform`, `chat_id`, `user_id`, `call_id`, nonce, and expiry.
- Store only token hashes server-side.
- Room IDs are random and per-call.
- LiveKit grants, if used, are scoped to exactly one room.
- Never log raw room URLs, JWTs, access tokens, or media keys.
- Redacted logs include `call_id`, `platform`, `state`, and reason code.
- Any cross-platform routing such as Telegram `/call me` to SimpleX requires explicit identity mapping config before it is enabled.

For browser media encryption:

- Use HTTPS/WSS at minimum.
- If LiveKit E2EE is enabled, keep the media key out of server logs and URLs seen by the HTTP server.
- Hermes must access decrypted audio to transcribe/respond, so E2EE protects transport/SFU paths, not the local Hermes process.

---

## 7. Loud Failure Model

Every failed call attempt returns a user-visible message and logs a structured reason.

Required reason codes:

- `call_private_chat_required`
- `call_auth_denied`
- `call_already_active`
- `call_browser_provider_missing`
- `call_public_url_missing`
- `call_room_create_failed`
- `call_token_expired`
- `call_join_auth_failed`
- `call_media_bridge_failed`
- `call_simplex_ws_disconnected`
- `call_simplex_native_unavailable`
- `call_simplex_native_timeout`
- `call_simplex_native_encryption_unsupported`
- `call_simplex_native_media_failed`

Examples:

```text
Calls are private-only. DM me /call to create a private room.
```

```text
Call setup failed: browser call provider is not configured.
```

```text
SimpleX-native calls are unavailable: native WebRTC bridge is not enabled.
```

---

## 8. Implementation Phases

### Phase 1: Browser `/call` MVP

Deliver:

- command registry entry;
- gateway dispatch handler;
- call manager;
- private chat enforcement;
- browser room provider interface;
- token minting/verification;
- Telegram and SimpleX outbound link delivery;
- `/call status`;
- `/call end`;
- loud logging and tests.

This phase does not require SimpleX-native call media.

### Phase 2: Browser Media Bridge

Deliver:

- bridge room audio into Hermes STT;
- stream Hermes responses into TTS;
- send audio back into the room;
- support turn detection and interruption;
- log media pipeline failures loudly.

Cherry-pick OpenClaw concepts here, especially session lifecycle and audio source/sink abstractions.

### Phase 3: SimpleX Native Signaling

Deliver:

- parse/handle incoming SimpleX call invitation events;
- expose `supports_native_call`;
- send explicit SimpleX call commands through the daemon;
- wire state transitions to the call manager;
- fail loudly when media bridge is disabled.

This phase can validate signaling without claiming full audio support.

### Phase 4: SimpleX Native Media Endpoint

Deliver:

- WebRTC endpoint compatible with SimpleX signaling;
- compressed SDP/ICE handling;
- SimpleX AES-GCM media transform compatibility;
- inbound/outbound audio bridge into the shared media runtime;
- manual test with the SimpleX mobile app calling Hermes.

---

## 9. Test Plan

### Unit Tests

- command registry includes `/call`, `/call browser`, `/call native`, `/call status`, `/call end`;
- `/call` resolves to browser mode;
- `/call browser` resolves to browser mode;
- `/call native` rejects unsupported platforms;
- private-chat enforcement rejects group/channel requests;
- token creation binds platform/chat/user/call/expiry;
- token verification rejects expired, replayed, or mismatched tokens;
- call manager enforces one active call per route;
- call manager cleans expired sessions;
- logs redact room URLs and tokens.

### Gateway Tests

- Telegram DM `/call` sends a browser room link in Telegram.
- SimpleX DM `/call` sends a browser room link in SimpleX.
- Telegram group `/call` fails with private-only message.
- SimpleX group `/call` fails with private-only message.
- `/call status` reports idle, waiting, active, expired, and ended states.
- `/call end` ends a current browser room.
- provider misconfiguration returns a loud user-facing error.

### SimpleX Native Tests

- incoming call invitation from allowed contact is detected;
- incoming call invitation from unauthorized contact is rejected/logged;
- native path returns `call_simplex_native_unavailable` until media endpoint is enabled;
- `/_call` command payloads are serialized exactly as SimpleX expects;
- native status changes map into call manager states.

### Manual Verification

- Send `/call` in Telegram DM and join from mobile browser.
- Send `/call` in SimpleX DM and join from mobile browser.
- Verify group/channel `/call` does not create a room.
- Verify room link expires.
- Verify logs contain reason codes and no raw JWT/link secrets.
- Verify one active call per user route.
- After native bridge exists: call Hermes from SimpleX mobile app and complete a short spoken turn.

---

## 10. Open Questions For Implementation Planning

- Which private HTTPS endpoint should serve the browser room: existing dashboard host, a dedicated call subservice, or a LiveKit-hosted path behind Tailscale?
- Which local STT/TTS stack should be enabled first on this Mac for low-latency calls?
- Should Phase 1 use a stub room provider for tests before LiveKit is fully configured?
- Is EC2 SSH inspection still needed to recover a missing SimpleX-native prototype, or should Hermes rebuild from current SimpleX source plus OpenClaw runtime concepts?

