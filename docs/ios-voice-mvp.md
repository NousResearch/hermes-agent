# iOS Voice MVP: make Jeeves feel like a phone call

Status: technical recommendation for an MVP spike  
Target user flow: open iPhone app -> press **Dial Jeeves** -> full-duplex-ish voice session starts -> press **Hang Up** -> session ends.

## Recommendation

Build the first iOS version as an **in-app, user-initiated audio session** using native iOS audio APIs plus a simple Hermes/Jeeves backend session over HTTPS/WebSocket. Do **not** start with CallKit, PushKit, Twilio, LiveKit, or custom WebRTC unless the first prototype proves that half-duplex turn-taking is not good enough.

The fastest viable path is:

1. SwiftUI button creates a `voice_session` on the Hermes backend.
2. iOS configures `AVAudioSession` as `.playAndRecord` with `.voiceChat` mode, requests microphone permission, and starts `AVAudioEngine`.
3. Client captures mic audio, applies voice processing, voice activity detection (VAD), and sends complete user turns to the backend.
4. Backend runs the normal Hermes/Jeeves agent loop, streams partial text/status events, synthesizes response audio, and returns audio chunks/URLs.
5. iOS plays response audio immediately, allows barge-in only after the MVP is stable, and exposes one obvious **Hang Up** action that stops capture/playback and closes the backend session.

This gives the user the phone-call affordance without taking on telephony infrastructure, App Store VoIP review risk, APNs/PushKit complexity, or WebRTC media server operations.

## Why this path

- Apple documents `AVAudioSession.Category.playAndRecord` as the category for simultaneous input and output, including VoIP-style apps; it continues with the silent switch and screen locked, and background continuation requires the `audio` background mode.
- Apple documents `AVAudioSession.Mode.voiceChat` for two-way voice communication and notes that `AVAudioEngine` voice processing or Audio Unit Voice I/O is needed for voice-specific processing such as echo cancellation and automatic gain correction.
- CallKit is for native system calling UI and coordination with the system Phone experience. It is valuable later, but it does not provide the media transport or agent session by itself.
- PushKit is specifically for incoming VoIP calls and, for modern iOS apps, requires CallKit handling. Jeeves' MVP is user-initiated outgoing sessions, so PushKit adds review and correctness burden without solving the core experience.
- LiveKit/Twilio/WebRTC are excellent when we need low-latency full duplex, room semantics, PSTN/SIP, or production VoIP behavior. They are overkill for validating "Jeeves in my ear" with one user talking to one agent.

## MVP architecture

```text
+----------------------+          HTTPS/WebSocket           +-------------------------+
| iOS Jeeves app       |  -------------------------------->  | Hermes voice gateway    |
| SwiftUI call screen  |                                    |                         |
| AVAudioSession       |  audio turn: PCM/Opus/AAC chunks   | STT / VAD finalization  |
| AVAudioEngine input  |  session/control events            | Hermes agent run        |
| AVAudioPlayerNode    |  <--------------------------------  | TTS/audio streaming     |
+----------------------+      assistant audio + events       +-------------------------+
```

### iOS client responsibilities

- UI: one screen with `Dial Jeeves`, connected/capturing/speaking states, transcript captions, and `Hang Up`.
- Audio session:
  - request `NSMicrophoneUsageDescription` permission;
  - set category `.playAndRecord`;
  - set mode `.voiceChat`;
  - prefer speaker by default for handheld mode, but support Bluetooth HFP/headphones;
  - activate only while the session is live.
- Capture:
  - use `AVAudioEngine` input tap at 16 kHz or 24 kHz mono PCM for easiest backend ingestion;
  - enable voice processing where available;
  - local VAD or simple energy threshold to detect end-of-turn;
  - upload a complete user turn first; stream chunks only if latency requires it.
- Playback:
  - start with server-returned audio files/chunks and `AVAudioPlayerNode`/`AVPlayer`;
  - duck or pause capture during playback for the first version to avoid echo loops;
  - add barge-in/cancellation after the basic loop is reliable.
- Lifecycle:
  - `Hang Up` closes WebSocket, stops engine, stops playback, deactivates audio session;
  - tolerate lock screen only after foreground MVP works;
  - no incoming-call UI in MVP.

### Backend contract

Start with a small, explicit voice gateway that wraps existing Hermes/Jeeves execution instead of inventing a phone stack.

#### Create session

`POST /v1/voice/sessions`

Request:

```json
{
  "persona": "jeeves",
  "client": "ios",
  "audio": { "input_format": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1 },
  "conversation_id": "optional-existing-thread-id"
}
```

Response:

```json
{
  "session_id": "voice_sess_...",
  "ws_url": "wss://.../v1/voice/sessions/voice_sess_.../events",
  "expires_at": "2026-..."
}
```

#### WebSocket events

Client -> server:

```json
{ "type": "audio.turn.start", "turn_id": "turn_1" }
{ "type": "audio.chunk", "turn_id": "turn_1", "encoding": "base64", "data": "..." }
{ "type": "audio.turn.end", "turn_id": "turn_1" }
{ "type": "session.cancel_response" }
{ "type": "session.hangup" }
```

Server -> client:

```json
{ "type": "session.ready" }
{ "type": "user.transcript.final", "turn_id": "turn_1", "text": "..." }
{ "type": "assistant.status", "state": "thinking" }
{ "type": "assistant.transcript.delta", "text": "..." }
{ "type": "assistant.audio.chunk", "encoding": "base64", "mime_type": "audio/mpeg", "data": "..." }
{ "type": "assistant.done" }
{ "type": "session.ended", "reason": "client_hangup" }
```

Implementation shortcut: if binary WebSocket frames are not already supported, use base64 chunks for the first spike and replace them with binary frames once the loop works.

## Phase plan

### Phase 0: desktop/local proof

- Add the voice gateway endpoints against a local Hermes process.
- Use a Python or web test client to send a recorded utterance and receive playable TTS.
- Measure agent + STT + TTS latency before touching iOS polish.

### Phase 1: iOS foreground call MVP

- SwiftUI app with Dial/Hang Up only.
- Foreground-only `.playAndRecord` + `.voiceChat` audio session.
- Turn-based capture: user speaks, silence finalizes, backend responds, app plays audio.
- Persist transcript/session id so failures are debuggable.

### Phase 2: make it feel call-like

- Lower turn latency with streaming STT/TTS.
- Add barge-in: user speech cancels current assistant playback and sends `session.cancel_response`.
- Add route controls and reliable Bluetooth behavior.
- Add lock-screen/background audio only after App Store rationale is clear and foreground UX is stable.

### Phase 3: decide on native calling infrastructure

Choose based on observed product need:

- **Keep AVAudioSession + WebSocket** if all calls are user-initiated from inside the app.
- **Add CallKit** if we need native call controls, system interruptions, recents-like affordances, or a more authentic Phone UI for outgoing calls.
- **Add PushKit + CallKit** only if Jeeves needs to ring the user for incoming calls. VoIP pushes must be reported through CallKit quickly and are not a general background wake-up mechanism.
- **Move to WebRTC/LiveKit** if true low-latency full duplex and robust network adaptation become the bottleneck.
- **Use Twilio** only if PSTN/SIP, real telephone numbers, or carrier-style calling features become requirements.

## Risks and caveats

- **Latency:** turn-based HTTP/WebSocket is simpler but less magical than full duplex. Mitigate with streaming TTS first; adopt WebRTC only if needed.
- **Echo and feedback:** speaker playback plus mic capture needs voice processing and/or capture pause during assistant speech. Start conservative.
- **Background behavior:** `playAndRecord` can continue with the screen locked when configured for background audio, but App Store review expects the background mode to match user-visible functionality. Avoid claiming background/VoIP behavior before the app actually needs it.
- **PushKit review risk:** do not use PushKit for reminders, agent nudges, or generic wakeups. It is for VoIP incoming calls and modern iOS requires CallKit handling.
- **CallKit scope:** CallKit provides native call UI and system coordination; it is not a transport and does not remove the need for audio capture, backend session control, STT, LLM, and TTS.
- **Backend concurrency:** voice sessions need cancellation, timeout, and cleanup paths because users will hang up mid-agent-run.

## What not to build yet

- No PSTN phone number.
- No Twilio SIP/Programmable Voice integration.
- No LiveKit room or SFU deployment.
- No custom WebRTC stack.
- No PushKit incoming-call path.
- No full native Phone app clone.
- No wake-word/background always-listening feature.

## Decision checkpoint

After Phase 1, continue on the simple path if median "end of user speech -> first assistant audio" is acceptable for a phone-like assistant. If not, first optimize streaming STT/TTS and response prefetching. Only then graduate to LiveKit/WebRTC.

## Sources consulted

- Apple Developer Documentation: `AVAudioSession.Category.playAndRecord` — simultaneous recording/playback, silent switch/lock-screen behavior, background audio requirement.
- Apple Developer Documentation: `AVAudioSession.Mode.voiceChat` — two-way VoIP-style voice mode, Bluetooth HFP, voice-processing caveats.
- Apple Developer Documentation: CallKit — native system calling UI and system coordination for VoIP services.
- Apple Developer Documentation: Responding to VoIP Notifications from PushKit — PushKit incoming VoIP flow and iOS 13+ CallKit requirement.
- LiveKit Swift quickstart — Swift SDK, voice AI starter app, microphone/background audio setup.
- OpenAI Realtime/WebRTC documentation — useful reference for future low-latency realtime media, but not required for the Hermes-native MVP.
- Twilio Voice iOS SDK documentation — useful if PSTN/SIP/call-provider features become requirements, not needed for an in-app Jeeves MVP.
