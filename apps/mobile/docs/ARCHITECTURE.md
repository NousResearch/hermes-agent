# Hermes iPhone App Architecture

Hermes iPhone is the phone-native Hermes surface: chat, capture, approvals, agent activity, and self-shaping UI.

## Current MVP

- Expo + React Native + TypeScript.
- Hermes remains the agent runtime on the server/desktop backend.
- Mobile connects to Hermes dashboard/TUI gateway over JSON-RPC WebSocket.
- Mobile records explicit capture sessions with `expo-audio`, prefers external USB/DJI input when available, and stores audio in the document directory.
- Self-shaping starts as a safe declarative `AppExperienceSpec`, not arbitrary generated code.
- Connection settings are stored locally with AsyncStorage; the app supports token-mode and OAuth-ticket-mode URL construction.
- Chat streaming currently handles core message/tool/approval event types and renders the latest mobile transcript as cards.
- Capture upload currently has a typed multipart helper; the durable retry queue and server endpoint are next.

## Next implementation slice

1. Add a Hermes server endpoint for `POST /api/mobile/capture-segment` that accepts multipart audio + metadata.
2. Persist `CaptureUploadSegment` rows locally with AsyncStorage or SQLite.
3. Enqueue completed recordings from `useCaptureRecorder`.
4. Retry uploads on app foreground/network recovery.
5. Convert `approval.request` and `clarify.request` events into actionable mobile cards with approve/deny/respond buttons.

## What was borrowed from Hermes Desktop

Hermes Desktop is MIT and lives in `NousResearch/hermes-agent/apps/desktop`. Concepts ported here:

- streaming gateway event names (`message.delta`, `tool.start`, `approval.request`, etc.)
- JSON-RPC WebSocket client shape from `apps/shared`
- session/create/resume/submit mental model
- tool activity as first-class UI, but rendered as mobile cards rather than desktop panes

## Self-shaping tiers

1. Runtime spec changes: theme/layout/copy/card ordering. Fast, reversible, safe.
2. Expo OTA changes: new JS screens/components after tests/review.
3. Native/app-store changes: new permissions, background behavior, auth/security, native modules.

## Tomorrow hardware test

Before trusting it for a real investor meeting, validate on iPhone with the DJI receiver:

1. Plug receiver into USB-C.
2. Start Hermes recording.
3. Confirm input label is external/USB or verify by tapping the transmitter while covering phone mic.
4. Lock screen for 20 minutes.
5. Stop, play back, verify duration and audio source.
6. Use Voice Memos / Just Press Record as fallback until Hermes passes this test.
