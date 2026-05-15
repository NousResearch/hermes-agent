# Hermes Mobile Cockpit MVP Implementation Plan

> **For Hermes:** Implement directly in the existing dashboard web app; use existing `/v1/runs` API-server endpoints rather than creating a new agent loop.

**Goal:** Add a mobile-first cockpit page for Dalton to task Hermes by voice/text, watch live progress, approve/deny actions, stop runs, and use a hands-free mode while driving.

**Architecture:** Frontend-only MVP in the React dashboard. The cockpit calls the Hermes API server (`/v1/runs`, `/v1/runs/{id}/events`, approval, stop) with a configurable API base URL and optional bearer key. Browser SpeechRecognition and speechSynthesis provide MVP voice support with typed fallbacks.

**Tech Stack:** React 19, Vite, TypeScript, existing Hermes dashboard components/styles, browser Web Speech API, Server-Sent Events.

---

## Tasks

1. Add `CockpitPage.tsx` with mobile-first UI, configurable API base/key, push-to-talk, hands-free toggle, speech synthesis, run creation, SSE event stream, approval controls, stop button, and status cards.
2. Register `/cockpit` route and sidebar nav item in `App.tsx`.
3. Add setup/usage documentation under `docs/mobile-cockpit.md`.
4. Verify with `npm run build`; capture browser/API-server limitations.

## Acceptance criteria

- `/cockpit` loads in the dashboard.
- User can type or dictate a command and start a run.
- Live SSE events append to the activity stream and update status.
- Completed output appears and can be spoken aloud.
- Approval request events show approve/deny/session/always controls.
- Stop button calls the run stop endpoint.
- Hands-free mode supports spoken commands: yes/approve/send it, no/cancel/deny, stop, repeat, clear.
- MVP is safe by default: no auto-approval; spoken approval requires explicit confirmation.
