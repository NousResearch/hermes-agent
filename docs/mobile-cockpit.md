# Hermes Mobile Cockpit MVP

The Mobile Cockpit is a phone-first dashboard/PWA for using Hermes like an assistant in the truck: record a request, transcribe it locally on the Hermes machine, watch live work, approve risky actions, stop runs, and optionally use a hands-free turn-taking mode.

## What it uses

- Dashboard route: `/cockpit`
- Local/server-side STT: `POST /v1/cockpit/transcribe` using `faster-whisper`
- Run control:
  - `POST /v1/runs`
  - `GET /v1/runs/{run_id}/events`
  - `POST /v1/runs/{run_id}/approval`
  - `POST /v1/runs/{run_id}/stop`
- PWA files:
  - `/manifest.webmanifest`
  - `/sw.js`

## Start it locally

From the Hermes repo:

```bash
cd /home/dalton/.hermes/hermes-agent
npm --prefix web install
npm --prefix web run build
hermes dashboard --host 127.0.0.1 --port 9119 --tui --skip-build --no-open
```

Open on the machine:

```text
http://127.0.0.1:9119/cockpit
```

## Phone access

Android microphone access and PWA install both require **HTTPS** or localhost. For Dalton's phone, prefer Tailscale HTTPS/private access rather than a naked public tunnel.

Current WSL Tailscale DNS observed during setup:

```text
desktop-vcb4ksf-1.tail87092b.ts.net
```

Target phone URL shape:

```text
https://desktop-vcb4ksf-1.tail87092b.ts.net/cockpit
```

Target cockpit API base URL shape:

```text
https://desktop-vcb4ksf-1.tail87092b.ts.net/v1
```

If using separate ports instead of a reverse proxy, the dashboard and API server must both be exposed over HTTPS and API-server CORS must allow the dashboard origin. A same-origin Tailscale Serve reverse proxy is preferred.

## Install as an Android home-screen app

1. Open the HTTPS cockpit URL in Android Chrome.
2. Tap the cockpit **Install app** button if available, or Chrome menu → **Add to Home screen** / **Install app**.
3. Launch from the home-screen icon.
4. Allow microphone permission.

## Walkie-talkie mode

- Tap **Push to talk**.
- Speak.
- Tap **Stop & transcribe**.
- Hermes transcribes locally with Whisper.
- Tap **Send to Hermes** or edit the transcript first.

## Hands-free mode

Hands-free is turn-based, phone-call-like, but intentionally not always streaming forever yet:

- tap **Hands-free on**,
- cockpit records short turns,
- sends audio to local Whisper,
- auto-submits recognized commands,
- Hermes speaks back with browser TTS,
- cockpit resumes listening after done/blocked/approval.

Supported spoken controls:

- `stop` / `cancel` / `interrupt`
- `repeat`
- `clear`
- `approve`
- `deny`

Safety rule: hands-free mode never auto-approves risky actions. If Hermes requests approval, the cockpit shows an approval card and waits for explicit spoken or tapped approval.

## Browser limitations

- Android Chrome `SpeechRecognition` is no longer required; the cockpit records audio and uses server-side Whisper.
- Android still controls microphone permission and may pause recording if the browser/app is backgrounded or the screen locks.
- HTTPS is required for phone microphone + PWA install.
- Full natural full-duplex phone-call behavior should use LiveKit/Pipecat/native Android later.

## Next steps

- Configure a same-origin HTTPS Tailscale Serve route for dashboard + API.
- Test on Android Chrome from the home screen.
- If background/lock-screen behavior is not good enough, build a tiny native Android app or Mumble bridge.
