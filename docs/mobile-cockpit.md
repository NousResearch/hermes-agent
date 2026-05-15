# Hermes Mobile Cockpit MVP

The Mobile Cockpit is a phone-first dashboard page for using Hermes like an assistant in the truck: dictate a request, watch live work, approve risky actions, stop runs, and optionally use a minimal-touch hands-free mode.

## Start it

From the Hermes repo:

```bash
cd /home/dalton/.hermes/hermes-agent
npm --prefix web install
npm --prefix web run build
hermes dashboard --host 127.0.0.1 --port 9119 --tui
```

Then open:

```text
http://127.0.0.1:9119/cockpit
```

If the Hermes API server is not already running, start the gateway with the API server platform enabled. The cockpit uses these existing endpoints:

- `POST /v1/runs`
- `GET /v1/runs/{run_id}/events`
- `POST /v1/runs/{run_id}/approval`
- `POST /v1/runs/{run_id}/stop`

## Phone access

For real phone use, prefer a private network path, not a public unauthenticated tunnel:

- Tailscale/WireGuard to the WSL/host machine, or
- a private Cloudflare Access route, or
- same trusted LAN while parked/in the office.

In the cockpit settings panel, set **API base URL** to the Hermes API server URL reachable from the phone, for example:

```text
http://<tailscale-or-lan-host>:8642
```

If `api_server.api_key` is configured, paste it into the optional API key field.

## Hands-free mode

Hands-free mode is a browser-based MVP:

- speaks status changes and Hermes responses using `speechSynthesis`,
- attempts continuous speech recognition where the browser allows it,
- submits dictated commands after a short silence,
- supports voice commands: `approve`, `deny`, `stop`, `repeat`, and `clear`.

Safety rule: hands-free mode never auto-approves risky actions. If Hermes requests approval, the cockpit shows an approval card and waits for explicit spoken or tapped approval.

## Browser limitations

- Speech recognition is browser-dependent. Chrome/Android has the best support.
- iOS Safari support for continuous recognition is limited and may require manual taps.
- Browser speech recognition requires HTTPS or localhost on many devices.
- Cross-origin API use requires the API server CORS settings to allow the cockpit origin.

## Next steps

- Add a small authenticated reverse proxy so `/cockpit` and `/v1/runs` are same-origin on mobile.
- Add a real wake-word / WebRTC voice pipeline with LiveKit or Pipecat.
- Add PWA manifest/home-screen install polish.
- Add per-contact/per-system trusted approval rules.
