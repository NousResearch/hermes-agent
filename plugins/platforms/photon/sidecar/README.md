# Photon sidecar

Small Node helper that bridges Hermes Agent to Spectrum's SDK
(`spectrum-ts`). Hermes is Python; Spectrum is TypeScript-first, so both
inbound and outbound iMessage traffic go through this sidecar.

The sidecar:

- runs Spectrum 12 with `providers: [...]`; cloud mode loads `imessage` from
  `spectrum-ts/providers/imessage`, while `PHOTON_LOCAL=true` loads
  `localIMessage` from the explicit `@spectrum-ts/imessage-local` package
- relies on Spectrum 12's native ordered mixed text/attachment handling; no
  installed dependency files are rewritten at startup
- exposes a loopback-only HTTP control channel for the Python adapter
  to push send/typing requests (auth via `X-Hermes-Sidecar-Token`)
- drains the inbound message stream so `spectrum-ts` keeps its
  reconnect/heartbeat machinery alive and Hermes can receive inbound messages
  over the adapter's authenticated `GET /inbound` NDJSON stream

## Install

```bash
cd plugins/platforms/photon/sidecar
npm install
```

The Hermes plugin's `hermes photon setup` command runs `npm install`
here automatically.

## Run standalone

For debugging:

```bash
PHOTON_PROJECT_ID=... PHOTON_PROJECT_SECRET=... \
PHOTON_SIDECAR_PORT=8789 PHOTON_SIDECAR_TOKEN=$(openssl rand -hex 16) \
node index.mjs
```

Local macOS iMessage debugging:

```bash
PHOTON_LOCAL=true \
PHOTON_SIDECAR_PORT=8789 PHOTON_SIDECAR_TOKEN=$(openssl rand -hex 16) \
node index.mjs
```

In normal use, the Python adapter supervises this process — start,
restart on crash, kill on shutdown — and never asks the user to run
it by hand.

## Why a sidecar at all?

Photon's Spectrum send path is exposed through the TypeScript SDK's
`Space.send(...)` API. Hermes is Python, so replies go through this sidecar
until Photon ships a public HTTP send endpoint.

When Photon ships an HTTP send endpoint, the plan is to retire this
sidecar entirely and call it directly from Python.  The plugin's
outbound code path is already isolated behind small helpers
(`_sidecar_send`, `_sidecar_send_richlink`, and `_sidecar_send_attachment` in
`adapter.py`) to make that swap localized.
