# Photon sidecar

Small Node helper that bridges Hermes Agent to Photon's Spectrum SDK
(`spectrum-ts`).  Hermes is Python; Photon has no public HTTP
send-message endpoint today; replies therefore go through this sidecar.

The sidecar:

- runs `Spectrum({ projectId, projectSecret, providers: [imessage.config()] })`
- exposes a loopback-only HTTP control channel for the Python adapter
  to push send/typing requests (auth via `X-Hermes-Sidecar-Token`)
- drains the inbound message stream so `spectrum-ts` keeps its
  reconnect/heartbeat machinery alive (real inbound delivery is via
  Photon's signed webhook hitting our Python aiohttp server)

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

In normal use, the Python adapter supervises this process — start,
restart on crash, kill on shutdown — and never asks the user to run
it by hand.

## Why a sidecar at all?

Photon publishes webhooks (inbound) but their docs state explicitly:

> Pass `space.id` to `Space.send(...)` from a separate `spectrum-ts`
> SDK instance to reply.  No public HTTP send endpoint exists today.

— https://photon.codes/docs/webhooks/events

When Photon ships an HTTP send endpoint, the plan is to retire this
sidecar entirely and call it directly from Python.  The plugin's
outbound code path is already isolated behind a single helper
(`_sidecar_send` in `adapter.py`) to make that swap a one-file change.
