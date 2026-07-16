# Photon sidecar

Small Node helper that bridges Hermes Agent to Spectrum's SDK
(`spectrum-ts`). Hermes is Python; Spectrum is TypeScript-first, so both
inbound and outbound iMessage traffic go through this sidecar.

The sidecar:

- runs `Spectrum({ projectId, projectSecret, providers: [imessage.config()] })`
  for Photon Cloud, or `Spectrum({ providers: [imessage.config({ local: true })] })`
  when `PHOTON_LOCAL=true`
- exposes a loopback-only HTTP control channel for the Python adapter
  to push send/typing requests (auth via `X-Hermes-Sidecar-Token`)
- streams inbound SDK messages to the Python adapter over loopback NDJSON

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

Photon publishes webhooks (inbound) but their docs state explicitly:

> Pass `space.id` to `Space.send(...)` from a separate `spectrum-ts`
> SDK instance to reply.  No public HTTP send endpoint exists today.

— https://photon.codes/docs/webhooks/events

When Photon ships an HTTP send endpoint, the plan is to retire this
sidecar entirely and call it directly from Python.  The plugin's
outbound code path is already isolated behind a single helper
(`_sidecar_send` in `adapter.py`) to make that swap a one-file change.
