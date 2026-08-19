# Photon sidecar

Small Node helper that bridges Hermes Agent to Photon's Spectrum SDK
(`spectrum-ts`).  Hermes is Python; Photon has no public HTTP
send-message endpoint today; replies therefore go through this sidecar.

The sidecar:

- runs `Spectrum({ projectId, projectSecret, providers: [imessage.config()] })`
- exposes a loopback-only HTTP control channel for the Python adapter
  to push send/typing requests (auth via `X-Hermes-Sidecar-Token`)
- drains the inbound message stream so `spectrum-ts` keeps its
  reconnect/heartbeat machinery alive and Hermes can receive inbound messages
  over the adapter's loopback `GET /inbound` stream

## Install

```bash
cd plugins/platforms/photon/sidecar
npm install
```

The Hermes plugin's `hermes photon setup` command runs `npm install`
here automatically.

## Inbound attachment handles

Inbound attachment bytes never enter NDJSON and are never cached to plaintext
disk by the Photon adapter. The normalized event contains the existing
metadata plus a random 48-character opaque `handle`. The sidecar holds a copy
of the bytes in memory with fixed limits: 20 MiB per item, 64 MiB total, 64
items, and a five-minute TTL. Capacity and read failures emit metadata without
a handle and never include provider error text.

An authenticated `GET /attachment/<handle>` returns the raw bytes once with
`Cache-Control: no-store` and `X-Content-Type-Options: nosniff`. The exact path
accepts no query string or suffix. Consumption is atomic: concurrent or replay
requests receive the same content-free 404 as expired and unknown handles.
Buffers are zeroed after response completion, on expiry, and on shutdown.

The Python adapter preserves the handle in `MessageEvent.raw_message` but does
not fetch it or create a media cache file. A downstream secure consumer must
authorize the sender and redeem the handle within the TTL before the attachment
becomes model-ready.

Handle-bearing events also carry a random `deliveryId`. The sidecar retains the
event until the adapter posts `/inbound-ack` after sender authorization, secure
handle acceptance, and successful message processing. A dropped connection
replays the same delivery without redeeming the handle twice. The queue is
bounded to 128 events with a five-minute TTL and wipes attachment buffers when
an event expires or the process shuts down.

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

Photon's Spectrum send path is exposed through the TypeScript SDK's
`Space.send(...)` API. Hermes is Python, so replies go through this sidecar
until Photon ships a public HTTP send endpoint.

When Photon ships an HTTP send endpoint, the plan is to retire this
sidecar entirely and call it directly from Python.  The plugin's
outbound code path is already isolated behind small helpers
(`_sidecar_send`, `_sidecar_send_richlink`, and `_sidecar_send_attachment` in
`adapter.py`) to make that swap localized.
