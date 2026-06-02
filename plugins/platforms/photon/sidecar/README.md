# Photon Sidecar

Do not run this process directly in normal use. It is a private child process
started and supervised by `plugins/platforms/photon/adapter.py`; it is not a
standalone service, CLI, webhook receiver, or local HTTP server.

This sidecar exists because Hermes is a Python application while Photon Spectrum
currently exposes the runtime messaging SDK through Node/TypeScript packages.
The Photon platform boundary inside Hermes remains the Python adapter. The
sidecar is only the small Node runtime bridge that lets that adapter call
`spectrum-ts` and subscribe to Spectrum inbound messages without moving Photon
gateway logic into JavaScript.

## How It Fits

```text
Hermes gateway
  <in-process Python platform adapter API>
plugins/platforms/photon/adapter.py
  <private stdin/stdout newline-delimited JSON>
plugins/platforms/photon/sidecar/index.mjs
  <Photon Spectrum SDK: spectrum-ts + iMessage provider>
Photon Spectrum
```

The gateway does not call this sidecar directly. The gateway loads the Photon
Python adapter like any other Hermes messaging platform adapter. The adapter
implements the Hermes-facing contract:

- starts and stops the private sidecar
- receives normalized SDK events and creates Hermes `MessageEvent` objects
- sends outbound messages and returns Hermes `SendResult` objects
- owns inbound dedupe, health/status, and current-home runtime state

The sidecar owns only the Node/Spectrum side:

- imports `spectrum-ts` and `spectrum-ts/providers/imessage`
- creates the Spectrum app with `PHOTON_PROJECT_ID` and
  `PHOTON_PROJECT_SECRET`
- reads inbound messages from `app.messages`
- normalizes low-level SDK message payloads before sending them to Python
- resolves Spectrum spaces and calls `space.send(...)` for outbound text
- emits SDK errors in the local sidecar protocol shape

Send-once mode (`node index.mjs --send-once` or
`PHOTON_SIDECAR_MODE=send-once`) initializes Spectrum, accepts one `send`
command on stdin, emits one structured response, and exits. It does not iterate
`app.messages`, so cron/proactive delivery does not open a second inbound
stream.

There are no Hermes-managed Photon webhooks, Cloudflare tunnels, public health
checks, or local HTTP endpoints in the primary Photon runtime path.

## Communication Protocols

### Gateway To Photon Adapter

Communication between the Hermes gateway and Photon is the normal in-process
Python platform adapter interface. The gateway calls adapter lifecycle and
send methods, and the adapter hands inbound `MessageEvent` objects back through
Hermes gateway machinery.

```text
outbound: Hermes reply -> gateway -> PhotonAdapter.send(...) -> SendResult
inbound:  PhotonAdapter.handle_message(MessageEvent) -> gateway session handling
```

### Photon Adapter To Sidecar

Communication between `adapter.py` and this sidecar is newline-delimited JSON
over the sidecar process stdin/stdout pipes. This is private local IPC, not HTTP.

`adapter.py` sends commands on stdin:

```json
{"requestId":"...","type":"send","spaceId":"...","text":"..."}
{"requestId":"...","type":"typing","spaceId":"..."}
{"requestId":"...","type":"shutdown"}
```

The sidecar emits messages on stdout:

```json
{"type":"ready","pid":12345,"startedAt":"...","protocolVersion":1}
{"type":"event","event":{"id":"...","space":{"id":"..."},"content":{"type":"text","text":"..."}}}
{"type":"response","requestId":"...","ok":true,"data":{"messageId":"..."}}
{"type":"response","requestId":"...","ok":false,"error":{"code":"...","message":"...","retryable":true}}
{"type":"stream_error","error":{"code":"...","message":"...","retryable":true}}
```

### Sidecar To Photon Spectrum

The sidecar talks to Photon Spectrum through the official Spectrum SDK packages:

- `spectrum-ts`
- `spectrum-ts/providers/imessage`

Inbound delivery comes from the Spectrum app message stream:

```js
for await (const [space, message] of app.messages) {
  // normalize and emit to adapter.py
}
```

Outbound delivery resolves a Spectrum space and sends SDK text content:

```js
const result = await space.send(spectrumText(messageText));
```

Management mode also uses authenticated Spectrum project-user API calls for
phone listing/add/remove operations, but that mode is separate from the normal
gateway runtime path.

## Install

```bash
cd plugins/platforms/photon/sidecar
npm install
```

The `hermes photon setup <phone>` command runs `npm install` here when sidecar
dependencies are missing.

## Runtime

The normal runtime owner is the Hermes gateway:

```text
hermes gateway
  -> loads PhotonAdapter
  -> PhotonAdapter launches this sidecar
  -> sidecar connects to Spectrum SDK
```

For debugging Photon runtime behavior, inspect the gateway log:

```bash
tail -f /Users/raysmacbookair/.hermes/logs/gateway.log
```
