# Photon Setup QA Notes

Date: 2026-05-26

Scope: QA notes from testing the Photon iMessage integration on the
`hermes/hermes-1552fa93` branch.

## Confirmed Errors (Read First)

This section is the canonical list of confirmed QA failures. The longer
sections below are historical reference notes from earlier runs and should not
be treated as the current test plan.

### 1. Device login rejected Hermes' client id

Symptom:

```text
invalid_client / Invalid client ID
```

Cause:

- Hermes originally sent `client_id=hermes-agent`.
- Hosted Photon accepted the allowlisted `photon-cli` device client, not
  `hermes-agent`.

Fix/status:

- Current code defaults the Photon device login client to `photon-cli` with
  scope `openid profile email`.
- Long-term fix is still to register a real `hermes-agent` client on the
  Photon dashboard side and switch Hermes back to that id.

Hard justification:

- This is a server-side OAuth/device-flow allowlist failure. Local retry logic
  cannot make `hermes-agent` valid until Photon allows it.

### 2. `spectrum-ts@0.1.x` cannot start against current Photon/Spectrum

Symptoms:

```text
getaddrinfo ENOTFOUND spectrum-cloud.photon.codes
Spectrum Cloud authentication failed (404)
```

Cause:

- `package.json` previously allowed the sidecar to install the old
  `spectrum-ts@0.1.x` line.
- That SDK line defaulted to retired Spectrum hosts and, even after overriding
  the hostnames, still called an obsolete cloud authentication path.

Fix/status:

- Current sidecar dependency is pinned to `spectrum-ts@~1.7.2` and a lockfile
  is present.
- `1.7.2` starts against the current hosted service and passed
  `npm audit --omit=dev` during QA. `1.13.1` also connected, but its dependency
  graph had high-severity audit findings.

Hard justification:

- The crash happened during `Spectrum({ projectId, projectSecret, providers:
  [imessage.config()] })`, before the sidecar health endpoint became ready.
- Host overrides changed the failure from DNS to a 404 auth failure, proving
  the old SDK was not merely missing a hostname setting.

### 3. Old gateway command was invalid

Symptom:

```text
hermes gateway start --platform photon
```

does not work because `gateway start` does not accept `--platform`.

Fix/status:

- Use foreground QA mode while testing:

```bash
hermes gateway run -v
```

- For always-on local use:

```bash
hermes gateway install --force
hermes gateway start
```

Hard justification:

- `gateway run -v` gives live startup, sidecar, webhook, inbound, and send logs
  in one terminal. `gateway start` is the launchd/background-service path.

### 4. Existing/stale gateway process can block a fresh run

Symptom:

```text
Gateway already running (PID ...)
```

Cause:

- The gateway uses a single-process guard so two gateway instances do not fight
  over the webhook port, sidecar, and logs.

Fix/status:

```bash
hermes gateway stop
hermes gateway run -v
```

or:

```bash
hermes gateway run --replace -v
```

Hard justification:

- Only one gateway should own `PHOTON_WEBHOOK_PORT` and supervise the Photon
  sidecar. Replacing/stopping the old process is safer than running two
  listeners.

### 5. Local Hermes only receives webhooks after exposing and registering the local URL

Symptom:

- `hermes photon webhook list` can show registered hooks, but local Hermes gets
  no `inbound message` logs.
- Registered URLs such as production/stable domains do not deliver to the local
  `127.0.0.1:8788` listener.

Cause:

- Photon POSTs to the exact public URL registered under
  `https://spectrum.photon.codes/projects/{project_id}/webhooks/`.
- A local gateway listener is not reachable from Photon unless it is exposed
  with Cloudflare Tunnel, ngrok, or a real public hostname.

Fix/status:

```bash
cloudflared tunnel --url http://127.0.0.1:8788
hermes photon webhook register https://YOUR-TUNNEL.trycloudflare.com/photon/webhook
```

- After registering a new webhook, restart the gateway so it loads the new
  `PHOTON_WEBHOOK_SECRET`.

Hard justification:

- The local gateway only processed webhooks once the tunnel URL was registered
  and the secret matched the registered URL.

### 6. Gateway allowlist can block valid Photon deliveries

Symptom:

```text
No user allowlists configured
```

or inbound messages arrive from Photon but are denied before the agent runs.

Cause:

- The Photon adapter protects the bot by default. A sender must be listed in
  `PHOTON_ALLOWED_USERS`, or local dev must explicitly opt into
  `PHOTON_ALLOW_ALL_USERS=true`.

Fix/status:

```bash
PHOTON_ALLOWED_USERS=+15551234567
```

Hard justification:

- Photon transport success does not imply the sender is authorized to invoke
  Hermes. The gateway intentionally gates this before conversation handling.

### 7. Outbound send could not resolve webhook space ids

Symptom:

```text
photon-sidecar: handler error: Error: unable to resolve space id any;-;+134****3167
```

Cause:

- Webhooks identify an iMessage conversation with a canonical Spectrum space id
  like `any;-;+15551234567`.
- The current `spectrum-ts` iMessage helper resolves direct-message send spaces
  by recipient address, such as `+15551234567`, not by the full webhook id.

Fix/status:

- The sidecar now caches send-capable `Space` objects from the inbound stream.
- If the space is not cached and the id is a shared-line DM, the sidecar strips
  the `any;-;` prefix and resolves the send space with
  `imessage(app).space("+15551234567")`.

Hard justification:

- Passing `any;-;+...` into the iMessage helper produced the wrong lookup shape.
  Passing the E.164 recipient address resolved the send-capable DM space.
- Keeping the webhook `space.id` as the Hermes chat id preserves the gateway
  contract while adapting only the sidecar's SDK lookup.

### 8. Outbound send used the wrong `spectrum-ts` content shape

Symptom:

```text
TypeError: c.build is not a function
```

with stack frames under `spectrum-ts/dist/.../resolveContents`.

Cause:

- The sidecar called:

```js
space.send(text, { replyTo })
```

- Current `spectrum-ts` expects send contents/builders. The second positional
  argument was treated like another content object, but `{ replyTo }` has no
  `.build()` method.

Fix/status:

- The sidecar now imports the SDK `text(...)` builder and sends:

```js
space.send(text(messageText))
```

- Threaded replies are intentionally not wired yet. If Hermes passes `replyTo`,
  the sidecar logs that it is ignored and sends a plain text message.

Hard justification:

- Current Photon docs show plain sends as `space.send(text("Hello"))` or
  `space.send("Hello")`.
- Current reply docs use `message.reply(...)` or
  `space.send(reply(text(...), message))`, which requires the original Spectrum
  message object, not just Hermes' stored message id.

### 9. Existing Photon projects are not imported automatically

Symptom:

- `hermes photon login` can succeed, but local `~/.hermes/auth.json` may still
  have no `credential_pool.photon_project` entry.

Cause:

- `hermes photon setup` can create and store a new project, but does not yet
  offer a "select/import an existing dashboard project" path.

Fix/status:

- For QA, project credentials were recovered with the existing dashboard bearer
  token and stored through the same helper setup uses.
- Product follow-up: add project list/import support to `hermes photon setup`.

Hard justification:

- The Spectrum sidecar cannot start without `project_id` and `project_secret`,
  even when dashboard login has succeeded.

### 10. Local npm cache state can block sidecar install

Symptom:

- `npm install` fails before installing `spectrum-ts`, usually with cache or
  ownership errors under `~/.npm`.

Cause:

- This is local machine npm cache state, not a Photon API failure.

Fix/status:

- Use a clean npm cache for QA or fix ownership on `~/.npm`.
- Product follow-up: make `hermes photon install-sidecar` detect this case and
  print a targeted fix.

Hard justification:

- The gateway cannot start the sidecar until `plugins/platforms/photon/sidecar`
  has installed Node dependencies.

## Current Known-Good QA Path

Use this path for fresh testing:

```bash

```

Then send an iMessage to the assigned Photon number and watch for:

```text
inbound message: platform=photon ...
conversation turn: ... platform=photon ...
response ready: platform=photon ...
[Photon] Sending response ...
```

After the latest sidecar patch, restart the gateway before retesting outbound
delivery so Node reloads `sidecar/index.mjs`.

## Historical Reference

The sections below are older investigation notes and raw context. They remain
useful for understanding why the fixes were made, but the confirmed error list
above is the source of truth for current QA.