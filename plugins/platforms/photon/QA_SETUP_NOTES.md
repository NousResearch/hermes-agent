# Photon Setup QA Notes

Date: 2026-05-26

Scope: QA notes from testing the Photon iMessage integration on the
`hermes/hermes-1552fa93` branch.

## Summary

The Photon setup flow was intended to let a user run:

```bash
hermes photon setup --phone +15551234567
```

and end up with local credentials in `~/.hermes/auth.json`, sidecar
dependencies installed, and a gateway that can connect to Photon. In this
test, that path did not work end to end without fixes. The final working
state required a device-login client compatibility fix, a sidecar SDK update,
explicit current Spectrum endpoints, corrected gateway docs, and local gateway
configuration.

The gateway is now considered healthy when logs show:

```text
[spectrum.lifecycle] Spectrum started
photon-sidecar: listening on 127.0.0.1:8789
[photon] connected
✓ photon connected
Gateway running with 1 platform(s)
```

The old broken states were:

```text
invalid_client / Invalid client ID
getaddrinfo ENOTFOUND spectrum-cloud.photon.codes
Spectrum Cloud authentication failed (404)
No user allowlists configured
```

## 1. Why `hermes photon setup --phone ...` Did Not Fully Create `auth.json`

The code is supposed to create and update `~/.hermes/auth.json`.

Relevant flow:

- `hermes photon login` runs Photon device login and stores a dashboard bearer
  token under `credential_pool.photon`.
- `hermes photon setup` reuses that token or runs login, creates a Photon
  project, then stores `project_id` and `project_secret` under
  `credential_pool.photon_project`.
- `--phone` is only used for Spectrum user creation. It does not supply or
  discover project credentials by itself.

What went wrong:

- Device login initially failed before anything useful could be persisted
  because hosted Photon rejected the Hermes client ID.
- After login worked, the local `auth.json` had a Photon token but did not have
  `credential_pool.photon_project`.
- The setup command can create a new project, but it does not currently have an
  "import my existing Photon project" path. If the project already exists in the
  dashboard, the CLI does not list/select/import it into local credentials.
- We did not need to hand-edit JSON manually. We used the existing stored bearer
  token to look up the existing project through the Photon dashboard API, then
  stored the project credentials with the same helper that setup uses.

Expected improvement:

- `hermes photon setup` should either create and store a project every time
  local project credentials are missing, or offer an import/select-existing
  project path.
- `hermes photon status` should make partial state clearer: token present but
  no project credentials means login succeeded, but Spectrum credentials are
  not configured.

## 2. Code Changes Needed To Make Photon Work

### Device Login Client ID

Problem:

- Hermes sent `client_id=hermes-agent`.
- Photon dashboard rejected it with `invalid_client`.
- The Photon dashboard server allowlist accepted `photon-cli`, not
  `hermes-agent`.

Change:

- Set the default Photon device client to `photon-cli`.
- Send scope `openid profile email`.
- Updated auth tests to assert the current request body.

Files:

- `plugins/platforms/photon/auth.py`
- `tests/plugins/platforms/photon/test_auth.py`

Long-term fix:

- Register/allowlist a real `hermes-agent` client ID on the Photon dashboard
  side, then switch Hermes back to its own client ID.

### Sidecar SDK Version

Problem:

- The sidecar installed `spectrum-ts@0.1.2` from `^0.1.0`.
- That old SDK defaulted to retired/unresolving hostnames:
  `spectrum-cloud.photon.codes` and `spectrum-imessage.photon.codes`.
  Those endpoints no longer exist in the current hosted Photon/Spectrum
  environment. The failure was not that Hermes could not send a message yet;
  the SDK could not finish startup because DNS lookup for the old cloud host
  failed with `getaddrinfo ENOTFOUND spectrum-cloud.photon.codes`.
- After overriding DNS hostnames, the old SDK still called an obsolete token
  path and failed with `Spectrum Cloud authentication failed (404)`.

What was run:

- The failure came from starting the Photon gateway/sidecar, not from a
  particular outbound message. Running the gateway:

```bash
hermes gateway run -v
```

  caused `PhotonAdapter` to spawn:

```bash
node plugins/platforms/photon/sidecar/index.mjs
```

  with `PHOTON_PROJECT_ID`, `PHOTON_PROJECT_SECRET`, `PHOTON_SIDECAR_PORT`,
  and `PHOTON_SIDECAR_TOKEN` in the child environment. `index.mjs` then called
  `Spectrum({ projectId, projectSecret, providers: [imessage.config()] })`.
  With `spectrum-ts@0.1.2`, that SDK initialization path attempted to contact
  `spectrum-cloud.photon.codes` and crashed before the sidecar health endpoint
  was ready.

- To isolate whether the problem was only DNS/default-host drift, the current
  endpoints were also supplied via environment overrides:

```bash
SPECTRUM_CLOUD_URL=https://spectrum.photon.codes
SPECTRUM_IMESSAGE_ADDRESS=imessage.spectrum.photon.codes:443
```

  Then the gateway/sidecar was started again. This got past the dead hostname
  but still failed during the same SDK startup/authentication step with
  `Spectrum Cloud authentication failed (404)`, which showed the old SDK was
  also using an obsolete Spectrum cloud token/auth path.

Change:

- Updated the sidecar dependency off the broken `0.1.x` line. The current
  repo pins `spectrum-ts@~1.7.2`: that version starts successfully against the
  current Spectrum service and `npm audit --omit=dev` reports zero
  vulnerabilities. Earlier QA also proved `1.13.1` could connect, but its
  dependency graph carried high-severity audit findings.
- Installed the updated dependency and committed a lockfile so clean installs
  do not drift back to the obsolete SDK line.

Files:

- `plugins/platforms/photon/sidecar/package.json`
- `plugins/platforms/photon/sidecar/package-lock.json`

### Explicit Spectrum Runtime Endpoints

Problem:

- Existing local sidecar installs could keep using stale SDK defaults.
- The sidecar needed the current Spectrum cloud URL and iMessage gRPC endpoint.

Change:

- Added adapter logic to pass:
  - `SPECTRUM_CLOUD_URL=https://spectrum.photon.codes`
  - `SPECTRUM_IMESSAGE_ADDRESS=imessage.spectrum.photon.codes:443`
- Added tests for URL normalization.
- Documented both environment variables.

Files:

- `plugins/platforms/photon/adapter.py`
- `tests/plugins/platforms/photon/test_adapter_config.py`
- `plugins/platforms/photon/plugin.yaml`
- `plugins/platforms/photon/README.md`
- `plugins/platforms/photon/sidecar/README.md`
- `plugins/platforms/photon/sidecar/index.mjs`

### Gateway Command Docs

Problem:

- The Photon README/setup text said to run:

```bash
hermes gateway start --platform photon
```

- The installed CLI does not support `--platform` on `gateway start`.

Change:

- Updated setup guidance to use foreground QA mode:

```bash
hermes gateway run -v
```

- For always-on local testing, installed the macOS launchd service with:

```bash
hermes gateway install --force
hermes gateway start
```

Files:

- `plugins/platforms/photon/README.md`
- `plugins/platforms/photon/cli.py`

## 3. Configuration Changes Made During QA

Local files configured:

- `~/.hermes/auth.json`
  - Stores the Photon dashboard token.
  - Stores the Spectrum project ID and project secret under
    `credential_pool.photon_project`.

- `~/.hermes/.env`
  - Stores `PHOTON_WEBHOOK_SECRET`.
  - Stores `SPECTRUM_CLOUD_URL=https://spectrum.photon.codes`.
  - Stores `SPECTRUM_IMESSAGE_ADDRESS=imessage.spectrum.photon.codes:443`.
  - Stores `PHOTON_ALLOWED_USERS=<your E.164 phone number>`.

- `~/Library/LaunchAgents/ai.hermes.gateway.plist`
  - Installed so the gateway can run as a macOS user service.

Important behavior:

- `hermes` CLI chat and the Photon gateway are separate paths. Asking the
  normal CLI to "use photon-platform" may not work unless there is already a
  known gateway target. The e2e test should start from an inbound iMessage to
  Photon so the gateway has the real Spectrum space ID.
- `photon-platform` is the plugin package name. The runtime platform is
  `photon`.
- Gateway allowlists matter. Without `PHOTON_ALLOWED_USERS` or
  `PHOTON_ALLOW_ALL_USERS=true`, inbound users are denied even if Photon itself
  connects.

## What Worked

- Device login worked after using the Photon-accepted client ID.
- Existing Photon project credentials worked after being stored locally.
- Sidecar dependencies installed successfully using a clean npm cache.
- `spectrum-ts@~1.7.2` connected successfully with the current Spectrum
  endpoints and produced a clean production npm audit.
- The gateway connected successfully and reported one connected platform.
- Focused tests passed:

```text
tests/plugins/platforms/photon: 30 passed
```

## What Did Not Work

- `client_id=hermes-agent` was rejected by hosted Photon.
- `hermes photon setup` did not import an existing dashboard project into
  local `auth.json`.
- `spectrum-ts@0.1.2` could not connect to the current Photon/Spectrum service.
- The old docs referenced a stale `gateway start --platform photon` command.
- A stale running gateway process kept retrying the old broken host until it was
  restarted.
- The local npm cache had ownership/cache pollution problems, which blocked
  `npm install` until using a clean cache or fixing `~/.npm` ownership.

## Open Follow-Ups

- Add a real `hermes-agent` OAuth/device-flow client on the Photon dashboard
  side and stop using `photon-cli` as a compatibility client.
- Add `hermes photon setup` support for listing/importing an existing Photon
  project when local `credential_pool.photon_project` is missing.
- Consider making `hermes photon install-sidecar` use a project-local or
  temporary npm cache, or detect common npm cache permission failures and print a
  targeted fix.
- Add a clearer first-run check for gateway allowlists.
- Add an e2e smoke command or doc section for the correct Photon test path:
  start gateway, send inbound iMessage, verify reply, then inspect logs.
