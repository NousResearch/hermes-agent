# Dashboard passkeys (WebAuthn) — setup guide

Passkeys give the Hermes dashboard a **second factor on the password login**:
you sign in with your username and password, then complete a WebAuthn
assertion (Touch ID, Windows Hello, a security key, or your password manager).
The passkey never replaces the password — it hardens it, and the password step
works unchanged for clients that can't do WebAuthn.

This guide takes you from a fresh clone to a working passkey login, including
the reverse-proxy deployment that is the usual gotcha.

## What you need

- A Linux/macOS box where you can run the dashboard, reachable from your
  reverse proxy.
- A proxy already serving **https://your.domain** and forwarding to the
  dashboard (any of Traefik / nginx / Caddy). It must pass `X-Forwarded-Proto`
  (they all do by default) and WebSockets (nginx: the `Upgrade`/`Connection`
  headers).
- A browser with a platform authenticator (Touch ID / Windows Hello) or a
  hardware key, on a device you trust.

## 1. Install

```sh
git clone -b feat/dashboard-webauthn https://github.com/eps-epsiloneridani/hermes-agent.git
cd hermes-agent
HERMES_REPO_URL=https://github.com/eps-epsiloneridani/hermes-agent.git \
  bash install.sh --branch feat/dashboard-webauthn
```

Wait for `✓ Install complete!`. The installer brings its own Python and tools;
the `cua-driver` step may pause for a minute while it installs X11 support
libraries — that is automatic, not a hang. If the **web UI build** fails with
a rolldown/npm "Cannot find native binding" error, that is the known
npm optional-dependencies bug (npm/cli#4828); remove
`node_modules/` and `package-lock.json` and re-run the build (the next
`hermes` launch retries it).

## 2. Start the dashboard beyond loopback

The dashboard listens on `127.0.0.1:9119` by default, which your proxy cannot
reach:

```sh
hermes dashboard --host 0.0.0.0
```

On first start, with nothing configured, it asks how to authenticate — choose
**[1] Username & password**. This password is the *first* factor the passkey
will protect. (If the prompt doesn't appear because another provider such as
Portal OAuth is already configured, don't worry — the setup wizard in step 4
offers to create it.)

## 3. Point your proxy

Forward `https://your.domain` → `http://<machine-ip>:9119`.

## 4. Run the setup wizard

```sh
hermes dashboard webauthn setup
```

It checks its own prerequisites first — config readability and the `fido2`
library (offering to install the `webauthn` extra for you) — before asking
anything. Then:

- **RP ID** — your registrable domain, e.g. `your.domain` (no scheme, no
  port). Passkeys bind to it **forever**; it is frozen once the first key is
  enrolled.
- **RP name** — the label browsers show in their passkey UI (the domain is a
  fine default).
- Confirm the offered origin `https://your.domain`.
- If no password login exists yet it offers to create one now.
- It then offers your **detected reverse proxy** (it reads the established
  connections to the dashboard port) and writes
  `dashboard.trusted_proxies` for you. Without that setting the dashboard
  deliberately ignores `X-Forwarded-Proto` (fail-closed), thinks the request
  is plain HTTP, and the passkey option never appears — this is the single
  most-missed step. Non-interactive installs can pass `--trusted-proxy <ip>`.

## 5. Restart and enrol

Restart the dashboard, then enrol a key:

```sh
hermes dashboard webauthn enrol --label "my laptop"
```

Open the printed one-shot URL (valid 10 minutes) **in the browser you will log
in with** and complete the ceremony. Enrol a **second** key from a different
device — one lost authenticator must not lock you out. List or revoke keys
with `hermes dashboard webauthn --help`.

## 6. Log in

At `https://your.domain`: username + password, then the passkey prompt.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| No passkey button on the login page | `trusted_proxies` missing/wrong (step 4), or fido2 not installed (`hermes pm install --extra webauthn`) — then restart the dashboard. Check `curl -s https://your.domain/api/auth/providers` lists `webauthn`. |
| `{"detail": "Not Found"}` on an enrolment link | The link is one-shot and expires after 10 minutes, or the provider wasn't registered when it was minted. Re-run `enrol` and open the fresh URL. |
| 502 from the proxy | The dashboard is on the loopback bind (step 2) or the proxy target IP is wrong. |
| Enrolment page says the link is unknown/expired | Tokens are single-use; mint a new one. |

## Security model

- **RP ID binds passkeys forever** — changing it bricks every enrolled key.
- `trusted_proxies` must list your proxy's specific address; wildcard routes
  are refused (a broad trust would let any LAN peer forge
  `X-Forwarded-Proto` and impersonate the https gate).
- Plain-HTTP LAN access stays password-only by design; the passkey ceremony
  requires a secure context (https via your proxy, or an SSH-tunnel
  `http://localhost:<port>` path).
- Enrolment tokens are single-use, expire in 10 minutes, and the enrolment
  route answers 404 to anything without a valid token (no route probing).
