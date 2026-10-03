# Hermes Desktop Web

Hermes Desktop Web is the browser counterpart of native Hermes Desktop. It
imports the complete Desktop renderer directly from `apps/desktop/src` and
replaces only the Electron host boundary with a browser adapter.

The Desktop renderer is not copied into this directory and must not be edited
to support Web.

## Architecture

```text
hermes dashboard
    └── independent Dashboard service, normally 127.0.0.1:9119

hermes desktop-web
    ├── Python Desktop Web launcher
    ├── dedicated headless Hermes backend child
    │   └── private OS-assigned loopback port
    └── standalone Web host
        └── 127.0.0.1:13043
```

The two commands are independent and may run in parallel. Desktop Web does not
call `hermes dashboard`, reuse its process, or stop it.

## Start

```bash
hermes desktop-web
```

Defaults:

```text
host: 127.0.0.1
port: 13043
```

The command follows the native Desktop lifecycle as closely as possible:

1. builds or reuses the Desktop Web `dist`;
2. starts an owned child:

   ```text
   hermes serve --host 127.0.0.1 --port 0 --no-open --isolated
   ```

3. waits for the child’s internal:

   ```text
   HERMES_BACKEND_READY port=<private-port>
   ```

4. starts `web/server.mjs` with that private backend URL;
5. emits:

   ```text
   HERMES_DESKTOP_WEB_READY port=13043
   ```

6. stops the Web host and its owned backend child together.

The backend child is the real Hermes Agent runtime. It owns sessions, profiles,
model/provider calls, tools, Gateway RPC, filesystem operations, Git operations,
and agent execution. The Web process is only the browser host and transport
adapter.

## Command options

```bash
hermes desktop-web --help
hermes desktop-web --host 127.0.0.1 --port 13043
hermes desktop-web --no-open
hermes desktop-web --skip-build
hermes desktop-web --status
hermes desktop-web --stop
```

- `--host` — Web host bind address; default `127.0.0.1`.
- `--port` — Web host port; default `13043`; `0` allows OS assignment.
- `--no-open` — do not open a browser.
- `--skip-build` — require and use the existing `dist/index.html`.
- `--status` — show only Desktop Web’s process tree and exit.
- `--stop` — stop only Desktop Web and its owned backend child.

`--stop` never stops Dashboard or an independently started Hermes backend.

## Browser bridge

`web/src/preload.ts` implements the browser version of the
`window.hermesDesktop` contract declared by the untouched Desktop renderer.

`web/src/main.ts` installs that bridge before importing the Desktop entrypoint:

```ts
import { installWebHermesDesktop } from './preload'

installWebHermesDesktop()
void import('@desktop/main')
```

Supported browser/backend paths include:

- HTTP API requests through the owned backend proxy;
- authenticated Gateway WebSocket forwarding;
- filesystem tree/read/write operations through backend APIs;
- Git status, diff, review, branch, and worktree APIs;
- browser clipboard and notifications;
- browser microphone permissions;
- browser file selection and downloads;
- external links and browser tabs;
- browser-local zoom behavior.

Electron-only capabilities are intentionally unavailable or degraded safely:

- native PTY terminal;
- OS terminal launching;
- HUD and pet overlay windows;
- global OS shortcuts;
- native file dialogs;
- OS reveal/trash/rename;
- Electron updater/bootstrap/uninstall;
- OS window inspection;
- native plugin installation;
- Electron-specific PR-comment operations.

Unsupported methods must not be implemented as arbitrary command execution or
repeated startup alerts.

## Backend and settings

Desktop Web starts a dedicated local backend child on every normal invocation,
matching native Desktop’s local startup behavior. The child uses the active
Hermes profile/home environment and listens only on a private OS-assigned
loopback port.

The browser bridge also exposes the Desktop connection/settings contract. Web
connection metadata is stored separately under:

```text
$HERMES_HOME/desktop-web/connections.json
```

Browser credentials are not stored in `localStorage` or returned in registry
responses. Backend authentication remains the source of truth.

Desktop Web has its own public URL configuration:

```yaml
desktop_web:
  public_url: https://example.ts.net:13043
```

or:

```text
HERMES_DESKTOP_WEB_PUBLIC_URL
```

Desktop Web authentication is independent from Dashboard authentication. Loopback
Host authorities bypass this host-level gate; every non-loopback/public authority
requires the bundled username/password provider before requests are proxied.
Configure it under the separate `desktop_web.basic_auth` namespace:

```yaml
desktop_web:
  public_url: https://example.ts.net:13043
  basic_auth:
    username: desktop-user
    password_hash: scrypt$N$r$p$base64-salt$base64-digest
    secret: 32-or-more-random-bytes
    session_ttl_seconds: 43200
```

`password_hash` is preferred and uses the same scrypt format as Dashboard’s
bundled basic provider. `password` is supported as a fallback and is hashed in
memory at startup. On the first interactive launch, when `public_url` or a
non-loopback bind engages the gate and no credentials are configured,
`hermes desktop-web` offers to collect the username and password, writes a
scrypt `password_hash` and signing `secret` to `config.yaml`, and then starts
normally. This onboarding is intentionally skipped for non-interactive
services and background launches; those fail closed instead of creating
credentials without an operator present. For secrets, environment variables
override config values:

```text
HERMES_DESKTOP_WEB_BASIC_AUTH_USERNAME
HERMES_DESKTOP_WEB_BASIC_AUTH_PASSWORD_HASH
HERMES_DESKTOP_WEB_BASIC_AUTH_PASSWORD
HERMES_DESKTOP_WEB_BASIC_AUTH_SECRET
HERMES_DESKTOP_WEB_BASIC_AUTH_TTL_SECONDS
```

Keep the password/signing secret in the profile `.env` where possible. The
Desktop Web cookie is HttpOnly, SameSite=Lax, Secure for HTTPS public authorities,
and never forwarded to the owned backend. The backend’s separate ephemeral native
session token is injected only into the private child proxy path and is never
returned to the browser.

Desktop Web does not use these Dashboard identities:

```text
HERMES_DASHBOARD_READY
HERMES_DASHBOARD_PUBLIC_URL
dashboard.public_url
```

## Tailscale

The Web host speaks plain HTTP locally. Tailscale Serve should terminate HTTPS
and proxy to HTTP:

```bash
tailscale serve --bg --https=13043 http://127.0.0.1:13043
tailscale serve status --json
```

Use the canonical HTTPS hostname:

```text
https://<tailscale-machine-name>:13043/
```

Do not configure the local upstream as
`https://127.0.0.1:13043`; that sends TLS to the plain-HTTP Web host and causes
invalid-request/502 errors.

When Host validation is active, configure `desktop_web.public_url` with the
canonical HTTPS hostname. Do not disable Host validation to work around a
wrong URL.

When Desktop Web is explicitly bound to `0.0.0.0` or `::`, it accepts any
syntactically valid Host authority so reverse proxies such as Cloudflare
Tunnels can forward their public hostname. Loopback and explicit-address binds
remain restricted to their intended hostnames, and malformed Host authorities
are still rejected. This Host behavior does not disable authentication,
same-origin checks, or backend authorization.

## Build and test

Use the existing Desktop workspace tools; do not add another frontend or proxy
stack:

```bash
npm install --workspace apps/desktop/web
npm run typecheck --workspace apps/desktop/web
npm test --workspace apps/desktop/web
npm run build --workspace apps/desktop/web
node --check apps/desktop/web/server.mjs
python3 -m py_compile hermes_cli/main.py hermes_cli/subcommands/desktop_web.py
```

The web build imports the complete Desktop source graph directly through the
Vite aliases in `web/vite.config.ts`. It uses the existing React Compiler,
Tailwind, Vite, TypeScript, and Vitest versions.

`tsconfig.web.json` includes the imported Desktop/shared graph and excludes
Electron. TypeScript `noCheck` is currently intentional because the untouched
upstream graph has unrelated declaration/union diagnostics; Vite’s production
build is the full browser module/syntax validation, supplemented by Web tests.

## Source rules for contributors

Do not:

- copy `apps/desktop/src` into `apps/desktop/web`;
- edit `apps/desktop/src/**`;
- edit `apps/desktop/electron/**`;
- import Electron or Node runtime modules into browser code;
- use `cmd_dashboard` for Desktop Web startup;
- add a second chat UI;
- add an unauthenticated arbitrary command endpoint;
- put credentials in source, browser storage, fixtures, screenshots, or logs.

The required CLI integration lives in:

```text
hermes_cli/main.py
hermes_cli/subcommands/desktop_web.py
```

Dashboard implementation files remain independent and should not be changed for
Desktop Web behavior.

See `plan001.md` for the detailed implementation plan and acceptance matrix.
