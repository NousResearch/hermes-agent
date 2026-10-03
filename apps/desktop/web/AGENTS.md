# Hermes Desktop Web Engineering Guide

Read this guide with the repository-root `AGENTS.md` and the Desktop-wide guide
at `apps/desktop/AGENTS.md`. This file contains the rules specific to the
browser host in this directory.

## Purpose and architecture

Hermes Desktop Web is the browser counterpart of native Hermes Desktop. It
imports the complete renderer directly from `../src` and replaces only the
Electron host boundary with a browser implementation of `window.hermesDesktop`.
It is not the Dashboard and must not become a second chat implementation.

```text
hermes dashboard
    └── independent Dashboard service, normally 127.0.0.1:9119

hermes desktop-web
    ├── Python Desktop Web launcher
    ├── owned headless Hermes backend child
    │   └── private OS-assigned loopback port
    └── standalone Node Web host
        └── normally 127.0.0.1:13043
```

`hermes dashboard` and `hermes desktop-web` are independent commands and may
run concurrently. Desktop Web must never call `cmd_dashboard`, reuse the
Dashboard lifecycle, stop Dashboard, or use Dashboard readiness/public-URL
identifiers.

## Native Desktop-like lifecycle

The canonical command is:

```bash
hermes desktop-web
```

Normal startup follows native Desktop’s local-runtime model:

1. Build `apps/desktop/web/dist`, or reuse it when the build is skipped.
2. Start an owned child equivalent to:

   ```text
   hermes serve --host 127.0.0.1 --port 0 --no-open --isolated
   ```

3. Wait for the child’s generic `HERMES_BACKEND_READY port=<N>` line.
4. Start `server.mjs` with `http://127.0.0.1:<N>` as its private backend URL.
5. Emit `HERMES_DESKTOP_WEB_READY port=13043` for the Web host.
6. On shutdown, interruption, or startup failure, clean up the Web host and
   the owned backend child together.

The child is the real Hermes Agent runtime. It owns profiles, sessions, model
calls, tools, Gateway RPC, filesystem/Git APIs, and agent execution. The Node
host owns static assets, browser capability adaptation, and authenticated
HTTP/WebSocket transport only.

The launcher sets `HERMES_DESKTOP_WEB=1` for the Web process and removes that
marker from the backend child. The child is marked internally with
`HERMES_DESKTOP_WEB_CHILD=1` and uses a private OS-assigned port; `13001` is
not a Desktop Web default.

## CLI contract

```bash
hermes desktop-web --help
hermes desktop-web
hermes desktop-web --no-open
hermes desktop-web --skip-build
hermes desktop-web --host 127.0.0.1 --port 13043
hermes desktop-web --status
hermes desktop-web --stop
```

Options:

- `--host` — Web host bind address; default `127.0.0.1`.
- `--port` — Web host port; default `13043`; `0` permits OS assignment.
- `--no-open` — do not open a browser automatically.
- `--skip-build` — use the existing `dist`; fail clearly if `index.html` is
  absent.
- `--status` — show only Desktop Web-owned processes and exit.
- `--stop` — stop only the Desktop Web host and its owned backend child.

Status and cleanup must use command identity and process ancestry. Never kill
every process listening on a port. `--stop` must not affect Dashboard, an
independently launched `hermes serve`, native Electron Desktop, or unrelated
applications.

## Source and build boundaries

The authoritative renderer is:

```text
apps/desktop/src/
```

The authoritative Electron host is:

```text
apps/desktop/electron/
```

Both are read-only boundaries for Desktop Web work. Do not copy renderer files
into this directory, edit the renderer or Electron host to solve a Web issue,
or add a second React chat surface.

`vite.config.ts` aliases `@desktop` and `@` to the live renderer tree. The Web
entrypoint must install the browser bridge before dynamically importing the
untouched Desktop entrypoint:

```ts
import { installWebHermesDesktop } from './preload'

installWebHermesDesktop()
void import('@desktop/main')
```

`src/preload.ts` is browser code implementing the renderer’s declared
`window.hermesDesktop` contract. It must not import Electron, `node:fs`,
`node:child_process`, `node-pty`, or other Node-only runtime modules into the
browser bundle.

## Browser capability policy

Use browser equivalents only when they preserve the Desktop contract:

| Capability | Browser behavior |
|---|---|
| HTTP API | same-origin request through the owned backend proxy |
| Gateway WebSocket | authenticated same-origin upgrade forwarding |
| Filesystem/project tree | authenticated backend filesystem API |
| Git/review | authenticated backend Git API |
| Clipboard | browser Clipboard API |
| Notifications | browser Notification API |
| Microphone | browser `getUserMedia` |
| File selection | browser file input, without fabricated server paths |
| Downloads | browser Blob/object URL download |
| External links | safe browser tab/window |

Native PTY/terminal launch, HUD or pet windows, global shortcuts, native file
manager dialogs, reveal/trash, Electron updater/bootstrap/uninstall, OS window
inspection, and native plugin installation are unsupported or safely disabled.
Unsupported methods must return controlled capability errors or harmless no-ops.
Never replace them with arbitrary command execution, fake absolute VPS paths, or
repeated disruptive `alert()` calls.

## Backend, connections, and settings

The browser host forwards backend-owned work; it does not execute agent logic.
The browser-facing connection/settings contract is implemented through the Web
host and stores Web connection metadata separately at:

```text
$HERMES_HOME/desktop-web/connections.json
```

Registry responses must not expose tokens or credentials. Do not put provider
keys, session tokens, or connection secrets in source, browser `localStorage`,
test fixtures, screenshots, or logs. A profile preference in localStorage is
not a credential and must remain distinct from secret storage.

Desktop Web’s optional public URL is separate from Dashboard:

```yaml
desktop_web:
  public_url: https://example.ts.net:13043
```

or:

```text
HERMES_DESKTOP_WEB_PUBLIC_URL
```

Never substitute:

```text
HERMES_DASHBOARD_READY
HERMES_DASHBOARD_PUBLIC_URL
dashboard.public_url
```

The backend child uses the active Hermes profile/home environment. Web-only
variables must not leak into it. Backend authentication remains authoritative;
preserve `401`/`403` responses and mint fresh one-time WebSocket tickets for
connections that require them.

## Standalone server and security

`server.mjs` is an independent static and HTTP/WebSocket proxy host. It must
remain separate from Dashboard server startup and must not invoke
`cmd_dashboard`. It provides `/healthz`, Web connection/settings routes, static
asset delivery, HTTP forwarding, and WebSocket upgrade forwarding.

Keep these protections intact:

- loopback-only binding by default;
- host/origin validation for Web-host and state-changing requests;
- bounded request bodies;
- static path traversal prevention;
- safe cookie/session forwarding;
- preserved upstream authentication failures;
- no unauthenticated arbitrary command endpoint;
- no permissive CORS workaround;
- no credential logging.

Filesystem, terminal, Git, and agent operations must remain authenticated
backend operations. Do not expose arbitrary shell execution from browser input.

## Tailscale and reverse proxy

The Web host speaks plain HTTP locally. Tailscale Serve must terminate HTTPS and
forward HTTP to the local listener:

```bash
tailscale serve --bg --https=13043 http://127.0.0.1:13043
tailscale serve status --json
curl -fsS http://127.0.0.1:13043/healthz
curl -k -I https://<tailscale-machine-name>:13043/
```

Do not configure the local upstream as `https://127.0.0.1:13043`; that sends TLS
to a plain-HTTP application and causes invalid-request or 502 failures. When
Host validation is enabled, use the canonical HTTPS hostname in
`desktop_web.public_url`.

## Development and verification

Use the existing workspace toolchain:

```bash
npm install --workspace apps/desktop/web
npm run typecheck --workspace apps/desktop/web
npm test --workspace apps/desktop/web
npm run build --workspace apps/desktop/web
node --check apps/desktop/web/server.mjs
python3 -m py_compile hermes_cli/main.py hermes_cli/subcommands/desktop_web.py
git diff --check
```

The Web typecheck project is `../tsconfig.web.json`. It includes the imported
Desktop/shared graph and excludes Electron. Its `noCheck` setting is
intentional because the untouched upstream renderer graph currently contains
unrelated declaration/union diagnostics. Vite’s production build is the full
browser module/syntax check, supplemented by focused Web tests. Do not solve
those diagnostics by editing `apps/desktop/src`.

Lifecycle acceptance requires verifying startup, both readiness markers,
`/healthz`, HTTP proxying, WebSocket proxying, private child-port behavior,
`--status`, `--stop`, and Dashboard survival after Desktop Web stops. Do not
claim authenticated Desktop parity without a real browser run covering the
shell, profiles, sessions, chat streaming, project tree, filesystem, and Git UI.

## Review checklist

- [ ] Renderer source is imported directly, not copied.
- [ ] `apps/desktop/src` and `apps/desktop/electron` are unchanged.
- [ ] Desktop Web does not call `cmd_dashboard`.
- [ ] Dashboard and Desktop Web run independently and concurrently.
- [ ] Desktop Web owns an isolated `hermes serve` child.
- [ ] The child uses a private OS-assigned port.
- [ ] Default Web port is `13043`.
- [ ] Web readiness is `HERMES_DESKTOP_WEB_READY`.
- [ ] Dashboard readiness remains `HERMES_DASHBOARD_READY`.
- [ ] Web configuration uses `desktop_web.public_url`.
- [ ] `--status` and `--stop` are process-tree scoped.
- [ ] Browser limitations fail safely.
- [ ] No arbitrary browser-driven command execution exists.
- [ ] Typecheck, tests, build, syntax, and diff checks pass.
- [ ] No credentials appear in code, fixtures, logs, or documentation.
