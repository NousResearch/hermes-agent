# Desktop UI in the browser

Use the existing Desktop renderer with a Hermes backend on your server. The
browser bundle replaces Electron's host adapter; chat, session state, models,
tools, skills, memory, files and Git still belong to the same backend.

## Run

From a source checkout with the Desktop workspace dependencies installed:

```sh
npm run build:browser --workspace apps/desktop
hermes serve --ui desktop --skip-build --host 127.0.0.1 --port 9119
```

Without `--skip-build`, the launcher prepares the Desktop workspace and builds
the browser bundle. `hermes dashboard --ui desktop` selects the same frontend.
Bare `hermes serve` stays headless; bare `hermes dashboard` keeps the existing
Dashboard. The option selects the frontend of that backend process; it does not
start another backend or proxy an already running server. Restart an existing
server with `--ui desktop` to enable the UI. Attach-first discovery refuses a
running server that serves a different frontend instead of opening the wrong UI.

For a WireGuard VPS, bind to its WireGuard address and restrict the port to
trusted VPN peers. Configure the existing `dashboard.public_url` and authentication
provider, then use that URL from your phone or browser. HTTPS is needed for browser
secure-context features such as microphone and clipboard access, including over
an encrypted VPN. An HTTPS reverse proxy should forward WebSocket upgrades and
preserve the configured public hostname. Existing reverse-proxy prefix support
(`X-Forwarded-Prefix`) also applies.

## Authentication and ownership

There is one origin and one authentication implementation: Hermes's existing
backend. Password login and OAuth use its login routes, callbacks, HttpOnly
cookies, refresh and logout. Each gated WebSocket dial requests a fresh single-use
`/api/auth/ws-ticket`. An ungated loopback instance uses the token injected by the
existing SPA bootstrap. No new token store, browser password form, authentication
namespace, CORS exemption or credential proxy is introduced.

A browser tab connects to the backend serving that tab. Open another backend's
URL in another tab to use it. Profiles on the same backend retain explicit profile
routing; session popouts preserve profile and spectator state. Native Desktop's
multi-server SSH/cloud connection manager is not a browser credential store.

Files selected or dropped from your phone/laptop use opaque attachment handles;
server project folders use a backend directory picker. Attachments are limited to
16 MB each, 32 retained handles, and a 30-minute lifetime. HTML previews run in an
opaque sandbox origin without backend cookies. External websites may refuse
iframe embedding through their own browser security policy; open them in a tab.

The terminal pane uses the backend's existing POSIX PTY bridge and WebSocket auth.
It is available only with the Desktop browser frontend, only for a local terminal
backend, and never on locked hosted filesystem deployments. It scrubs subprocess
credentials, caps concurrent shells at eight, and closes each shell on socket
shutdown. It deliberately refuses a host-shell fallback when the profile uses a
sandboxed terminal backend. Shell sessions end on network disconnect; agent work
continues independently in the backend.

OS-wide HUD/pet windows, global shortcuts, installing/updating a local Desktop
runtime, and native OS dialogs remain native application features. Browser tabs,
uploads/downloads, backend folders, and sandboxed frames provide browser equivalents
where possible. Link-title/favicon enrichment and automatic preview-file watching
are not available in this first browser adapter. Project defaults are browser
preferences scoped to the backend URL and profile. Changing backend model/provider settings still uses the normal
backend APIs; backend provider OAuth is independent of browser sign-in.

## Development and validation

The build reuses `vite.config.ts`, the Desktop dependencies, and the shared
renderer entry point. `tsconfig.browser.json` inherits strict checking; no
`noCheck` override is used. The browser preview adapter replaces the native
webview at build time while reusing the existing file/artifact preview components.

```sh
npm run typecheck:browser --workspace apps/desktop
npm run test:browser --workspace apps/desktop
npm run build:browser --workspace apps/desktop
scripts/run_tests.sh tests/hermes_cli/test_browser_frontend.py tests/hermes_cli/test_browser_terminal.py
scripts/run-in-hermes-env env HERMES_PYTHON=python3 npm run test:browser:e2e --workspace apps/desktop
python scripts/check
```

The browser journey uses the repository mock inference server and a temporary
Hermes home. It exercises real password login, cookie-authenticated chat/tool
streaming, reload/resume, file reads, spectator popouts and a phone viewport.
Backend tests separately cover the OAuth callback/session/ticket round trip and
real PTY execution, disconnect cleanup, capacity limits and sandbox refusal.

## Hosting performance

Serve the production build over HTTP/2 where available and enable gzip or Brotli
for public `/assets/` responses. Preserve the backend's immutable cache headers
for hashed assets; do not cache authenticated API responses. Compression can be
limited to public assets so login and other secret-bearing responses are excluded.
The browser build preloads the renderer during host/locale initialization without
evaluating it before host installation. Overlapping configuration reads share a
request within the same backend/profile/auth/timeout scope; results are not cached,
and writes invalidate overlapping reads at both start and completion.

On coarse-pointer touch devices editable fields use a 16px minimum font
size to prevent Safari focus zoom; larger user-selected text sizes remain intact.
The mobile journey checks focused composer/profile editor sizing and horizontal
overflow. At phone widths (640px and below), previews cover the viewport with a
Close action and safe-area spacing instead of squeezing the chat into a split
panel. Opening a phone preview dismisses input focus; its overlay blocks the
chat's type-to-focus behavior until closed. Wider viewports keep docked previews.
Touch tabs expose their close action without hover and keep it inside the visible
tab width. Tapping chat outside an expanded narrow sidebar dismisses it. The
browser footer uses 44px touch controls, wrapping its groups and allowing each
group to scroll horizontally rather than clipping actions.
