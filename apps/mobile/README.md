# Hermes Mobile Client (iOS)

A native iOS app that bundles the **Hermes desktop renderer** (from
`nousresearch/hermes-agent`) and presents it 1:1 — same look and interaction as
the desktop app. It connects **only in remote-gateway mode** (base URL + session
token) to a self-hosted Hermes server. There is no local or cloud/OAuth mode,
and the app never modifies the server.

## 1. Architecture

```
desktop-port/build.sh
  ├─ checks out the hermes-agent renderer at a pinned tag (or a local source),
  │  resets it pristine, applies patches
  ├─ builds the renderer with vite            → desktop-port/vendor/apps/desktop/dist
  ├─ copies the result to                     → desktop-port/dist/
  ├─ injects the browser shim                 → desktop-port/dist/index.html
  └─ fixes bundled font paths (fix-assets.mjs)→ desktop-port/dist/ui-fonts/

Capacitor app (webDir: desktop-port/dist)
  ├─ WKWebView loads the bundled renderer locally (capacitor://localhost)
  ├─ browser shim window.hermesDesktop replaces the Electron bridge
  │    ├─ REST: CapacitorHttp (native URLSession) + header X-Hermes-Session-Token
  │    └─ WS:   direct WebSocket + ?token=
  └─ connection config: localStorage["hermes.remoteGateway"] = { url, token }
                                          ↓
                                   self-hosted Hermes gateway (unmodified)
```

**Where the shim sits:** `desktop-port/scripts/inject-shim.mjs` writes
`desktop-port/shim/hermes-web-shim.js` as a classic, synchronously-executed
`<script>` **before** the app bundle's `<script type="module">` in
`dist/index.html`, so `window.hermesDesktop` exists before the renderer's boot
path reads it.

**What the shim replaces:** on the desktop, the Electron main process provides
the `window.hermesDesktop` bridge (REST, WebSocket, connection config, the
gateway-settings panel methods). The shim implements the same contract in a
plain browser — remote-only, backed by the `{url, token}` block stored in
`localStorage`. Contract details with file:line references to the renderer
source are in `desktop-port/shim/CONTRACT.md`.

**Why CapacitorHttp:** the gateway only allows browser CORS for localhost
origins. Native HTTP (CapacitorHttp, no Origin header) bypasses that with no
server change — the same way the Electron main process does on desktop.

**iOS-specific adaptations** (all in the shim CSS/JS, additive, no renderer fork):
- **UIScene lifecycle** (`ios/App/App/SceneDelegate.swift` + `UIApplicationSceneManifest`
  in `Info.plist`): required on iOS 26/27, otherwise launch aborts with
  `NoSceneLifecycleAdoption`.
- **Static viewport:** `scrollEnabled:false` (capacitor.config) + a CSS lock
  (`overscroll-behavior`, pinch guard) — no page-level rubber-band scroll; only
  inner containers (transcript, lists) scroll.
- **Safe areas** (`env(safe-area-inset-*)`): the header clears the Dynamic
  Island; the status bar is a filled bottom bar reaching the screen edge with
  its text kept above the rounded-corner zone — device-adaptive (insets are 0
  on square-cornered devices).
- **Composer:** the detached "popout" mode is disabled on mobile so the input
  stays docked at the bottom.
- **Touch:** the side panels that collapse to overlays at narrow width are
  operable via the title-bar toggle buttons.

## 2. Build & install on a device

**Prerequisites:** Xcode, Node 22+, CocoaPods; a device (or simulator) that can
reach your gateway; an Apple Developer team configured in Xcode.

```bash
./desktop-port/build.sh      # or: HERMES_AGENT_SRC=/path/to/hermes-agent ./desktop-port/build.sh
npx cap sync ios
npx cap open ios             # in Xcode: pick your device/team, Run (▶)
```

`build.sh` produces `desktop-port/dist/`; `cap sync` copies it into the iOS
project. Set your own Apple Developer **team** and, if you fork the app id, your
own **bundle identifier** in Xcode's Signing & Capabilities (the project ships
with an empty team and the `com.nousresearch.hermes.mobile` id).

## 3. Connecting

On first launch (no stored connection) the app shows its own **"Connect"**
screen: enter the gateway base URL and the session token, then tap **Connect**.
The screen validates live against the gateway (HTTP **and** WebSocket) and only
persists on success — an invalid token never reaches `localStorage`.

**Token mode:** this client targets a token-mode gateway (`GET /api/status`
returns `auth_required: false`). Obtain the gateway's session token from your
server per the hermes-agent dashboard setup.

**Token rotation / server restart:** if the stored token becomes invalid, the
renderer enters its boot-failure state; the shim detects that (via a
MutationObserver on the boot-failure overlay) and re-opens the Connect screen
automatically — URL prefilled, token blank. Enter the new token and reconnect.

## 4. Transport security (HTTP gateways)

By default iOS allows only HTTPS. If your gateway is HTTPS (recommended, e.g.
`tailscale serve --https=443`), nothing else is needed. For a plain-HTTP gateway
on a private network, add an ATS exception for **your** host in
`ios/App/App/Info.plist` (a commented template is included there). Do not ship a
real hostname in a public build.

## 5. Updating to a new renderer version

1. Bump `PIN_TAG` in `desktop-port/build.sh` to the new hermes-agent tag.
2. `./desktop-port/build.sh`
3. `npx cap sync ios`
4. Rebuild in Xcode (Run) — or `npx cap run ios --target <device>`.

**WKWebView cache:** iOS caches bundled files aggressively across app updates.
`inject-shim.mjs` appends a content-based `?v=<hash>` to the shim URL
(cache-buster) so shim/layout changes load reliably after a rebuild + relaunch.
If a stale build persists, uninstall the app once
(`xcrun devicectl device uninstall app --device <id> <bundle-id>`) and reinstall
— note this clears `localStorage`, so the gateway URL and token must be
re-entered.

## 6. Deliberate scope (v1)

- No terminal panel (the desktop app's terminal is local-only, not remote-capable).
- No native push notifications.
- No in-app auto-update UI — updates go through a new build (§5).
- No pet-overlay window.
- Remote-gateway mode only — no local gateway, no cloud/OAuth mode.

## 7. Known refinements

- **Title-bar toggle hit targets** (~20×22 CSS px) are smaller than Apple's 44pt
  guidance. Their function was originally broken at narrow width (dead taps) and
  is fixed via `desktop-port/patches/0001-touch-toggle-sidebar-filebrowser-narrow-viewport.patch`;
  enlarging only the tap area (an invisible `::after`) is a possible follow-up.
- **Connect screen, empty token field:** if only the URL is changed and the
  token field is left blank, the shim reuses the previously stored token
  (mirrors the desktop app's token inheritance in `coerceRemote()`).
- **No focus trap in the Connect modal.**
- **Token storage:** the session token is stored in `localStorage` (as the
  desktop app does in remote mode). Hardening via the iOS Keychain is a possible
  next step.

## 8. Upstreaming

This project vendors the renderer at a pinned tag. See `UPSTREAMING.md` for how
it maps into `hermes-agent` as an in-tree `apps/mobile` subproject (build from
the local `apps/desktop` instead of cloning) and for the renderer-side fixes
worth contributing directly.

## License

MIT — see `LICENSE`. Bundles and builds on the Hermes renderer
(`nousresearch/hermes-agent`, MIT).
