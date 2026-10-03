# Upstreaming into hermes-agent

This repo is currently a **standalone Capacitor wrapper** that vendors the
Hermes renderer at a pinned tag. To land it in the official `nousresearch/hermes-agent`
monorepo as a first-class mobile client (alongside `apps/desktop` and the
dashboard), the following adjustments apply.

## Placement

Move the authored files under `apps/mobile/`:

```
apps/mobile/
  capacitor.config.ts
  ios/                         # Capacitor iOS project
  shim/hermes-web-shim.js      # browser implementation of the window.hermesDesktop bridge
  shim/CONTRACT.md
  scripts/inject-shim.mjs
  scripts/fix-assets.mjs
  test/                        # node --test suites (bridge behavior + build-path)
  build.sh
  package.json
```

## Build from the in-tree renderer (no vendoring)

`build.sh` supports building from a local renderer source instead of cloning:

```bash
HERMES_AGENT_SRC=../.. ./build.sh     # from apps/mobile, builds ../../apps/desktop
```

When `HERMES_AGENT_SRC` is set, the script builds `apps/desktop` in that tree
and skips the git clone + pristine-reset of the pinned vendor tag. In-tree,
drop the vendoring path entirely and depend on `apps/desktop` + `apps/shared`
directly.

## Renderer-side fixes

1. **Touch toggle** — **already handled upstream; no change needed.** Current
   `main` routes `toggleSidebarOpen` / `toggleFileBrowserOpen`
   (`apps/desktop/src/store/layout.ts`) through `revealNarrowPane()`, which
   dispatches the pane-reveal event at narrow width — so the title-bar toggles
   work on touch out of the box. (An earlier iteration of this port carried a
   vendor patch for this against an older tag; it has been dropped.)
2. **UIScene lifecycle** — the renderer runs fine, but any WKWebView host on
   iOS 26/27 needs `UIApplicationSceneManifest` + a `SceneDelegate`. Lives in
   the mobile host (`ios/`), documented for other embedders.
3. **Font path** — the renderer references `@nous-research/ui` fonts via an
   absolute `/node_modules/...` URL that 404s when served from a non-root/static
   host; `scripts/fix-assets.mjs` rebases them post-build (for both the
   standalone and in-tree builds). A relative or configurable font base in
   `apps/desktop`'s Vite build would remove the need for that script entirely.

## Signing / identifiers

The project ships with an **empty** `DEVELOPMENT_TEAM` and the bundle id
`com.nousresearch.hermes.mobile`. Contributors set their own team in Xcode.
No signing certificate or provisioning profile is committed.

## Transport security

No hardcoded ATS exception ships (see `Info.plist`'s commented template).
Document that self-hosted HTTP gateways require an HTTPS front (e.g.
`tailscale serve --https=443`) or a user-added ATS exception.

## Future SSH tunnel support (deliberately not v1)

The mobile client currently accepts a directly reachable HTTPS/token gateway.
Do not add an SSH implementation to the browser shim: the desktop's
`apps/desktop/electron/ssh-connection.ts` is an Electron-free OpenSSH manager,
but iOS cannot invoke the user's `ssh` binary or safely reuse its control
sockets. A responsible iOS implementation needs an explicitly approved native
SSH dependency or implementation, Keychain-backed credentials, host-key
verification, and lifecycle tests on a simulator/device. None of those can be
validated on Linux.

The smallest future integration seam should be a native, capability-scoped
`SshTunnel` service owned by the iOS host. Its contract should be limited to:

1. `open(configuration)`: authenticate and establish the SSH session, failing
   closed on host-key changes and never prompting from the WebView.
2. `forward(localPort, remoteHost, remotePort)`: bind only to `127.0.0.1`,
   return the selected local endpoint, and report tunnel death.
3. `cancelForward(...)` and `close()`: be idempotent and tear down all native
   resources when the scene resigns active or the connection changes.

The shim would remain transport-agnostic: after the native service returns a
loopback endpoint, the existing `{url, token}` remote-gateway path would use
that endpoint, with no private key or SSH details entering JavaScript. The
native layer must own credential storage, known-hosts policy, reconnect/backoff,
and logging redaction. The first implementation should add native unit tests
for command/endpoint validation and lifecycle idempotency, plus an iOS
integration test proving HTTP and WebSocket traffic both traverse the tunnel;
the existing Node shim tests should continue to cover profile guards and token
handling independently.
