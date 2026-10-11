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
