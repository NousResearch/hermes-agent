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
  patches/                     # only patches NOT yet merged upstream (see below)
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

## Renderer-side fixes worth contributing directly (not as carried patches)

These live in `patches/` today only because this wrapper builds an unmodified
tag. In-tree they should be proper changes to `apps/desktop` and can drop from
`patches/`:

1. **UIScene lifecycle** — the renderer runs fine, but any WKWebView host on
   iOS 26/27 needs `UIApplicationSceneManifest` + a `SceneDelegate`. Belongs in
   the mobile host, documented for other embedders.
2. **Font path** — the renderer references `@nous-research/ui` fonts via an
   absolute `/node_modules/...` URL that 404s when served from a non-root/static
   host. A relative or configurable font base in `apps/desktop`'s build would
   remove the need for `fix-assets.mjs`.
3. **Touch toggle** (`patches/0001-*`) — `toggleSidebarOpen` /
   `toggleFileBrowserOpen` don't dispatch the pane-reveal event at narrow width
   (unlike `toggleReview`), so title-bar taps are dead on touch. A one-line fix
   in `apps/desktop/src/store/layout.ts` mirroring `toggleReview` fixes it for
   all narrow/touch consumers.

## Signing / identifiers

The project ships with an **empty** `DEVELOPMENT_TEAM` and the bundle id
`com.nousresearch.hermes.mobile`. Contributors set their own team in Xcode.
No signing certificate or provisioning profile is committed.

## Transport security

No hardcoded ATS exception ships (see `Info.plist`'s commented template).
Document that self-hosted HTTP gateways require an HTTPS front (e.g.
`tailscale serve --https=443`) or a user-added ATS exception.
