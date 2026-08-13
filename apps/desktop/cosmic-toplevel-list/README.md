# cosmic-toplevel-list

A one-shot COSMIC (`cosmic-comp`) toplevel enumerator used by Hermes Desktop's
HUD to provide window-under awareness on the native Wayland COSMIC session.

It connects to the compositor over the Wayland protocol
`ext_foreign_toplevel_list_v1` and prints every open toplevel as JSON:

```json
[
  {
    "title": "brdpest@pop-os: ~ — COSMIC Terminal",
    "app_id": "com.system76.CosmicTerm",
    "identifier": "fMAyKoCdzve7Y9USulejTNjBVS8izFyS",
    "geometry": null
  }
]
```

`--active-only` prints just the focused window.

## Build

```sh
cargo build --release
# binary: target/release/cosmic-toplevel-list
```

Place the binary on `PATH` (or next to the Hermes Desktop executable) when
packaging. The desktop pack does this automatically: `scripts/stage-native-deps.mjs`
runs `cargo build --release` and copies the binary into
`dist/node_modules/cosmic-toplevel-list`, and electron-builder's
`extraResources` ships it into `process.resourcesPath` (where `cosmic.ts`
resolves it at runtime). On non-Linux targets the stage is a no-op.

## Why a separate binary?

The Wayland client is written in Rust (`wayland-client` +
`cosmic-protocols`). Shipping it as a small prebuilt helper keeps the Electron
app's Node dependency surface unchanged and mirrors how the app already shells
out to platform tools (`xprop`, Hyprland's socket).

## Known COSMIC 1.0 limitation

`cosmic-comp` 1.0 serves `title`/`app_id`/`identifier` but does **not** emit
`geometry` or `pid` over its `zcosmic_toplevel_info_v1` extension. Geometry is
therefore reported as `null`; for pixel-exact window positions, run Hermes
under XWayland (`desktop.ozone_platform_hint: x11`).
