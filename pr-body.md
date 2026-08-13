## Problem

On COSMIC / Wayland two things broke for Hermes Desktop's HUD:

1. **Always-on-top is ignored.** `cosmic-comp` does not honor Electron's
   `setAlwaysOnTop` for native-Wayland clients, so the floating overlay sinks
   behind other windows (issue #84011).
2. **Window-under awareness had no path.** COSMIC runs every app as a native
   Wayland client and exposes no X11 `_NET_CLIENT_LIST`, so the generic
   `get-windows`/xprop enumerator found nothing.

## Fix

This PR makes COSMIC a first-class, fully-working platform and lets the user
**choose XWayland or Native Wayland** — exactly as requested.

### 1. Always-on-top via XWayland (`desktop.ozone_platform_hint`)
A new config key `desktop.ozone_platform_hint` (default `auto`) bridges to
`ELECTRON_OZONE_PLATFORM_HINT` at launch, matching the existing
`desktop.disable_gpu` → `HERMES_DESKTOP_DISABLE_GPU` contract (explicit env
still wins). Set `ozone_platform_hint: x11` and Hermes becomes an X11 client,
which `cosmic-comp` honors for always-on-top. Behavior-neutral by default.

### 2. Native-Wayland window enumeration (`apps/desktop/cosmic-toplevel-list`)
A small Rust helper that connects to `cosmic-comp` over the **native Wayland**
protocol `ext_foreign_toplevel_list_v1` and prints every open toplevel as JSON
(`title`, `app_id`, `identifier`; `geometry: null`). Verified live: it
enumerates all real COSMIC windows with no X11 involved.

`apps/desktop/electron/cosmic.ts` is the provider: on COSMIC it shells out to
`cosmic-toplevel-list` first (native Wayland), and falls back to the shared
`get-windows`/xprop enumerator when the helper is absent (i.e. under
XWayland). It returns `null` off-COSMIC so the Hyprland → COSMIC → X11 fallback
chain is untouched. `window-below.ts` routes through it.

### The binary is built and shipped by the repo's own pipeline

`apps/desktop/cosmic-toplevel-list` is a first-party Rust crate. The desktop
pack builds and ships it:

- `scripts/stage-native-deps.mjs` gains `stageCosmicToplevelList({ platform })`
  — runs `cargo build --release` (Linux only; a no-op on macOS/Windows) and
  copies the executable into `dist/node_modules/cosmic-toplevel-list`. It is
  invoked from both the `npm run build` stage (CLI) and `before-pack.mjs`
  (per-target, alongside node-pty/get-windows). A missing cargo toolchain or
  source dir fails soft (warn + return null) so the desktop build still
  completes; the COSMIC path then falls back to XWayland.
- `package.json` `extraResources` ships `dist/node_modules/cosmic-toplevel-list`
  into `process.resourcesPath`.
- `apps/desktop/electron/cosmic.ts` resolves the binary from
  `process.resourcesPath`, probing both the directory-matcher and file-matcher
  layouts electron-builder can produce for `extraResources` (so it works
  regardless of which form the pack emits), with a bare-name PATH fallback for
  dev builds — so a stock `hermes desktop` produced by this repo's build
  actually runs it.

Verified: `node scripts/stage-native-deps.mjs linux` builds + stages the
binary, and the staged binary enumerates 12 live COSMIC windows over native
Wayland.

### Known COSMIC 1.0 limitation
`cosmic-comp` 1.0 serves `title`/`app_id`/`identifier` but **does not emit
geometry or pid** over its `zcosmic_toplevel_info_v1` extension (confirmed:
`get_cosmic_toplevel` is accepted but yields zero events). So on **native
Wayland** the HUD has name-based window awareness (`bounds` is reported as
null); for **pixel-exact geometry**, run under XWayland via
`ozone_platform_hint: x11`. Both options work — the user picks.

## Tests
- Python: `tests/hermes_cli/test_gui_command.py` — ozone hint parsing + env
  bridge (explicit env wins). 11 passed, 4 platform-skipped.
- TS vitest: `electron/cosmic.test.ts` (native helper preferred, fallback when
  helper missing/empty, null off-COSMIC) + `electron/window-below.test.ts`
  (COSMIC failure note). 24 passed.

## Verification (live, on pop-os COSMIC)
- `cosmic-toplevel-list` enumerated 12 real windows (COSMIC Terminal, Brave ×
  6, Slack, Paperclip, …) over native Wayland.
- `tsc --noEmit` clean across `cosmic.ts`/`window-below.ts`/`hyprland.ts`.

Fixes #84011
