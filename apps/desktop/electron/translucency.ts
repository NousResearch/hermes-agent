/**
 * Main-process side of window translucency.
 *
 * The mapping itself (modes, clamping, the clear-mode opacity ramp) lives in
 * apps/shared so the renderer and the main process cannot drift; this module
 * adds only what needs a BrowserWindow to mean anything.
 *
 * The import is relative rather than `@hermes/shared/translucency`: the
 * electron bundle is built by esbuild with no tsconfig path resolution (see
 * scripts/bundle-electron-main.mjs), so a bare specifier would typecheck and
 * then fail to bundle.
 */

import {
  backgroundMaterialFor,
  glassActive,
  type TranslucencyState,
  vibrancyFor,
  windowOpacityFor,
  type WindowsBackgroundMaterial
} from '../../shared/src/translucency'

export {
  backgroundMaterialFor,
  clampIntensity,
  DEFAULT_GLASS_MATERIAL,
  DEFAULT_GLASS_SCOPE,
  defaultTranslucencyState,
  defaultTranslucencyValues,
  GLASS_MATERIALS,
  GLASS_SCOPES,
  glassActive,
  type GlassMaterial,
  glassMaterialForPicker,
  glassMaterialsFor,
  glassSupportedOn,
  glassSurfaceKeep,
  hudFrostFor,
  normalizeBook,
  normalizeMaterial,
  normalizeMode,
  normalizeScope,
  normalizeState,
  resolveTranslucency,
  setTranslucencyValues,
  TRANSLUCENCY_CURVE,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_OPACITY_FLOOR,
  type TranslucencyState,
  translucencySupportedOn,
  vibrancyFor,
  windowOpacityFor,
  WINDOWS_BACKGROUND_MATERIALS,
  WINDOWS_GLASS_MIN_BUILD,
  type WindowsBackgroundMaterial
} from '../../shared/src/translucency'

/**
 * BrowserWindow constructor options for a chat window's backing, given the
 * translucency state at creation time.
 *
 * Glass active → OMIT `backgroundColor` entirely. Electron reads a window as
 * translucent when it carries a vibrancy or a backdrop material, and hands a
 * translucent window a transparent default backing, so the platform material
 * shows through the page from the first frame. Passing an alpha color instead
 * does NOT work — constructor alpha needs `transparent: true`, and `#00000000`
 * on a normal window is quietly treated as opaque.
 *
 * Glass inactive → the opaque themed backing (anti-flash paint before the
 * renderer's first paint, and what clear mode fades against).
 *
 * A runtime `setBackgroundColor` swap (see applyWindowTranslucency in main)
 * only settles reliably on a window that has been compositing for a while —
 * measured on macOS 26 / Electron 40, swaps issued during roughly the first
 * seconds of a fresh process were lost, including from 'ready-to-show' and
 * 'did-finish-load' — so creation must not rely on a post-creation fixup.
 */
export function windowBackingOptions(state: TranslucencyState, themedColor: string): { backgroundColor?: string } {
  return glassActive(state) ? {} : { backgroundColor: themedColor }
}

/**
 * Return the native Windows backdrop only when glass is actually active.
 * Electron treats even the `none` backdrop as a translucent-window opt-in,
 * which disables native maximize and resize commands on Windows.
 */
export function windowBackgroundMaterialOptions(
  state: TranslucencyState,
  isWindows: boolean,
  glassSupported: boolean
): { backgroundMaterial?: WindowsBackgroundMaterial } {
  return isWindows && glassSupported && glassActive(state) ? { backgroundMaterial: backgroundMaterialFor(state) } : {}
}

/**
 * Whether a window's native opacity is worth setting at all.
 *
 * Fully opaque is what a window already is, so asking for it looks free. On
 * Windows it is the opposite of free: `setOpacity` puts `WS_EX_LAYERED` on the
 * window and calls `SetLayeredWindowAttributes(..., LWA_ALPHA)` before it even
 * looks at the value, and nothing ever takes the style back off
 * (`NativeWindowViews::SetOpacity` → `SetLayered`). A layered window
 * composites through the legacy redirection surface — which Windows documents
 * as mutually exclusive with `UpdateLayeredWindow`, and which DWM will not
 * draw a system backdrop behind. So `opacity: 1` on a glass window buys
 * nothing and costs it its acrylic.
 *
 * `current` is the way back: a window that is already faded has already paid
 * for the layering, and it has to be able to return to opaque — so it keeps
 * getting the call even when the value it is going to is 1.
 *
 * What this cannot do is un-layer. Electron offers no way back off
 * `WS_EX_LAYERED`, so a Windows window that has been faded once keeps the
 * layered compositing path until it is recreated. Not opening the door on the
 * default path is the whole of the fix; someone who deliberately fades and
 * then returns to glass still wants a restart.
 */
export function opacityNeedsSetting(next: number, current = 1): boolean {
  return next < 1 || current < 1
}

/**
 * BrowserWindow constructor options for a chat window's native opacity. Empty
 * unless the state actually asks the window to fade — see opacityNeedsSetting.
 */
export function windowOpacityOptions(state: TranslucencyState): { opacity?: number } {
  const opacity = windowOpacityFor(state)

  return opacityNeedsSetting(opacity) ? { opacity } : {}
}

/** Opt-in for alpha-capable chat windows on Linux (`HERMES_DESKTOP_TRANSPARENT=1`). */
export const LINUX_TRANSPARENT_ENV = 'HERMES_DESKTOP_TRANSPARENT'

/**
 * Linux has no native material, so the translucency setting stays off there.
 * A compositor (Hyprland, KWin, picom, …) can still blur whatever sits behind
 * an alpha-capable window, which lets a theme's `customCSS` paint a
 * translucent background. Electron only honours alpha when the window is
 * CREATED `transparent`, so this is a launch-time opt-in rather than a live
 * toggle. Every built-in theme still paints an opaque page, so opting in
 * changes nothing until a theme asks for transparency; whole-window opacity
 * is never used, so text and images stay solid.
 */
export function linuxTransparentWindows(platform: string, env: Record<string, string | undefined>): boolean {
  return platform === 'linux' && env[LINUX_TRANSPARENT_ENV] === '1'
}

/** What `chatWindowSurfaceOptions` needs from the main process. */
export interface ChatWindowSurfaceEnv {
  isMac: boolean
  isWindows: boolean
  glassSupported: boolean
  /** `linuxTransparentWindows(...)` for this process. */
  linuxTransparent: boolean
  /** The opaque themed anti-flash backing. */
  themedColor: string
}

// Constructor options every chat window shares for its translucency surface:
// the platform material, the webContents backing, and a native opacity only if
// the state actually fades — all under the CURRENT state. Glass omits
// backgroundColor so the material shows from the first frame (Electron hands a
// translucent window a transparent default backing, and runtime swaps are lost
// early in a window's life — see applyWindowTranslucency); otherwise the opaque
// themed anti-flash backing. An opted-in Linux window is instead created
// `transparent` with a clear backing (see linuxTransparentWindows).
//
// Call sites also register the window in translucencyBackedWindows so a live
// toggle can re-apply. The HUD, pet overlay, quick entry and wake indicator
// are `transparent: true` windows that own their backgrounds and are
// deliberately not chat windows.
export function chatWindowSurfaceOptions(state: TranslucencyState, env: ChatWindowSurfaceEnv) {
  if (env.linuxTransparent) {
    return { transparent: true, backgroundColor: '#00000000' }
  }

  return {
    vibrancy: env.isMac ? vibrancyFor(state) : undefined,
    // Pin the material to its ACTIVE appearance: several NSVisualEffectView
    // materials collapse to a shared inactive look when the window blurs
    // (measured on macOS 26: sidebar, popover and under-window composited
    // pixel-identically once unfocused), which would quietly erase the
    // user's frost choice whenever they click elsewhere. Only observable
    // under glass — everywhere else the page buries the material.
    visualEffectState: env.isMac ? ('active' as const) : undefined,
    // NOT `transparent: true` on Windows. The backdrop material already makes
    // the window translucent on its own: `IsTranslucent` answers yes off
    // `background_material_` alone, which is what gives the page its transparent
    // default backing, and `SetBackgroundMaterial` flips widget translucency
    // live, so a Clear→Glass toggle needs no recreate either way. Its one gate
    // is a frameless window, and `titleBarStyle: 'hidden'` already makes
    // `has_frame()` false here.
    //
    // What `transparent` adds on top is permanent and unwanted: it pins the
    // widget to kTranslucent for the window's whole life, so even glass-OFF
    // windows pay a DirectComposition redraw per frame (electron#39895), and it
    // opts into the documented transparent-window limits — including that a
    // RESIZABLE transparent window is unsupported and breaks (electron#48421).
    // Every chat window is resizable.
    ...windowBackgroundMaterialOptions(state, env.isWindows, env.glassSupported),
    ...windowOpacityOptions(state),
    ...windowBackingOptions(state, env.themedColor)
  }
}

/** Runtime backing for a registered chat window: clear under glass or for an opted-in Linux window. */
export function windowBackingColor(state: TranslucencyState, themedColor: string, linuxTransparent: boolean): string {
  return linuxTransparent || glassActive(state) ? '#00000000' : themedColor
}
