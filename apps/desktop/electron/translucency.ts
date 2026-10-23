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

import type { BrowserWindow } from 'electron'

import {
  backgroundMaterialFor,
  glassActive,
  type TranslucencyState,
  windowOpacityFor,
  type WindowsBackgroundMaterial
} from '../../shared/src/translucency'

/**
 * Electron 40.10.2's material setter also replaces the backing colour: white
 * for 'none', transparent for glass. Restore our backing after every material
 * write, even when the glass-active state did not change.
 */
export function applyWindowsBackdrop(
  win: Pick<BrowserWindow, 'setBackgroundColor' | 'setBackgroundMaterial'>,
  state: TranslucencyState,
  changed: { backing: boolean; material: boolean },
  themedColor: string
): void {
  if (changed.material) {
    win.setBackgroundMaterial(backgroundMaterialFor(state))
  }

  if (changed.backing || changed.material) {
    win.setBackgroundColor(glassActive(state) ? '#00000000' : themedColor)
  }
}

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
 * Opaque windows keep their constructor backing. Live material resets go
 * through applyWindowsBackdrop so Electron cannot replace the theme colour.
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
 * The native opacity a window should carry, given whether this platform can do
 * translucency at all.
 *
 * `translucencySupportedOn` drops the Settings row on everything but macOS and
 * Windows, on the strength of Electron documenting `setOpacity` as a no-op
 * elsewhere. On Linux it is not one: the X11 window carries the value as
 * `_NET_WM_WINDOW_OPACITY`, and a native Wayland surface composites the same
 * fade into the frame, so a Linux window really does render faded when handed
 * the value. The gate therefore belongs where the value reaches the window, not
 * only where the row is drawn, because a state persisted or imported from
 * another platform reaches this path without ever passing the Settings check.
 */
export function nativeOpacityFor(state: TranslucencyState, supported: boolean): number {
  return supported ? windowOpacityFor(state) : 1
}

/**
 * BrowserWindow constructor options for a chat window's native opacity. Empty
 * unless the state actually asks the window to fade — see opacityNeedsSetting.
 */
export function windowOpacityOptions(state: TranslucencyState, supported = true): { opacity?: number } {
  const opacity = nativeOpacityFor(state, supported)

  return opacityNeedsSetting(opacity) ? { opacity } : {}
}
