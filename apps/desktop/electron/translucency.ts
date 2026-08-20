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

import { glassActive, type TranslucencyState } from '../../shared/src/translucency'

export {
  backgroundMaterialFor,
  clampIntensity,
  DEFAULT_GLASS_MATERIAL,
  DEFAULT_GLASS_SCOPE,
  GLASS_MATERIALS,
  GLASS_SCOPES,
  glassActive,
  type GlassMaterial,
  glassMaterialForPicker,
  glassMaterialsFor,
  glassSupportedOn,
  glassSurfaceKeep,
  hudFrostFor,
  normalizeMaterial,
  normalizeMode,
  normalizeScope,
  normalizeState,
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
 * Glass active → OMIT `backgroundColor` entirely: on a `vibrancy` window the
 * NSVisualEffectView then shows through a transparent page from the first
 * frame. Passing an alpha color instead does NOT work — Electron only supports
 * constructor alpha with `transparent: true`, and `#00000000` on a normal
 * window is quietly treated as opaque.
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
