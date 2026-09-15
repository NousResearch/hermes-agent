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

import { glassActive, type TranslucencyState, windowOpacityFor } from '../../shared/src/translucency'

export const TRANSLUCENCY_REASSERT_SETTLE_DELAY_MS = 400

type TranslucencyReassertWindow = object & {
  getBounds?: () => unknown
  isDestroyed?: () => boolean
  on?: (event: string, listener: () => void) => unknown
}

type TranslucencyReassertScreen = {
  getDisplayMatching?: (bounds: unknown) => { scaleFactor?: unknown } | null | undefined
  on?: (event: string, listener: (...args: unknown[]) => void) => unknown
}

export function displayMetricsRequireTranslucencyReassert(changedMetrics: unknown): boolean {
  return Array.isArray(changedMetrics) && changedMetrics.includes('scaleFactor')
}

export function scaleFactorRequiresTranslucencyReassert(previous: number | null, next: unknown): next is number {
  return typeof next === 'number' && Number.isFinite(next) && next > 0 && next !== previous
}

/**
 * Mixed-DPI DWM recovery writes. Electron 40.10.2's SetBackgroundMaterial('none')
 * paints white, so Glass-off / Clear must not enter this path. Opacity stays
 * out: a setOpacity write would layer the window and kill acrylic.
 */
export function translucencyReassertForDpiChange(state: TranslucencyState): {
  backing: true
  material: true
  opacity: false
} | null {
  return glassActive(state) ? { backing: true, material: true, opacity: false } : null
}

export function installTranslucencyReassertOnWindowEvents(
  win: TranslucencyReassertWindow,
  displayScreen: TranslucencyReassertScreen,
  reassert: () => void,
  lastScaleFactors: WeakMap<object, number>,
  platform = process.platform
): void {
  if (platform !== 'win32' || typeof win?.on !== 'function') {
    return
  }

  const reassertForScaleFactorChange = (): number | null => {
    if (win.isDestroyed?.() || typeof win.getBounds !== 'function') {
      return null
    }

    try {
      const bounds = win.getBounds()

      if (!bounds || typeof displayScreen?.getDisplayMatching !== 'function') {
        return null
      }

      const scaleFactor = displayScreen.getDisplayMatching(bounds)?.scaleFactor

      if (typeof scaleFactor !== 'number' || !Number.isFinite(scaleFactor) || scaleFactor <= 0) {
        return null
      }

      if (scaleFactorRequiresTranslucencyReassert(lastScaleFactors.get(win) ?? null, scaleFactor)) {
        lastScaleFactors.set(win, scaleFactor)
        reassert()
      }

      return scaleFactor
    } catch {
      return null
    }
  }

  win.on('show', () => {
    if (reassertForScaleFactorChange() === null) {
      return
    }

    setTimeout(() => {
      if (!win.isDestroyed?.()) {
        reassert()
      }
    }, TRANSLUCENCY_REASSERT_SETTLE_DELAY_MS)
  })
  // Electron 40.10.2 emits `moved` only after a manual drag finishes. `move`
  // fires while the window crosses displays; the scale-factor gate still
  // limits the native write to once at the transition.
  win.on('move', reassertForScaleFactorChange)
  win.on('moved', reassertForScaleFactorChange)
  win.on('resized', reassertForScaleFactorChange)
}

export function installTranslucencyReassertOnDisplayMetrics(
  displayScreen: TranslucencyReassertScreen,
  reassertAll: () => void,
  platform = process.platform
): void {
  if (platform !== 'win32' || typeof displayScreen?.on !== 'function') {
    return
  }

  displayScreen.on('display-metrics-changed', (_event, _display, changedMetrics) => {
    if (displayMetricsRequireTranslucencyReassert(changedMetrics)) {
      reassertAll()
    }
  })
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
