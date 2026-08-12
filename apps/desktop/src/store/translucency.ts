/**
 * Window translucency (see-through window).
 *
 * One lever, 0–100. 0 = off (fully opaque, the default). Higher = more of the
 * desktop shows through. Two modes decide HOW it shows through:
 *
 * - 'clear' — the main process maps the lever to native window opacity
 *   (`setOpacity`), the same effect as the Windows shift-scroll trick. The
 *   whole window fades, text included. macOS + Windows; Linux has no runtime
 *   window opacity, so it's a no-op there.
 * - 'glass' — macOS only. The window stays opaque at the native level; this
 *   store thins the renderer's page surfaces instead (see the
 *   `[data-hermes-glass]` block in styles.css), so the vibrancy material every
 *   chat window already carries shows through as a matte blur while text keeps
 *   full contrast.
 *
 * The renderer owns both values and mirrors them to the main process over IPC.
 */

import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

export type TranslucencyMode = 'clear' | 'glass'

const KEY = 'hermes.desktop.translucency.v1'
const MODE_KEY = 'hermes.desktop.translucency-mode.v1'

/** Glass rides on macOS vibrancy; other platforms only have Clear. */
export const GLASS_SUPPORTED = typeof navigator !== 'undefined' && /mac/i.test(navigator.platform || navigator.userAgent || '')

// Slider bounds — mirror `electron/window-opacity.ts` (TRANSLUCENCY_MIN / MAX)
// so the control and the main-process clamp agree on the same range. The step
// is renderer-only: the main process takes any integer in range.
export const TRANSLUCENCY_MIN = 0
export const TRANSLUCENCY_MAX = 100
export const TRANSLUCENCY_STEP = 1

const clamp = (n: number): number => Math.min(TRANSLUCENCY_MAX, Math.max(TRANSLUCENCY_MIN, Math.round(n)))

const read = (): number => {
  const n = Number(storedString(KEY))

  return Number.isFinite(n) ? clamp(n) : 0
}

const readMode = (): TranslucencyMode => (GLASS_SUPPORTED && storedString(MODE_KEY) === 'glass' ? 'glass' : 'clear')

export const $translucency = atom<number>(typeof window === 'undefined' ? 0 : read())

export const $translucencyMode = atom<TranslucencyMode>(typeof window === 'undefined' ? 'clear' : readMode())

export function setTranslucency(intensity: number): void {
  $translucency.set(clamp(intensity))
}

export function setTranslucencyMode(mode: TranslucencyMode): void {
  $translucencyMode.set(mode === 'glass' && GLASS_SUPPORTED ? 'glass' : 'clear')
}

// Glass thins surfaces only in real chat windows (primary + secondary session
// windows). The HUD, pet overlay, quick entry and wake indicator are
// transparent special-purpose windows that manage their own backgrounds — a
// page-surface rewrite there would fight them.
const isChatWindow = (): boolean => {
  try {
    const win = new URLSearchParams(window.location.search).get('win')

    return win === null || win === 'secondary'
  } catch {
    return false
  }
}

/**
 * Percent of the surface tint KEPT at a given intensity. Mirrors clear mode's
 * opacity ramp (floor 0.3): at 100 the surfaces keep a 30% wash so the glass
 * stays matte — tinted blur, not bare desktop.
 */
export const glassSurfaceKeep = (intensity: number): number => 100 - clamp(intensity) * 0.7

const applyGlassSurfaces = (intensity: number, mode: TranslucencyMode): void => {
  if (typeof document === 'undefined') {
    return
  }

  const root = document.documentElement
  const on = mode === 'glass' && intensity > 0 && GLASS_SUPPORTED && isChatWindow()

  if (on) {
    root.setAttribute('data-hermes-glass', '')
    root.style.setProperty('--translucency-glass-keep', `${glassSurfaceKeep(intensity)}%`)
  } else {
    root.removeAttribute('data-hermes-glass')
    root.style.removeProperty('--translucency-glass-keep')
  }
}

if (typeof window !== 'undefined') {
  const sync = () => {
    const intensity = $translucency.get()
    const mode = $translucencyMode.get()

    persistString(KEY, String(intensity))
    persistString(MODE_KEY, mode)
    applyGlassSurfaces(intensity, mode)
    window.hermesDesktop?.setTranslucency?.({ intensity, mode })
  }

  $translucency.subscribe(sync)
  $translucencyMode.subscribe(sync)
}
