/**
 * Window translucency (see-through window).
 *
 * One lever, 0–100. 0 = off (fully opaque, the default). Higher = more of the
 * desktop shows through the whole window — the main process maps it to the
 * native window opacity (`setOpacity`), the same effect as the Windows
 * shift-scroll trick. macOS + Windows only; Linux has no runtime window
 * opacity, so it's a no-op there.
 *
 * The renderer owns the value and mirrors it to the main process over IPC.
 */

import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.translucency.v1'

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

export const $translucency = atom<number>(typeof window === 'undefined' ? 0 : read())

export function setTranslucency(intensity: number): void {
  $translucency.set(clamp(intensity))
}

if (typeof window !== 'undefined') {
  $translucency.subscribe(intensity => {
    persistString(KEY, String(intensity))
    window.hermesDesktop?.setTranslucency?.({ intensity })
  })
}
