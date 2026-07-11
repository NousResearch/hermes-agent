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

import { broadcastDesktopStateChange, onDesktopStateSync } from '@/lib/desktop-state-sync'
import { persistString, storedString } from '@/lib/storage'

const KEY = 'hermes.desktop.translucency.v1'

const clamp = (n: number): number => Math.min(100, Math.max(0, Math.round(n)))

const read = (): number => {
  const n = Number(storedString(KEY))

  return Number.isFinite(n) ? clamp(n) : 0
}

export const $translucency = atom<number>(typeof window === 'undefined' ? 0 : read())

export function setTranslucency(intensity: number): void {
  const next = clamp(intensity)
  $translucency.set(next)
  broadcastDesktopStateChange('translucency', { value: next })
}

if (typeof window !== 'undefined') {
  $translucency.subscribe(intensity => {
    persistString(KEY, String(intensity))
    window.hermesDesktop?.setTranslucency?.({ intensity })
  })

  onDesktopStateSync(message => {
    if (message.type === 'changed' && message.domain === 'translucency' && typeof message.value === 'number') {
      $translucency.set(clamp(message.value))
    }
  })
}
