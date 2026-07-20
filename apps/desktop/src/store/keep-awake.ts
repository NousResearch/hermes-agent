/**
 * Keep-awake — stop the machine sleeping during long, unattended runs.
 *
 * A device-local preference (each computer keeps its own), off by default. The
 * renderer owns the value and persists it; the main process holds the actual
 * power-save blocker (see electron/power-save.ts) and re-reads this on every
 * window load via the subscribe below. Linux/web builds without the bridge just
 * no-op.
 */

import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.keepAwake.v1'

export const $keepAwake = atom<boolean>(typeof window === 'undefined' ? false : storedBoolean(KEY, false))

export function setKeepAwake(on: boolean): void {
  $keepAwake.set(on)
}

export function toggleKeepAwake(): void {
  $keepAwake.set(!$keepAwake.get())
}

if (typeof window !== 'undefined') {
  $keepAwake.subscribe(on => {
    persistBoolean(KEY, on)
    window.hermesDesktop?.setKeepAwake?.(on)
  })
}
