/**
 * Keep-awake — stop the machine sleeping during long, unattended runs.
 *
 * A device-local preference (each computer keeps its own), off by default. This
 * atom backs the Settings → Advanced toggle and mirrors changes to the main
 * process, which owns the real power-save blocker AND its own persisted copy —
 * so a cold launch restores the blocker without the renderer visiting Settings
 * (see electron/main.ts + electron/power-save.ts). Linux/web without the bridge
 * just no-op.
 */

import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.keepAwake.v1'

export const $keepAwake = atom<boolean>(typeof window === 'undefined' ? false : storedBoolean(KEY, false))

export function setKeepAwake(on: boolean): void {
  $keepAwake.set(on)
}

if (typeof window !== 'undefined') {
  $keepAwake.subscribe(on => {
    persistBoolean(KEY, on)
    window.hermesDesktop?.setKeepAwake?.(on)
  })
}
