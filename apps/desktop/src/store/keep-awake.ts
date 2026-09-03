/**
 * Keep-awake — stop the machine sleeping during long, unattended runs.
 *
 * A device-local preference (each computer keeps its own), off by default. This
 * atom backs the Settings → Advanced picker and mirrors changes to the main
 * process, which owns the real power-save blocker AND its own persisted copy —
 * so a cold launch restores the blocker without the renderer visiting Settings
 * (see electron/main.ts + electron/power-save.ts). Linux/web without the bridge
 * just no-op.
 *
 * Modes: 'off' never holds the blocker; 'while-working' holds it only while a
 * turn is in flight (main reads that from the same active-work reports that
 * drive stream throttling); 'always' holds it around the clock, which is what
 * the pre-mode boolean toggle meant when it was on.
 */

import { atom } from 'nanostores'

import { persistString, storedBoolean, storedString } from '@/lib/storage'

export type KeepAwakeMode = 'always' | 'off' | 'while-working'

const KEY = 'hermes.desktop.keepAwakeMode.v1'
/** The boolean the toggle persisted before modes existed; read once to migrate. */
const LEGACY_KEY = 'hermes.desktop.keepAwake.v1'

function isKeepAwakeMode(value: unknown): value is KeepAwakeMode {
  return value === 'off' || value === 'while-working' || value === 'always'
}

function initialKeepAwakeMode(): KeepAwakeMode {
  const stored = storedString(KEY)

  if (isKeepAwakeMode(stored)) {
    return stored
  }

  // No mode saved yet: honour what the old toggle said. On meant "always".
  return storedBoolean(LEGACY_KEY, false) ? 'always' : 'off'
}

export const $keepAwakeMode = atom<KeepAwakeMode>(typeof window === 'undefined' ? 'off' : initialKeepAwakeMode())

export function setKeepAwakeMode(mode: KeepAwakeMode): void {
  $keepAwakeMode.set(mode)
}

if (typeof window !== 'undefined') {
  $keepAwakeMode.subscribe(mode => {
    persistString(KEY, mode)
    window.hermesDesktop?.setKeepAwake?.(mode)
  })
}
