/**
 * Keep-awake — stop the machine sleeping during long, unattended runs.
 *
 * A device-local preference (each computer keeps its own), 'while-working' by default. This
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

import { translateNow } from '@/i18n'
import { persistString, storedBoolean, storedString } from '@/lib/storage'
import { notify } from '@/store/notifications'

import { DEFAULT_KEEP_AWAKE_MODE, type KeepAwakeMode, parseKeepAwakeMode } from '../../electron/power-save'

export type { KeepAwakeMode }

const KEY = 'hermes.desktop.keepAwakeMode.v1'
/** The boolean the toggle persisted before modes existed; read once to migrate. */
const LEGACY_KEY = 'hermes.desktop.keepAwake.v1'

function initialKeepAwakeMode(): KeepAwakeMode {
  const stored = parseKeepAwakeMode(storedString(KEY))

  if (stored) {
    return stored
  }

  // No mode saved yet: honour what the old toggle said. On meant "always".
  if (storedString(LEGACY_KEY) !== null) {
    return storedBoolean(LEGACY_KEY, false) ? 'always' : 'off'
  }

  return DEFAULT_KEEP_AWAKE_MODE
}

export const $keepAwakeMode = atom<KeepAwakeMode>(
  typeof window === 'undefined' ? DEFAULT_KEEP_AWAKE_MODE : initialKeepAwakeMode()
)

export function setKeepAwakeMode(mode: KeepAwakeMode): void {
  $keepAwakeMode.set(mode)
}

/** The one-time offer main sends after a timed sleep interrupted a turn with keep-awake off. */
export function offerKeepAwake(): void {
  notify({
    id: 'keep-awake-offer',
    kind: 'info',
    title: translateNow('keepAwakeOffer.title'),
    message: translateNow('keepAwakeOffer.message'),
    action: { label: translateNow('keepAwakeOffer.action'), onClick: () => setKeepAwakeMode('while-working') },
    durationMs: 0,
    persistent: true
  })
}

if (typeof window !== 'undefined') {
  $keepAwakeMode.subscribe(mode => {
    persistString(KEY, mode)
    window.hermesDesktop?.setKeepAwake?.(mode)
  })
  window.hermesDesktop?.onKeepAwakeOffer?.(offerKeepAwake)
}
