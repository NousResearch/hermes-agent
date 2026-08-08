/**
 * Long-horizon auto-continue — explicit mode switch (default OFF).
 *
 * Device-local preference mirrored to the main process, which also writes
 * `$HERMES_HOME/state/long-horizon-auto-continue.json` so the agent can read it.
 * Turning this on does NOT skip super-grill / budget gates (see runbook).
 */
import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const KEY = 'hermes.desktop.longHorizonAutoContinue.v1'

export const $longHorizonAutoContinue = atom<boolean>(
  typeof window === 'undefined' ? false : storedBoolean(KEY, false)
)

export function setLongHorizonAutoContinue(on: boolean): void {
  $longHorizonAutoContinue.set(on)
}

if (typeof window !== 'undefined') {
  $longHorizonAutoContinue.subscribe(on => {
    persistBoolean(KEY, on)
    window.hermesDesktop?.setLongHorizonAutoContinue?.(on)
  })
}
