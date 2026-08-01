/**
 * AC ↔ battery mirror of the main process's powerMonitor.
 *
 * Backstop polls are already event-demoted and visibility-gated; battery is
 * the third gate — on battery their cadence stretches (×4) so a laptop
 * running Hermes in the background isn't spending its charge on safety-net
 * refreshes. Live streaming and event-driven refreshes are untouched: this
 * only slows timers whose job is catching what a degraded socket missed.
 *
 * Imported for its side effect from `main.tsx` (same pattern as
 * store/active-work).
 */

import { atom } from 'nanostores'

export const $onBattery = atom<boolean>(false)

/** Multiply a backstop poll interval by this on battery. */
export const BATTERY_POLL_MULTIPLIER = 4

export function batteryPollInterval(intervalMs: number, onBattery: boolean): number {
  return onBattery ? intervalMs * BATTERY_POLL_MULTIPLIER : intervalMs
}

if (typeof window !== 'undefined') {
  const desktop = window.hermesDesktop

  void desktop?.getOnBattery?.().then(onBattery => $onBattery.set(onBattery))
  desktop?.onBatteryChanged?.(onBattery => $onBattery.set(onBattery))
}
