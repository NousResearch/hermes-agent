// Keep the live primary gateway socket alive across Vite HMR so editing UI never
// drops the agent session. Stash on dispose, re-adopt on remount. globalThis +
// self-accept so this module's own reload doesn't reset the cache. Prod strips
// import.meta.hot → byte-for-byte unchanged live unmount.

import type { HermesConnection } from '@/global'
import type { HermesGateway } from '@/hermes'

export interface GatewaySurvivor {
  gateway: HermesGateway
  profile: string
  connection: HermesConnection | null
}

// One slot on globalThis, keyed by a process-stable Symbol so repeated imports
// (across hot reloads) resolve the exact same store.
const SURVIVOR_KEY = Symbol.for('hermes.desktop.gatewaySurvivor')

interface SurvivorGlobal {
  [SURVIVOR_KEY]?: GatewaySurvivor | null
}

function slot(): SurvivorGlobal {
  return globalThis as unknown as SurvivorGlobal
}

/** Park the live gateway so the next module instance can re-adopt it. */
export function stashGatewaySurvivor(survivor: GatewaySurvivor): void {
  slot()[SURVIVOR_KEY] = survivor
}

/**
 * Take the parked gateway, if any. Single-shot: the slot is cleared on read.
 * The caller decides whether to adopt or discard it via `survivorIsStale` — a
 * socket that died while parked (e.g. backend restart between edits) is still
 * returned so the caller can close it and boot fresh.
 */
export function takeGatewaySurvivor(): GatewaySurvivor | null {
  const store = slot()
  const survivor = store[SURVIVOR_KEY] ?? null
  store[SURVIVOR_KEY] = null

  return survivor
}

/** A parked survivor whose socket is no longer open — caller should discard it. */
export function survivorIsStale(survivor: GatewaySurvivor): boolean {
  const state = survivor.gateway.connectionState

  return state !== 'open' && state !== 'connecting'
}

// Self-accept so editing THIS module doesn't blow away the cache it manages.
if (import.meta.hot) {
  import.meta.hot.accept()
}
