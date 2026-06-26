// gateway-hmr-survivor.ts
//
// Keep the live primary gateway WebSocket (and its connection metadata) alive
// across a Vite HMR update so editing UI never severs the agent session.
//
// Why this exists
// ---------------
// `useGatewayBoot` owns the primary socket inside a mount-once `useEffect`. The
// Python backend session is long-lived, but on every HMR update React Fast
// Refresh remounts the component subtree — which runs that effect's cleanup
// (`gateway.close()`, `closeSecondaryGateways()`, null-outs). The socket dies,
// the in-flight stream is lost, and the composer goes dead until a full
// reconnect. For an agent you're actively chatting with, a one-character CSS
// tweak shouldn't drop the conversation.
//
// The fix is to stash the open gateway here on an HMR dispose, skip the
// destructive teardown, then re-adopt the same still-open socket when the freshly
// loaded effect re-runs — so the session keeps streaming straight through.
//
// HMR-stable storage
// ------------------
// The survivor must outlive arbitrary module replacement, including this
// module's own. We therefore (a) stash on `globalThis` under a Symbol, and (b)
// self-accept HMR so editing this file never resets the cache. Production builds
// strip `import.meta.hot`, so the whole survivor path is dev-only and tree-shakes
// to nothing — the live-unmount path stays byte-for-byte unchanged.

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

/** True only in a dev server with HMR wired up. Production strips this. */
export function hmrActive(): boolean {
  return Boolean(import.meta.hot)
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
