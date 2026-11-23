import { atom } from 'nanostores'

import type { HermesGateway } from '@/hermes'
import { initialGatewayProfile } from '@/lib/platform'
import type { RegistryConfig } from '@/store/gateway-registry-config'
import type { Secondary } from '@/store/gateway-socket-entry'

// ── HMR-stable module state ─────────────────────────────────────────────────
// All mutable singletons (live sockets, active-profile routing, the event
// registry) live in ONE container parked on globalThis, NOT in module-level
// `let`/`const` bindings. Reason: gateway.ts is imported widely without an HMR
// boundary that accepts it, so editing it (or anything that fans out to it)
// makes Vite issue a FULL PAGE RELOAD — which would kill every live socket and
// drop the agent session on an unrelated edit. Persisting the state on
// globalThis + self-accepting HMR (bottom of gateway.ts) turns that full reload into
// an in-place hot update that preserves the sockets. Production strips
// import.meta.hot, and a fresh page realm starts with an empty container, so the
// runtime behavior is identical to plain module state.
export interface GatewayRegistryState {
  config: RegistryConfig | null
  primaryGateway: HermesGateway | null
  primaryProfile: string
  activeKey: string
  secondaries: Map<string, Secondary>
  // Auth rejection outlives the disposable socket, including background request leases.
  reauthFailures: Map<string, { connectionId: string | null; error: Error }>
  // Dial failures outlive the disposable socket too: a background request lease disposes its
  // entry on failure, so history kept on the entry reset on every poll (#121865).
  dialFailures: Map<string, { at: number; streak: number; connectionId: string | null; ownerProbe?: boolean }>
  /** Scopes that opened in this renderer generation, even if later pruned. */
  openedSecondaryScopes?: Set<string>
  /** Scopes whose re-activation after a connection redial has not landed yet. */
  reactivatingScopes?: Set<string>
  /** Routed prompt sockets held until their terminal turn event arrives. */
  turnLeases: Map<string, () => void>
  /** Debounced releases so an immediate chained turn can reuse its lease. */
  turnLeaseReleaseTimers: Map<string, ReturnType<typeof setTimeout>>
  $gateway: ReturnType<typeof atom<HermesGateway | null>>
  $activeProfile: ReturnType<typeof atom<string>>
}

const STATE_KEY = Symbol.for('hermes.desktop.gatewayRegistryState')

function createRegistryState(): GatewayRegistryState {
  return {
    config: null,
    primaryGateway: null,
    primaryProfile: initialGatewayProfile(),
    activeKey: initialGatewayProfile(),
    secondaries: new Map<string, Secondary>(),
    reauthFailures: new Map(),
    dialFailures: new Map(),
    openedSecondaryScopes: new Set<string>(),
    reactivatingScopes: new Set<string>(),
    turnLeases: new Map<string, () => void>(),
    turnLeaseReleaseTimers: new Map<string, ReturnType<typeof setTimeout>>(),
    // The active gateway instance, exposed for inline message-stream
    // components (inline ClarifyTool, model overlays) that call gateway
    // methods without the instance threaded down through props.
    $gateway: atom<HermesGateway | null>(null),
    // The PROFILE the active gateway is routed to (bare profile name, never a
    // composite registry scope). Owned exclusively by applyActive() so the
    // published profile can never diverge from the socket actually selected —
    // the split-brain where an eviction re-pointed activeKey at the primary
    // while the profile atom kept naming the evicted bot routed every
    // "loki" session.resume to the default backend (#89206 wake failures).
    $activeProfile: atom<string>(initialGatewayProfile())
  }
}

// Dev only: park the singletons on globalThis so an HMR re-eval of this module
// (gateway.ts, self-accepted at its bottom) hands back the SAME live sockets/atoms instead
// of resetting them — that's what keeps the agent session alive across UI edits.
// `import.meta.hot` is undefined in production, so Vite dead-code-eliminates the
// entire globalThis branch and prod uses a plain module-local singleton — no
// globalThis, no Symbol.for. Both realms load the module once, so the container's
// shape and lifetime are identical either way.
export function gatewayState(): GatewayRegistryState {
  if (import.meta.hot) {
    const store = globalThis as unknown as { [STATE_KEY]?: GatewayRegistryState }
    store[STATE_KEY] ??= createRegistryState()

    // Existing dev-HMR containers predate whole-turn leases.
    store[STATE_KEY].reauthFailures ??= new Map()
    store[STATE_KEY].dialFailures ??= new Map()
    store[STATE_KEY].turnLeases ??= new Map()
    store[STATE_KEY].turnLeaseReleaseTimers ??= new Map()

    return store[STATE_KEY]
  }

  return createRegistryState()
}
