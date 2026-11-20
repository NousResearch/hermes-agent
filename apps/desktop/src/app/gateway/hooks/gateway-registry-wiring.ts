// Topical sibling of use-gateway-boot.ts: how this window wires itself into the
// gateway registry (store/gateway.ts). The boot hook owns the lifecycle — `publish`,
// `deliverGatewayEvent` and the teardown latch are its — and this module only assembles
// the RegistryConfig they hang off, so the effect body stays a readable sequence of
// phases.

import { type GatewayEvent, JSON_RPC_METHOD_NOT_FOUND } from '@hermes/shared'

import type { HermesConnection } from '@/global'
import { RECONNECT_ATTEMPT_TIMEOUT_MS, withTimeout } from '@/lib/with-timeout'
import { gatewayActivationEpoch, type ScopedServerRequest } from '@/store/gateway'
import { secondaryReconnectsParked } from '@/store/gateway-park'
import type { RegistryConfig } from '@/store/gateway-registry-config'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $connection } from '@/store/session'
import { foregroundSessionScopes, forgetProfileOnlyRuntimeOwners } from '@/store/session-states'

import { liveWorkScopes } from './gateway-keepset'

/** The effect-body pieces the registry wiring closes over. */
export interface GatewayRegistryWiringContext {
  /** The hook's live-callbacks ref; a server→client request the host declines gets the -32601 answer. */
  callbacks: { current: { handleServerRequest: (request: ScopedServerRequest) => boolean } }
  /** Deliver a registry event through the dedupe gate into the store. */
  deliverGatewayEvent: (event: GatewayEvent) => void
  /** The window's IPC bridge (non-null: the effect guards it before wiring). */
  desktop: Window['hermesDesktop']
  /** The effect's teardown latch, read at call time. */
  isCancelled: () => boolean
  /** Publish the window's active connection; the registry's eviction fallback lands here. */
  publish: (connection: HermesConnection | null) => void
}

export function createGatewayRegistryWiring({
  callbacks,
  deliverGatewayEvent,
  desktop,
  isCancelled,
  publish
}: GatewayRegistryWiringContext): RegistryConfig {
  return {
    onServerRequest: request => {
      if (!callbacks.current.handleServerRequest(request)) {
        request.fail(JSON_RPC_METHOD_NOT_FOUND, `Hermes Desktop cannot answer ${request.method}`)
      }
    },
    // The primary socket has no secondary entry to carry registry identity.
    // Electron's published active descriptor is authoritative after boot;
    // a true legacy primary has no connectionId and remains unqualified.
    activeConnectionId: () => $connection.get()?.connectionId ?? null,
    // Every dispose path in the registry (live-work pruner AND the
    // refcount-0 request leases) spares a socket a mounted tile, the
    // primary thread or a just-created session's owner hold is bound to
    // (#93892).
    foregroundScopes: foregroundSessionScopes,
    // The keep-set follows the window's live work; consulted at dispose time, never during boot.
    liveScopes: () => liveWorkScopes(),
    // The registry's two dial points read this (#C2a): a parked window (hidden,
    // ladders standing down) opens no pool socket at all.
    routesParked: secondaryReconnectsParked,
    onLocalProfileRetired: forgetProfileOnlyRuntimeOwners,
    onActiveConnectionChanged: publish,
    // Keep $activeGatewayProfile in lockstep with the registry's OWN record
    // of which profile the active socket serves. The registry is the only
    // party that sees eviction fallbacks (connection removal,
    // profile delete → primary); before this mirror those fallbacks moved
    // the SOCKET back to the primary while the profile atom kept naming the
    // evicted bot. ensureGatewayProfile's "already active" fast path then
    // trusted the stale atom and skipped the re-swap, so every
    // session-scoped RPC for that bot went out on the primary socket — the
    // #89206 "Waking up… → retries gave up" wake failure, while the bot's
    // own backend sat healthy and idle.
    onActiveRouteChanged: profile => {
      const key = normalizeProfileKey(profile)

      if (normalizeProfileKey($activeGatewayProfile.get()) !== key) {
        $activeGatewayProfile.set(key)
      }
    },
    onEvent: deliverGatewayEvent,
    onActiveConnectionInvalidated: (fallbackProfile, invalidationEpoch) => {
      $activeGatewayProfile.set(fallbackProfile)
      // Bounded like every other getConnection() call in the boot hook (#93454):
      // an eviction fallback (connection removal, profile delete)
      // must not latch the profile atom to a connection that never resolves
      // if the main-process IPC round-trip wedges.
      void withTimeout(
        desktop.getConnection(fallbackProfile),
        RECONNECT_ATTEMPT_TIMEOUT_MS,
        'Timed out resolving the fallback gateway connection'
      )
        .then(connection => {
          if (!isCancelled() && gatewayActivationEpoch() === invalidationEpoch) {
            publish(connection)
          }
        })
        .catch(() => {
          if (!isCancelled() && gatewayActivationEpoch() === invalidationEpoch) {
            publish(null)
          }
        })
    }
  }
}
