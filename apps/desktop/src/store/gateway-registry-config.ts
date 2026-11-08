import type { GatewayEvent } from '@hermes/shared'

import type { HermesConnection } from '@/global'
import type { ScopedServerRequest } from '@/store/gateway'

/** Hooks the gateway registry (store/gateway.ts) calls into its host. */
export interface RegistryConfig {
  /** Electron's published descriptor is authoritative for a primary gateway's
   * registry identity. Kept as a getter so gateway.ts does not own or duplicate
   * the connection store. */
  activeConnectionId?: () => null | string
  onEvent: (event: GatewayEvent) => void
  /** Server→client request (clarify, approval, …) from ANY socket the registry owns; the
   *  request's `respond` already routes to the socket it came from. `profile` /
   *  `connectionId` tag the source the same way events are tagged. */
  onServerRequest?: (request: ScopedServerRequest) => void
  onActiveConnectionInvalidated?: (fallbackProfile: string, activationEpoch: number) => void
  onActiveConnectionChanged?: (connection: HermesConnection) => void
  /**
   * Fires whenever applyActive() moves the active route to a (possibly
   * different) profile — including registry-internal eviction fallbacks
   * (connection removal, profile delete) that no renderer call
   * initiated. Consumers mirror this into $activeGatewayProfile so the
   * published profile can never diverge from the socket actually selected
   * (#89206: the stale-profile split-brain that stranded bot wake-ups).
   */
  onActiveRouteChanged?: (profile: string) => void
  /** Drop transient profile-pool runtime routes when local profile teardown
   * permanently retires their owning secondary. Exact registry routes are
   * deliberately outside this callback: a remote source may share the name. */
  onLocalProfileRetired?: (profile: string) => void
  /**
   * Scopes a FOREGROUND surface is bound to right now — every mounted
   * session tile's owner and the primary thread's (foregroundSessionScopes in
   * store/session-states; a config hook because that store imports this
   * one). Consulted by EVERY dispose path — the live-work pruner and the
   * dispose-at-refcount-0 request/relay leases alike (#93892): a tile's
   * resume mints its runtime on its owner's socket, and any path that closes
   * that socket makes the backend orphan-reap the runtime, whose
   * `session.reclaimed` unbinds the tile and re-arms its resume — a spinner
   * loop with no terminal state. Read at decision time, never cached: it
   * follows the tile set, so closing the tile releases the socket.
   */
  foregroundScopes?: () => ReadonlySet<string>
  /**
   * Scopes with a running or needs-input session runtime, in the same key
   * language as `foregroundScopes` (composite registry keys, bare profiles
   * for local/legacy entries). The wake-path liveness probe counts these as
   * in-flight work: prompt.submit returns before the turn ends, so
   * `activeRequests` is 0 during most of a turn.
   */
  liveScopes?: () => ReadonlySet<string>
}
