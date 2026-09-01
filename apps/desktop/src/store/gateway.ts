import { type ConnectionState, type GatewayEvent, registryBackendScopeKey, resolveGatewayWsUrl } from '@hermes/shared'
import { atom } from 'nanostores'

import type { HermesConnection } from '@/global'
import { HermesGateway, setApiRequestConnection } from '@/hermes'
import { reconnectBackoffDelayMs } from '@/lib/reconnect-backoff'
import { RECONNECT_ATTEMPT_TIMEOUT_MS, withTimeout } from '@/lib/with-timeout'
import { markNativeNotifyBaseline } from '@/store/notify-baseline'
import { setConnection, setGatewayState } from '@/store/session'

// ── Multi-profile gateway routing ──────────────────────────────────────────
// Concurrent sessions across profiles need concurrent sockets: the renderer's
// event handler is already session-keyed, so the only thing stopping two
// profiles streaming at once was the single swapping socket. We keep that one
// socket as the PRIMARY (window) backend — owned by use-gateway-boot, with all
// its boot-progress / sleep-wake machinery — and add one persistent SECONDARY
// socket per *other* profile that has live work. Every socket feeds the same
// handleGatewayEvent, so background sessions keep painting. Single-profile users
// only ever have the primary, so their path is byte-for-byte unchanged.

const normKey = (profile: string | null | undefined): string => (profile ?? '').trim() || 'default'

// Read connection state through a call so TS control-flow analysis doesn't
// narrow the getter to a constant across guards (it genuinely changes).
const isOpen = (gateway: HermesGateway | null): boolean => gateway?.connectionState === 'open'

interface RegistryConfig {
  /** Electron's published descriptor is authoritative for a primary gateway's
   * registry identity. Kept as a getter so gateway.ts does not own or duplicate
   * the connection store. */
  activeConnectionId?: () => null | string
  onEvent: (event: GatewayEvent) => void
  onActiveConnectionInvalidated?: (fallbackProfile: string, activationEpoch: number) => void
  onActiveConnectionChanged?: (connection: HermesConnection) => void
  /**
   * Fires whenever applyActive() moves the active route to a (possibly
   * different) profile — including registry-internal eviction fallbacks
   * (idle reap, connection removal, profile delete) that no renderer call
   * initiated. Consumers mirror this into $activeGatewayProfile so the
   * published profile can never diverge from the socket actually selected
   * (#89206: the stale-profile split-brain that stranded bot wake-ups).
   */
  onActiveRouteChanged?: (profile: string) => void
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
}

// ── Secondary (pool) backends ──────────────────────────────────────────────
interface Secondary {
  /** Scope key from registryBackendScopeKey(connectionId, profile). */
  scope: string
  profile: string
  /** Registry connection serving this socket; null = the local/legacy path. */
  connectionId: null | string
  connection: HermesConnection | null
  gateway: HermesGateway
  /** True after this entry completed at least one socket connection. */
  openedOnce: boolean
  activeRequests: number
  connectPromise: Promise<void> | null
  offEvent: () => void
  offState: () => void
  reconnectTimer: ReturnType<typeof setTimeout> | null
  reconnectAttempt: number
  reconnecting: boolean
  /** A material connection edit is waiting for live owners to drain. */
  pendingConnectionRedial: boolean
  /**
   * True when a foreground/prewarmed consumer owns this entry beyond one RPC.
   * Guards ONLY the dispose-at-refcount-0 paths (request/relay leases), never
   * the live-work pruner: it is a one-way latch that every hover pre-warm and
   * profile switch sets and nothing ever clears, so honoring it in
   * pruneSecondaryGateways would pin every socket ever warmed. A foreground
   * surface that must keep its owner socket (a mounted session tile, the
   * primary thread) is represented in the pruner's keep-set instead — see
   * foregroundSessionScopes in store/session-states (#93892).
   */
  retained: boolean
  /**
   * Bot-relay retainers pinning this socket open across drain ticks (#93594).
   * The relay's drain loop RPCs every registered connection on an interval;
   * without retention each tick dialed and tore down a fresh WebSocket per
   * connection (refcount hit 0 → dispose). Counted, not boolean, so relay
   * retention can never clobber (or be clobbered by) the foreground
   * `retained` flag. Only non-local registry routes are ever counted here —
   * see retainGatewayForRelay.
   */
  relayRetainCount: number
  // While true the entry auto-reconnects on drop; pruning flips it off so a
  // deliberate close doesn't trigger the backoff loop.
  wantOpen: boolean
  /**
   * Epoch-ms deadline while an activation (prepare/ensure) is mid-dial. The
   * live-work pruner must not dispose an entry the user is switching to: a
   * switch target is not yet the active key, has no live sessions and holds
   * no request lease, so during a cold pool spawn (~3s) every prune recompute
   * saw it as idle garbage and disposed it mid-dial — the root of the dead
   * profile clicks in #89622. Cleared when the activation settles; bounded so
   * an orphaned lease self-heals.
   */
  activationLeaseUntil: number
}

// How long a mid-dial activation holds its prune lease: covers a cold pool
// backend spawn + socket connect with margin, while still letting a leaked
// lease expire quickly enough for the reaper to reclaim the entry.
const ACTIVATION_LEASE_MS = 30_000

// ── HMR-stable module state ─────────────────────────────────────────────────
// All mutable singletons (live sockets, active-profile routing, the event
// registry) live in ONE container parked on globalThis, NOT in module-level
// `let`/`const` bindings. Reason: this module is imported widely without an HMR
// boundary that accepts it, so editing it (or anything that fans out to it)
// makes Vite issue a FULL PAGE RELOAD — which would kill every live socket and
// drop the agent session on an unrelated edit. Persisting the state on
// globalThis + self-accepting HMR (bottom of file) turns that full reload into
// an in-place hot update that preserves the sockets. Production strips
// import.meta.hot, and a fresh page realm starts with an empty container, so the
// runtime behavior is identical to plain module state.
interface GatewayRegistryState {
  config: RegistryConfig | null
  primaryGateway: HermesGateway | null
  /** Registry source currently served by primaryGateway, when known. */
  primaryConnectionId: null | string
  primaryProfile: string
  activeKey: string
  activationEpoch: number
  secondaries: Map<string, Secondary>
  /** Scopes that opened in this renderer generation, even if later pruned. */
  openedSecondaryScopes?: Set<string>
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
    primaryConnectionId: null,
    primaryProfile: 'default',
    activeKey: 'default',
    activationEpoch: 0,
    secondaries: new Map<string, Secondary>(),
    openedSecondaryScopes: new Set<string>(),
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
    $activeProfile: atom<string>('default')
  }
}

// Dev only: park the singletons on globalThis so an HMR re-eval of this module
// (self-accepted at the bottom) hands back the SAME live sockets/atoms instead
// of resetting them — that's what keeps the agent session alive across UI edits.
// `import.meta.hot` is undefined in production, so Vite dead-code-eliminates the
// entire globalThis branch and prod uses a plain module-local singleton — no
// globalThis, no Symbol.for. Both realms load the module once, so the container's
// shape and lifetime are identical either way.
function gatewayState(): GatewayRegistryState {
  if (import.meta.hot) {
    const store = globalThis as unknown as { [STATE_KEY]?: GatewayRegistryState }
    store[STATE_KEY] ??= createRegistryState()

    // Existing dev-HMR containers predate whole-turn leases.
    store[STATE_KEY].turnLeases ??= new Map()
    store[STATE_KEY].turnLeaseReleaseTimers ??= new Map()

    return store[STATE_KEY]
  }

  return createRegistryState()
}

const g = gatewayState()

// Dev HMR can hand a newer module an older state-container shape. Keep the
// generation ledger lazy so an already-open socket still survives the update.
const openedSecondaryScopes = (): Set<string> => (g.openedSecondaryScopes ??= new Set<string>())

// Re-exported as a stable binding: the atom instance lives in `g`, so every hot
// reload of this module hands back the SAME atom subscribers are already wired
// to. (A fresh `atom()` per reload would orphan existing subscriptions.)
export const $gateway = g.$gateway

// The profile the ACTIVE gateway is actually routed to. Registry-owned: the
// only writer is applyActive(), which sets it in the same synchronous step
// that selects the socket — so a consumer that reads this and then calls
// activeGateway() always gets a matching (profile, socket) pair. Renderer
// surfaces (store/profile.ts's $activeGatewayProfile) mirror this atom
// instead of writing their own copy.
export const $activeGatewayRoute = g.$activeProfile

/** Bare profile name the active gateway serves (never a composite scope). */
export function activeGatewayProfileKey(): string {
  return g.$activeProfile.get()
}

export function configureGatewayRegistry(cfg: RegistryConfig): void {
  g.config = cfg
}

/**
 * Feed a synthetic event through the exact same fan-out a real socket frame
 * takes (`config.onEvent` → the desktop's `handleGatewayEvent`). Used by
 * dev-only tooling to exercise the real event branches (e.g. the credit-notice
 * demo) without a backend that can produce the event on demand. No-op until a
 * registry is configured.
 */
export function emitLocalGatewayEvent(event: GatewayEvent): void {
  g.config?.onEvent(event)
}

export function setPrimaryGateway(gateway: HermesGateway | null, profile = 'default'): void {
  const next = normKey(profile)

  if (g.primaryGateway !== gateway) {
    g.primaryConnectionId = null
  }

  // Route identity is exact-scope, never bare-name (#93892 follow-up): when
  // the active route IS the primary and the primary re-homes to another
  // profile, the active key must follow it. Leaving the old bare profile name
  // behind lets a later same-named LOCAL secondary inherit the active-route
  // spare in pruneSecondaryGateways — a remote tile keep-set of composite
  // scopes then appears to "pin" that unrelated local socket forever.
  if (g.activeKey === g.primaryProfile) {
    g.activeKey = next
  }

  g.primaryGateway = gateway
  g.primaryProfile = next

  if (g.activeKey === g.primaryProfile) {
    setApiRequestConnection(g.primaryConnectionId)
  }
}

export function setPrimaryGatewayConnectionId(connectionId: null | string | undefined): void {
  // Hardening for #95628: while the active route is a secondary scope, the
  // window is looking at a NON-primary socket — any connection id flowing
  // through presentation-layer code at that moment describes the secondary,
  // not the primary. Accepting it would relabel the primary socket, so every
  // ambient API/WebSocket helper (and new-session routing) silently lands on
  // the wrong backend. The primary's own identity is (re)published by its
  // boot/reconnect path, which runs with the primary route active.
  if (!isActivePrimary()) {
    return
  }

  g.primaryConnectionId = (connectionId ?? '').trim() || null

  if (g.activeKey === g.primaryProfile) {
    setApiRequestConnection(g.primaryConnectionId)
  }
}

/** Publish the registry source owned by the window primary socket. */
export function setPrimaryGatewayConnection(connection: Pick<HermesConnection, 'connectionId'> | null): void {
  setPrimaryGatewayConnectionId(connection?.connectionId)
}

function isPrimaryRegistryRoute(connectionId: null | string, profile: string): boolean {
  const id = String(connectionId ?? '').trim()

  return (
    normKey(profile) === g.primaryProfile &&
    Boolean(id) &&
    Boolean(g.primaryConnectionId) &&
    id === g.primaryConnectionId
  )
}

/** True when `connectionId` is the window's already-attached source AND that
 *  source is a one-host-many-profiles remote (`sharedRemote`). Named member
 *  profiles on that host must reuse the primary socket — a registry secondary
 *  dials a second WebSocket at the same Tailscale URL, which accept/closes in
 *  ~30ms (`messages=1`) and never runs `session.create` (#96493). Isolated
 *  SSH/pooled backends (`sharedRemote: false`) still get their own secondary. */
async function isAttachedSharedRemote(connectionId: null | string, profile: string): Promise<boolean> {
  const id = String(connectionId ?? '').trim()
  const key = normKey(profile)

  if (!id || !g.primaryConnectionId || id !== g.primaryConnectionId) {
    return false
  }

  if (isPrimaryRegistryRoute(id, key)) {
    return false
  }

  const desktop = window.hermesDesktop

  if (!desktop?.getConnectionFor) {
    return false
  }

  try {
    const conn = await withTimeout(
      desktop.getConnectionFor({ connectionId: id, profile: key }),
      RECONNECT_ATTEMPT_TIMEOUT_MS,
      `Timed out resolving shared-remote route for "${key}"`
    )

    return Boolean(conn && typeof conn === 'object' && (conn as { sharedRemote?: boolean }).sharedRemote === true)
  } catch {
    // Probe failed. A secondary at this already-attached source is the #96493
    // ghost WebSocket (accept/close, messages=1). Prefer the primary until a
    // later probe can prove isolation (`sharedRemote: false`). Isolated SSH
    // still dials its own socket when getConnectionFor succeeds.
    return true
  }
}

async function requestOnPrimaryGateway<T>(
  method: string,
  params: Record<string, unknown>,
  timeoutMs?: number,
  signal?: AbortSignal
): Promise<T> {
  const gateway = g.primaryGateway

  if (!gateway || !isOpen(gateway)) {
    throw new Error('Hermes gateway unavailable')
  }

  return timeoutMs === undefined && signal === undefined
    ? gateway.request<T>(method, params)
    : gateway.request<T>(method, params, timeoutMs, signal)
}

export function isActivePrimary(): boolean {
  return g.activeKey === g.primaryProfile
}

/** Changes on every active route selection, including same-profile source swaps. */
export function gatewayActivationEpoch(): number {
  return Number.isFinite(g.activationEpoch) ? g.activationEpoch : 0
}

export function activeGateway(): HermesGateway | null {
  if (g.activeKey === g.primaryProfile) {
    return g.primaryGateway
  }

  // A named scope resolves to ITS socket or nothing. Falling back to the
  // primary here would silently route calls (sends, session ops, roster
  // requests) to the WRONG backend whenever the scope's entry is gone —
  // teardown sites keep the invariant "activeKey always resolves" by
  // re-pointing the active key at the primary when they evict it.
  return g.secondaries.get(g.activeKey)?.gateway ?? null
}

/**
 * The registry connection serving the gateway the user is currently looking
 * at. A registry-backed primary takes its identity from the published primary
 * connection, falling back to Electron's active descriptor until that is set;
 * a true legacy primary (no resolved connectionId) and profile-keyed local
 * secondaries remain null. Event consumers pair this with the event's own
 * `connectionId` tag so "from the active profile" really means "from the active SOURCE":
 * two connected gateways can both expose a 'default' profile, and a bare
 * profile comparison attributed gateway B's 'default' activity to gateway A.
 */
export function activeGatewayConnectionId(): null | string {
  if (g.activeKey === g.primaryProfile) {
    return g.primaryConnectionId ?? (g.config?.activeConnectionId?.()?.trim() || null)
  }

  return g.secondaries.get(g.activeKey)?.connectionId ?? null
}

/**
 * Registry connections currently served by a live (open-socket) secondary.
 * Used by the reconnect path when the restarted primary's own registry
 * identity is unknown: Bot runtimes owned by these connections are provably
 * NOT the restarted backend and keep their bindings; everything else re-resumes.
 */
export function liveSecondaryConnectionIds(): Set<string> {
  const live = new Set<string>()

  for (const entry of g.secondaries.values()) {
    if (entry.connectionId && isOpen(entry.gateway)) {
      live.add(entry.connectionId)
    }
  }

  return live
}

// Mirror a backend's connection state into the global composer state, but only
// when that backend is the one the user is currently looking at. Lets the
// composer reflect the active profile's socket without a background reconnect
// flipping the foreground enabled/disabled state.
function reportGatewayState(profile: string, state: ConnectionState): void {
  // Any socket opening replays parked prompts; hold OS notifications so a
  // launch/reconnect doesn't alert about state that already existed.
  if (state === 'open') {
    markNativeNotifyBaseline()
  }

  if (normKey(profile) === g.activeKey) {
    setGatewayState(state)
  }
}

export function reportPrimaryGatewayState(state: ConnectionState): void {
  reportGatewayState(g.primaryProfile, state)
}

function setActive(profile: string): void {
  const activationEpoch = beginGatewayActivation()
  applyActive(profile, activationEpoch)
}

function beginGatewayActivation(): number {
  g.activationEpoch = gatewayActivationEpoch() + 1

  return g.activationEpoch
}

function applyActive(profile: string, activationEpoch: number): boolean {
  if (gatewayActivationEpoch() !== activationEpoch) {
    return false
  }

  g.activeKey = normKey(profile)
  const gateway = activeGateway()
  g.$gateway.set(gateway)
  setGatewayState(gateway?.connectionState ?? 'closed')
  // Push the active scope's registry connection into the hermes module (null
  // for the local pool) so connection-building WS calls (pluginSocket) resolve
  // through the same source of truth every activation path maintains here —
  // registry-agent activations included, not just profile switches.
  setApiRequestConnection(activeGatewayConnectionId())

  // Publish the BARE profile this route serves, in the same synchronous step
  // as the socket selection. activeKey may be a composite registry scope
  // (connectionId::profile); consumers route RPCs by profile, so resolve it
  // through the secondary's own record. This atom is the single source of
  // truth for "which profile is the active gateway on" — every eviction /
  // fallback path funnels through applyActive, so the published profile can
  // never linger on a backend that is no longer selected (#89206).
  const routeProfile =
    g.activeKey === g.primaryProfile ? g.primaryProfile : (g.secondaries.get(g.activeKey)?.profile ?? g.primaryProfile)

  g.$activeProfile.set(routeProfile)
  g.config?.onActiveRouteChanged?.(routeProfile)

  return true
}

function publishActiveConnection(connection: HermesConnection): void {
  if (g.config?.onActiveConnectionChanged) {
    g.config.onActiveConnectionChanged(connection)
  } else {
    setConnection(connection)
  }
}

function clearTimer(entry: Secondary): void {
  if (entry.reconnectTimer !== null) {
    clearTimeout(entry.reconnectTimer)
    entry.reconnectTimer = null
  }
}

async function openSecondary(entry: Secondary): Promise<void> {
  const desktop = window.hermesDesktop

  if (!desktop) {
    return
  }

  if (entry.connectPromise) {
    await entry.connectPromise

    return
  }

  const pending = (async () => {
    // A secondary can be reopened directly by the next routed user action,
    // without passing through reconnectSecondary(). Its previous backend may
    // have been respawned, so every stored→runtime binding for this exact scope
    // is process-local stale state. Invalidate BEFORE connect publishes `open`:
    // otherwise an eager route effect / submit can send the old runtime id in
    // the narrow window between the new socket opening and post-connect cleanup.
    //
    // Dynamic import keeps the existing session-states → gateway module cycle
    // open. Awaiting it is intentional: correctness at the generation boundary
    // outranks the single local-module microtask this adds to a reconnect.
    const openedScopes = openedSecondaryScopes()
    const reopening = entry.openedOnce || entry.connection !== null || openedScopes.has(entry.scope)
    let reconcileBusyAfterOpen: null | (() => void) = null

    if (reopening) {
      try {
        const { reconcileBusyStatesOnReconnect, resetTileRuntimeBindings } = await import('@/store/session-states')

        resetTileRuntimeBindings({
          connectionId: entry.connectionId || 'local',
          profile: entry.profile
        })
        reconcileBusyAfterOpen = () => reconcileBusyStatesOnReconnect(entry.scope)
      } catch {
        // Best effort for partial test/HMR graphs. Production always loads the
        // real store; a failed import must not make the transport unrecoverable.
      }

      // Runtime re-mint also invalidates the status-stack gone-latch: ids
      // the dead runtime 4001'd may be live again once tiles re-resume.
      // Fire-and-forget: composer-status imports from this module, so the
      // import must stay dynamic (cycle), and it must NOT sit on the timed
      // redial path — awaiting the module load here pushed cold-start
      // redials past test/waitFor budgets. The reset needs no ordering
      // guarantee relative to the dial.
      void import('@/store/composer-status')
        .then(({ resetBackgroundPollingGuard }) => resetBackgroundPollingGuard())
        .catch(() => {
          // Best effort for partial test/HMR graphs, same as above.
        })
    }

    // Registry-scoped entries dial through getConnectionFor when the bridge has
    // it. Local/legacy entries retain the existing getConnection path. Both are
    // IPC round-trips into the main process with no timeout of their own
    // (#93454) — a wedged main-process round-trip otherwise hangs this await
    // forever, latching entry.connectPromise so every routed action against
    // this secondary (SSH terminal, messaging DELETE, session send, …) never
    // settles either. Bound the same way use-gateway-boot.ts bounds the
    // primary's equivalent awaits.
    const conn =
      entry.connectionId && desktop.getConnectionFor
        ? await withTimeout(
            desktop.getConnectionFor({ connectionId: entry.connectionId, profile: entry.profile }),
            RECONNECT_ATTEMPT_TIMEOUT_MS,
            `Timed out connecting to profile "${entry.profile}"`
          )
        : await withTimeout(
            desktop.getConnection(entry.profile),
            RECONNECT_ATTEMPT_TIMEOUT_MS,
            `Timed out connecting to profile "${entry.profile}"`
          )

    entry.connection = conn

    const wsDeps =
      entry.connectionId && desktop.getGatewayWsUrlFor
        ? {
            getGatewayWsUrl: () =>
              desktop.getGatewayWsUrlFor!({ connectionId: entry.connectionId, profile: entry.profile })
          }
        : entry.connectionId
          ? {}
          : desktop

    const wsUrl = await withTimeout(
      resolveGatewayWsUrl(wsDeps, conn),
      RECONNECT_ATTEMPT_TIMEOUT_MS,
      `Timed out re-minting the gateway WebSocket URL for profile "${entry.profile}"`
    )

    try {
      await entry.gateway.connect(wsUrl)
    } catch (error) {
      // Log the dial target for support, but RETHROW THE ORIGINAL ERROR —
      // reconnectSecondary classifies failures by message ("No connection
      // with id", "no longer exists") to fail-stop permanent conditions, and
      // wrapping here would break that. Callers decide surfacing (#81094).
      console.error(`[gateway] dial failed for scope="${entry.scope}" profile="${entry.profile}":`, error)
      throw error
    }

    entry.openedOnce = true
    openedScopes.add(entry.scope)

    try {
      reconcileBusyAfterOpen?.()
    } catch {
      // The socket is already open. A best-effort UI-state reconcile must not
      // turn that successful transport recovery into a reported dial failure.
    }

    if (!entry.wantOpen) {
      entry.gateway.close()

      return
    }

    if (g.activeKey === entry.scope) {
      publishActiveConnection(conn)
    }

    void desktop.touchBackend?.(entry.scope).catch(() => undefined)
  })()

  entry.connectPromise = pending

  try {
    await pending
  } finally {
    if (entry.connectPromise === pending) {
      entry.connectPromise = null
    }
  }
}

function scheduleReconnect(entry: Secondary): void {
  if (entry.reconnecting || entry.reconnectTimer !== null || !entry.wantOpen) {
    return
  }

  // Full-jitter exponential backoff — same shape (and same reason: avoid a
  // reconnect storm against a restarting gateway) as the primary's.
  const delay = reconnectBackoffDelayMs(entry.reconnectAttempt)
  entry.reconnectAttempt += 1
  entry.reconnectTimer = setTimeout(() => {
    entry.reconnectTimer = null
    void reconnectSecondary(entry)
  }, delay)
}

async function reconnectSecondary(entry: Secondary): Promise<void> {
  if (entry.reconnecting || !entry.wantOpen || isOpen(entry.gateway)) {
    return
  }

  entry.reconnecting = true

  try {
    await openSecondary(entry)
    entry.reconnectAttempt = 0
  } catch (error) {
    // The registry no longer knows this connection (removed while we were
    // backing off), or Electron's deletion guard reports the profile itself
    // gone/mid-delete. Both are permanent for this scoped socket — retrying
    // forever can never succeed and hammers the spawn guard every backoff
    // tick (#88769). Fail-stop: dispose the entry and evict it instead of an
    // infinite 15s-cap retry loop.
    if ((entry.connectionId && isMissingConnectionError(error)) || isMissingProfileError(error)) {
      entry.reconnecting = false
      disposeSecondary(entry)

      if (g.secondaries.get(entry.scope) === entry) {
        g.secondaries.delete(entry.scope)
      }

      restoreActiveToPrimaryIfEvicted()

      return
    }
    // Other transport failure → fall through to the backoff below.
  } finally {
    entry.reconnecting = false

    if (entry.wantOpen && !isOpen(entry.gateway)) {
      scheduleReconnect(entry)
    }
  }
}

// Electron's getConnectionFor rejects with `No connection with id "…"` when
// the registry entry is gone. That is a permanent condition for the scoped
// socket, unlike transient transport errors.
function isMissingConnectionError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return message.includes('No connection with id')
}

// Electron's spawn guard (assertLocalProfileCanStart) rejects with these when
// the profile's directory is gone or its DELETE is still in flight. For a
// renderer socket that condition is permanent: the backend it reconnects to
// can never come back, and every retry hammers the guard (#88769).
function isMissingProfileError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return message.includes('no longer exists') || message.includes('is being deleted')
}

function createSecondary(profile: string, connectionId: null | string = null): Secondary {
  const gateway = new HermesGateway()
  const scope = registryBackendScopeKey(connectionId, profile)

  const entry: Secondary = {
    scope,
    profile,
    connectionId,
    connection: null,
    gateway,
    openedOnce: false,
    activeRequests: 0,
    connectPromise: null,
    offEvent: () => {},
    offState: () => {},
    reconnectTimer: null,
    reconnectAttempt: 0,
    reconnecting: false,
    pendingConnectionRedial: false,
    retained: false,
    relayRetainCount: 0,
    wantOpen: true,
    activationLeaseUntil: 0
  }

  // Events keep carrying the bare profile — session routing is profile-keyed
  // everywhere. connectionId rides along for surfaces that need the source.
  entry.offEvent = gateway.onEvent(event => {
    g.config?.onEvent({ ...event, profile, ...(connectionId ? { connectionId } : {}) })
    releaseTerminalTurnLease(entry.scope, event)
  })
  entry.offState = gateway.onState(state => {
    reportGatewayState(scope, state)

    if (state === 'open') {
      entry.reconnectAttempt = 0
      clearTimer(entry)
    } else if (state === 'closed' || state === 'error') {
      // A dead socket cannot emit the terminal event that normally releases
      // its turn lease. Drop the orphaned lease before deciding whether this
      // route is still retained/active enough to reconnect.
      releaseTurnLeasesForScope(scope)

      if (entry.wantOpen) {
        scheduleReconnect(entry)
      }
    }
  })

  g.secondaries.set(scope, entry)

  return entry
}

// True when `profile`'s backend route resolves to the SHARED primary backend
// (global-remote case 3 in resolveProfileBackendRoute). Both shared-primary and
// pooled descriptors carry `profile` so WebSocket URL minting targets the right
// profile. `sharedPrimary` is the explicit discriminator; treating every tagged
// descriptor as shared strands local/own-remote pooled profiles on the default
// socket. Dialing a second socket at the shared descriptor is wrong — over SSH
// the second dial fails (tunnel/token are per-backend) and the closed socket
// poisons the active gateway with "not connected" even though the primary is
// open right next to it.
async function sharedPrimaryRoute(profile: string): Promise<boolean> {
  const desktop = window.hermesDesktop

  if (!desktop) {
    return false
  }

  try {
    // Unbounded IPC round-trip into main (#93454) — a wedge here must reject
    // like any other failure, not hang the route decision forever, since
    // every caller (gatewayForProfile → requestGatewayForProfile/Agent) awaits
    // this before it can fall back to dialing a secondary.
    const conn = await withTimeout(
      desktop.getConnection(profile),
      RECONNECT_ATTEMPT_TIMEOUT_MS,
      `Timed out resolving the shared-primary route for profile "${profile}"`
    )

    return Boolean(conn && typeof conn === 'object' && (conn as { sharedPrimary?: boolean }).sharedPrimary === true)
  } catch {
    return false
  }
}

// Resolve and open `profile`'s socket WITHOUT changing the active gateway.
// Shared global-remote profiles intentionally return the primary socket plus a
// request-scope flag; dedicated local/remote profiles use their pooled socket.
async function gatewayForProfile(
  profile: string,
  leaseRequest = false
): Promise<{ gateway: HermesGateway | null; key: string; release: () => void; scopeProfile: boolean }> {
  const key = normKey(profile)
  const noRelease = () => undefined

  if (key === g.primaryProfile) {
    return { gateway: g.primaryGateway, key, release: noRelease, scopeProfile: false }
  }

  if (await sharedPrimaryRoute(key)) {
    return { gateway: g.primaryGateway, key, release: noRelease, scopeProfile: true }
  }

  const entry = g.secondaries.get(key) ?? createSecondary(key)

  // Existing dev-HMR entries predate the request lease/ownership fields.
  if (!Number.isFinite(entry.activeRequests)) {
    entry.activeRequests = 0
  }

  if (typeof entry.retained !== 'boolean') {
    entry.retained = true
  }

  if (!leaseRequest) {
    entry.retained = true
  }

  entry.wantOpen = true

  if (leaseRequest) {
    entry.activeRequests += 1
  }

  let released = false

  const release = () => {
    if (!released && leaseRequest) {
      released = true
      entry.activeRequests = Math.max(0, entry.activeRequests - 1)

      if (
        entry.activeRequests === 0 &&
        !entry.retained &&
        !relayRetained(entry) &&
        !foregroundPinned(entry) &&
        g.activeKey !== entry.scope
      ) {
        disposeSecondary(entry)

        if (g.secondaries.get(entry.scope) === entry) {
          g.secondaries.delete(entry.scope)
        }
      }
    }
  }

  try {
    if (!isOpen(entry.gateway)) {
      await openSecondary(entry)
    }
  } catch (error) {
    release()
    throw error
  }

  return { gateway: entry.gateway, key, release, scopeProfile: false }
}

/**
 * Send a gateway RPC through a named Desktop profile without foregrounding it.
 * Global-remote routes share the primary socket and need an explicit profile
 * param; dedicated pooled backends are already scoped by their descriptor.
 */
export async function requestGatewayForProfile<T>(
  profile: string,
  method: string,
  params: Record<string, unknown> = {},
  timeoutMs?: number,
  signal?: AbortSignal
): Promise<T> {
  const route = await gatewayForProfile(profile, true)

  try {
    if (!route.gateway) {
      throw new Error(`Hermes gateway unavailable for profile "${route.key}"`)
    }

    const routedParams = route.scopeProfile ? { ...params, profile: route.key } : params

    // Same arity contract as the ambient path in session-request-router: only
    // pass the deadline args through when the caller set them, so a plain
    // profile-routed RPC keeps its two-argument call shape.
    return await (timeoutMs === undefined && signal === undefined
      ? route.gateway.request<T>(method, routedParams)
      : route.gateway.request<T>(method, routedParams, timeoutMs, signal))
  } finally {
    route.release()
  }
}

/**
 * Send a gateway RPC through one registry source without activating it. The
 * composite (connectionId, profile) pool key prevents same-named agents on two
 * sources from sharing a socket. Only null/empty ids retain the v1 profile
 * resolver; explicit `local` is a registry source and must use getConnectionFor.
 */
export async function requestGatewayForAgent<T>(
  connectionId: null | string,
  profile: string,
  method: string,
  params: Record<string, unknown> = {},
  timeoutMs?: number,
  signal?: AbortSignal
): Promise<T> {
  const key = normKey(profile)
  const scope = registryBackendScopeKey(connectionId, key)

  if (scope === key) {
    return requestGatewayForProfile<T>(key, method, params, timeoutMs, signal)
  }

  // A primary remote selected from the connection registry carries its source
  // id in the active connection descriptor. Requests for that exact
  // (connection, profile) already have an owning socket: the window primary.
  // Dialing a registry secondary here can resolve the same public endpoint to a
  // different backend/profile route, so durable session.resume reports
  // "session not found" while REST history from the primary remains visible.
  // Require both owner identities to agree before collapsing the route; a
  // different source or profile must retain its isolated secondary.
  if (isPrimaryRegistryRoute(connectionId, key)) {
    return requestGatewayForProfile<T>(key, method, params, timeoutMs, signal)
  }

  if (await isAttachedSharedRemote(connectionId, key)) {
    return requestOnPrimaryGateway<T>(method, { ...params, profile: key }, timeoutMs, signal)
  }

  if (!window.hermesDesktop?.getConnectionFor) {
    throw new Error('This Desktop build cannot dial registry connections. Update Hermes Desktop.')
  }

  const entry = g.secondaries.get(scope) ?? createSecondary(key, connectionId)

  // Existing dev-HMR entries predate request leases/ownership.
  if (!Number.isFinite(entry.activeRequests)) {
    entry.activeRequests = 0
  }

  if (typeof entry.retained !== 'boolean') {
    entry.retained = true
  }

  entry.wantOpen = true
  entry.activeRequests += 1

  try {
    if (!isOpen(entry.gateway)) {
      await openSecondary(entry)
    }

    return await (timeoutMs === undefined && signal === undefined
      ? entry.gateway.request<T>(method, params)
      : entry.gateway.request<T>(method, params, timeoutMs, signal))
  } finally {
    entry.activeRequests = Math.max(0, entry.activeRequests - 1)

    if (
      !drainPendingConnectionRedial(entry) &&
      entry.activeRequests === 0 &&
      !entry.retained &&
      !relayRetained(entry) &&
      !foregroundPinned(entry) &&
      g.activeKey !== entry.scope
    ) {
      disposeSecondary(entry)

      if (g.secondaries.get(entry.scope) === entry) {
        g.secondaries.delete(entry.scope)
      }
    }
  }
}

// ── Bot-relay socket retention (#93594) ─────────────────────────────────────
// The desktop bot relay RPCs EVERY registered connection on its drain loop.
// Each of those calls runs through requestGatewayForAgent's per-request lease,
// so a connection with no other consumer dialed a fresh WebSocket and tore it
// down again on every tick — a connect/disconnect pair per connection per tick
// flooding the gateway logs. While the relay is active, its routes hold a
// counted retention that keeps the pooled socket (and its existing
// scheduleReconnect/backoff machinery) alive across ticks; stopBotRelay (and
// plugin dispose) releases it, restoring the dispose-at-refcount-0 behavior.

/**
 * True when a foreground surface (mounted tile / primary thread) is bound to
 * this entry's scope (#93892). Registry-scoped entries match on their
 * composite key only; local/legacy entries also match on the bare profile —
 * the same key language pruneSecondaryGateways' keep-set speaks.
 */
function foregroundPinned(entry: Secondary): boolean {
  const scopes = g.config?.foregroundScopes?.()

  if (!scopes) {
    return false
  }

  return scopes.has(entry.scope) || (!entry.connectionId && scopes.has(entry.profile))
}

/** True when the bot relay currently pins this entry open. Number guard:
 *  dev-HMR entries predate the field. */
function relayRetained(entry: Secondary): boolean {
  return Number.isFinite(entry.relayRetainCount) && entry.relayRetainCount > 0
}

/**
 * Finish a material-edit redial once no request, relay, or foreground surface
 * still owns the old socket. Removal deliberately bypasses this drain: a
 * deleted source can never become valid again and must fail-stop immediately.
 */
function drainPendingConnectionRedial(entry: Secondary): boolean {
  if (
    entry.pendingConnectionRedial !== true ||
    entry.activeRequests > 0 ||
    relayRetained(entry) ||
    foregroundPinned(entry) ||
    g.secondaries.get(entry.scope) !== entry
  ) {
    return false
  }

  entry.pendingConnectionRedial = false
  const wasActive = g.activeKey === entry.scope
  disposeSecondary(entry)
  g.secondaries.delete(entry.scope)

  const reopen = wasActive
    ? ensureGatewayForAgent(entry.connectionId, entry.profile)
    : openGatewayForAgent(entry.connectionId, entry.profile)

  void reopen.catch(() => undefined)

  return true
}

/**
 * Pin the pooled socket for one relay route open across drain ticks. Returns
 * a once-only release. Local routes (null/empty or explicit `local` source)
 * are deliberately EXEMPT and get a no-op release: their Electron-spawned
 * backend answers to the idle reaper, and a relay pin would keep the
 * touch-loop pinging it forever, resurrecting backends the reaper is meant to
 * reclaim (see the retireLocalProfileGateways note). Local relay traffic is
 * either the primary socket (no churn) or a short-lived local dial — never
 * the remote reconnect flood this retention exists to stop.
 */
export function retainGatewayForRelay(connectionId: null | string, profile: string): () => void {
  const key = normKey(profile)
  const connection = String(connectionId ?? '').trim()

  if (!connection || connection === 'local') {
    return () => undefined
  }

  const scope = registryBackendScopeKey(connection, key)
  const entry = g.secondaries.get(scope) ?? createSecondary(key, connection)

  if (!Number.isFinite(entry.relayRetainCount)) {
    entry.relayRetainCount = 0
  }

  entry.relayRetainCount += 1
  entry.wantOpen = true

  let released = false

  return () => {
    if (released) {
      return
    }

    released = true
    entry.relayRetainCount = Math.max(0, (entry.relayRetainCount || 0) - 1)

    if (
      !drainPendingConnectionRedial(entry) &&
      entry.relayRetainCount === 0 &&
      entry.activeRequests === 0 &&
      !entry.retained &&
      !foregroundPinned(entry) &&
      g.activeKey !== entry.scope &&
      g.secondaries.get(entry.scope) === entry
    ) {
      disposeSecondary(entry)
      g.secondaries.delete(entry.scope)
    }
  }
}

/**
 * Hold `profile`'s socket open across a multi-RPC sequence without activating
 * it (#93602). Every requestGatewayForProfile/requestGatewayForAgent call is a
 * per-request lease: at refcount 0 a non-retained secondary is disposed, so a
 * session-scoped sequence (session.create → attach → prompt.submit) minted a
 * runtime id on a socket that closed between calls — the gateway detached the
 * session on WS disconnect and the next RPC hit 4001 "not in memory". Callers
 * acquire this lease before the first session-scoped RPC and release it in a
 * `finally`; the refcount keeps the socket (and the session it minted) alive
 * for the whole sequence. Primary/shared-primary routes return a no-op release.
 */
export async function retainGatewayForAgent(connectionId: null | string, profile: string): Promise<() => void> {
  const key = normKey(profile)
  const scope = registryBackendScopeKey(connectionId, key)

  if (scope === key) {
    // Plain-profile route: gatewayForProfile's request lease IS the retain —
    // hold it until the caller releases.
    const route = await gatewayForProfile(key, true)

    return route.release
  }

  if (isPrimaryRegistryRoute(connectionId, key) || (await isAttachedSharedRemote(connectionId, key))) {
    // Primary socket stays open for the window lifetime — no secondary to hold.
    return () => undefined
  }

  if (!window.hermesDesktop?.getConnectionFor) {
    // No registry dialing in this build — nothing to hold; the request path
    // will throw its own actionable error.
    return () => undefined
  }

  const entry = g.secondaries.get(scope) ?? createSecondary(key, connectionId)

  // Existing dev-HMR entries predate request leases/ownership.
  if (!Number.isFinite(entry.activeRequests)) {
    entry.activeRequests = 0
  }

  if (typeof entry.retained !== 'boolean') {
    entry.retained = true
  }

  entry.wantOpen = true
  entry.activeRequests += 1

  let released = false

  const release = () => {
    if (released) {
      return
    }

    released = true
    entry.activeRequests = Math.max(0, entry.activeRequests - 1)

    if (drainPendingConnectionRedial(entry)) {
      return
    }

    if (
      entry.activeRequests === 0 &&
      !entry.retained &&
      !relayRetained(entry) &&
      !foregroundPinned(entry) &&
      g.activeKey !== entry.scope
    ) {
      disposeSecondary(entry)

      if (g.secondaries.get(entry.scope) === entry) {
        g.secondaries.delete(entry.scope)
      }
    }
  }

  try {
    if (!isOpen(entry.gateway)) {
      await openSecondary(entry)
    }
  } catch (error) {
    release()
    throw error
  }

  return release
}

const turnLeaseKey = (scope: string, sessionId: string): string => `${scope}\u0000${sessionId}`
const TURN_LEASE_SETTLE_DELAY_MS = 500

function cancelTurnLeaseRelease(key: string): void {
  const timer = g.turnLeaseReleaseTimers.get(key)

  if (timer !== undefined) {
    clearTimeout(timer)
    g.turnLeaseReleaseTimers.delete(key)
  }
}

function releaseTurnLeasesForScope(scope: string): void {
  const prefix = `${scope}\u0000`

  for (const [key, timer] of [...g.turnLeaseReleaseTimers]) {
    if (key.startsWith(prefix)) {
      clearTimeout(timer)
      g.turnLeaseReleaseTimers.delete(key)
    }
  }

  for (const [key, release] of [...g.turnLeases]) {
    if (key.startsWith(prefix)) {
      release()
    }
  }
}

/**
 * Keep a routed Desktop prompt's socket alive after prompt.submit ACKs.
 *
 * Routed requests normally own a per-request lease. prompt.submit ACKs as soon
 * as the background turn starts, so releasing that lease at RPC completion
 * detaches the runtime session while the model is still working; the gateway's
 * 20-second orphan guard then interrupts it as `client_gone`. Hold one lease per
 * (route, runtime session) until message.complete/session.info settles the turn.
 */
export async function retainGatewayForSessionTurn(
  connectionId: null | string,
  profile: string,
  sessionId: string
): Promise<() => void> {
  // Primary events do not flow through a Secondary's terminal-event listener.
  // Registering a no-op lease here would leave a phantom key that can suppress
  // the real hold if this route is later re-homed as a secondary.
  if (isPrimaryRegistryRoute(connectionId, normKey(profile))) {
    return () => undefined
  }

  const scope = registryBackendScopeKey(connectionId, normKey(profile))
  const key = turnLeaseKey(scope, sessionId)

  cancelTurnLeaseRelease(key)

  // A busy-session redirect/queue can submit again while the original turn is
  // still retained. The existing lease owns that turn; the extra submit must
  // not replace or release it. The no-op means "another caller owns the
  // shared lease", not "this caller acquired a separately releasable lease".
  if (g.turnLeases.has(key)) {
    return () => undefined
  }

  const releaseRoute = await retainGatewayForAgent(connectionId, profile)
  let released = false

  const release = () => {
    if (released) {
      return
    }

    released = true

    if (g.turnLeases.get(key) === release) {
      g.turnLeases.delete(key)
    }

    cancelTurnLeaseRelease(key)
    releaseRoute()
  }

  g.turnLeases.set(key, release)

  return release
}

function releaseTerminalTurnLease(scope: string, event: GatewayEvent): void {
  const sessionId = String(event.session_id || '').trim()

  if (!sessionId) {
    return
  }

  const key = turnLeaseKey(scope, sessionId)

  if (event.type === 'message.start') {
    // The gateway emits settled session.info before immediately chaining a
    // queued/goal follow-up. Keep the same route alive for that next turn.
    cancelTurnLeaseRelease(key)

    return
  }

  if (event.type === 'session.reclaimed') {
    g.turnLeases.get(key)?.()

    return
  }

  const payload = event.payload as Record<string, unknown> | undefined

  if (event.type === 'session.info' && payload?.running === false && !g.turnLeaseReleaseTimers.has(key)) {
    // session.info(false) is the authoritative settled edge, but auto-followup
    // emits message.start immediately after it. A short debounce lets that
    // frame cancel release while still reclaiming ordinary completed turns.
    g.turnLeaseReleaseTimers.set(
      key,
      setTimeout(() => {
        g.turnLeaseReleaseTimers.delete(key)
        g.turnLeases.get(key)?.()
      }, TURN_LEASE_SETTLE_DELAY_MS)
    )
  }
}

// Open `profile`'s socket WITHOUT making it active — the hover-intent pre-warm
// (store/profile). Runs the same spawn + connect chain as a real switch, so by
// click time ensureGatewayForProfile finds an open socket and just activates
// it. No scheduleReconnect on failure: a hover is speculative, so a dead
// backend must not start a background retry loop — the real switch owns retry
// and error UX. An already-open (or primary) profile is a no-op.
export async function openGatewayForProfile(profile: string): Promise<void> {
  await gatewayForProfile(profile)
}

// ── Connection-scoped agents (multi-source roster) ─────────────────────────
// The (connectionId, profile) analogues of the profile functions above. A
// null connectionId falls straight through to the profile path. An explicit
// `local` id remains registry-scoped so it cannot inherit legacy remote v1
// routing. Feature-detected: without the Electron getConnectionFor door these
// throw, and roster surfaces disable non-local rows instead.

// `activationLease`: hold the same prune lease ensureGatewayForAgent holds for
// the whole dial. Phase one of the two-phase source switch (store/connections
// selectConnection) opens the target here and activates it right after; without
// the lease a live-work recompute during the cold spawn would dispose the entry
// mid-dial and the click would die (#89622). Plain pre-warms stay prunable —
// a hovered-but-never-activated socket must not be pinned off another source's
// live work.
export async function openGatewayForAgent(
  connectionId: null | string,
  profile: string,
  { activationLease = false }: { activationLease?: boolean } = {}
): Promise<void> {
  const scope = registryBackendScopeKey(connectionId, profile)

  if (scope === normKey(profile) || isPrimaryRegistryRoute(connectionId, profile)) {
    return openGatewayForProfile(profile)
  }

  if (await isAttachedSharedRemote(connectionId, profile)) {
    if (!isOpen(g.primaryGateway)) {
      throw new Error('Hermes gateway unavailable')
    }

    return
  }

  if (!window.hermesDesktop?.getConnectionFor) {
    throw new Error('This Desktop build cannot dial registry connections. Update Hermes Desktop.')
  }

  const entry = g.secondaries.get(scope) ?? createSecondary(profile, connectionId)
  entry.retained = true
  entry.wantOpen = true

  if (activationLease) {
    // Stays held after a successful open: the activation that follows releases
    // it (applyActive path), and one that never comes lets it expire.
    entry.activationLeaseUntil = Date.now() + ACTIVATION_LEASE_MS
  }

  if (isOpen(entry.gateway)) {
    return
  }

  try {
    await openSecondary(entry)
  } catch (error) {
    if (activationLease) {
      entry.activationLeaseUntil = 0
    }

    throw error
  }
}

export async function ensureGatewayForAgent(
  connectionId: null | string,
  profile: string,
  { signal }: { signal?: AbortSignal } = {}
): Promise<boolean> {
  const scope = registryBackendScopeKey(connectionId, profile)

  if (scope === normKey(profile) || isPrimaryRegistryRoute(connectionId, profile)) {
    if (signal?.aborted) {
      return false
    }

    await ensureGatewayForProfile(profile)

    return !signal?.aborted
  }

  if (await isAttachedSharedRemote(connectionId, profile)) {
    return Boolean(isOpen(g.primaryGateway) && !signal?.aborted)
  }

  if (!window.hermesDesktop?.getConnectionFor) {
    throw new Error('This Desktop build cannot dial registry connections. Update Hermes Desktop.')
  }

  const activationEpoch = beginGatewayActivation()

  let entry = g.secondaries.get(scope)

  if (!entry) {
    entry = createSecondary(profile, connectionId)
  }

  entry.retained = true
  entry.wantOpen = true
  // Lease the entry against the live-work pruner for the whole dial: the
  // switch target is not yet active and has no live sessions, so a prune
  // recompute firing mid-spawn would otherwise dispose it and this
  // activation would fail (#89622).
  entry.activationLeaseUntil = Date.now() + ACTIVATION_LEASE_MS

  if (!isOpen(entry.gateway)) {
    clearTimer(entry)
    entry.reconnectAttempt = 0

    try {
      await openSecondary(entry)
    } catch {
      scheduleReconnect(entry)
    }
  }

  // The activation is settling either way — release the prune lease.
  entry.activationLeaseUntil = 0

  // A timed-out owner may leave the dial running, but it no longer has the
  // right to move the foreground route when that work eventually settles.
  if (signal?.aborted) {
    return false
  }

  // A source edit/remove may dispose this entry while its dial is still in
  // flight. Only the still-registered, still-owned activation may publish --
  // and only when the WebSocket actually reached open: entry.connection is
  // set BEFORE the dial completes in openSecondary, so a transient first-dial
  // failure (caught above, left for scheduleReconnect) must not count as a
  // successful activation just because a connection descriptor exists
  // (issue #92265).
  const activated =
    entry.wantOpen &&
    g.secondaries.get(scope) === entry &&
    Boolean(entry.connection) &&
    isOpen(entry.gateway) &&
    applyActive(scope, activationEpoch)

  if (activated && entry.connection) {
    publishActiveConnection(entry.connection)
  }

  return activated
}

// Make `profile` the active gateway, lazily opening its socket if needed. The
// primary is a no-op fast path. Background sockets are never closed here.
export async function ensureGatewayForProfile(profile: string): Promise<void> {
  const key = normKey(profile)
  const activationEpoch = beginGatewayActivation()

  if (key === g.primaryProfile) {
    applyActive(key, activationEpoch)

    return
  }

  // Global-remote share (routing case 3): one remote host serves every
  // profile through the PRIMARY socket, scoped per request. Activate the
  // primary instead of dialing a doomed duplicate socket at the same
  // descriptor — $activeGatewayProfile still moves to `key`, so request
  // scoping and profile-aware surfaces behave identically.
  if (await sharedPrimaryRoute(key)) {
    applyActive(g.primaryProfile, activationEpoch)

    return
  }

  let entry = g.secondaries.get(key)

  if (!entry) {
    entry = createSecondary(key)
  }

  entry.retained = true
  entry.wantOpen = true
  // Lease the entry against the live-work pruner for the whole dial — the
  // profile-door twin of the agent path's lease above (#89622).
  entry.activationLeaseUntil = Date.now() + ACTIVATION_LEASE_MS

  try {
    if (!isOpen(entry.gateway)) {
      clearTimer(entry)
      entry.reconnectAttempt = 0

      try {
        await openSecondary(entry)
      } catch (error) {
        // #81094: a failed secondary dial must NOT fall through to setActive()
        // with a closed socket — that silently routes the user's messages to the
        // primary backend (cross-profile session writes). Keep the reconnect
        // schedule (transient failures still self-heal via the backoff below)
        // but RE-THROW so the profile-door caller surfaces the failure and skips
        // the activation. The agent-door twin (ensureGatewayForAgent) keeps its
        // boolean contract and is guarded by the activeGateway() null invariant.
        scheduleReconnect(entry)
        throw error
      }
    }
  } finally {
    // The activation is settling either way — release the prune lease.
    entry.activationLeaseUntil = 0
  }

  // Only publish when the WebSocket actually reached open -- entry.connection
  // is set before the dial completes, so a transient first-dial failure must
  // not count as a successful activation (issue #92265).
  if (
    entry.wantOpen &&
    g.secondaries.get(key) === entry &&
    isOpen(entry.gateway) &&
    applyActive(key, activationEpoch) &&
    entry.connection
  ) {
    publishActiveConnection(entry.connection)
  }
}

// Reconnect the active gateway after a transient request failure. Primary
// reconnects are owned by use-gateway-boot, so we only drive secondaries here.
export async function ensureActiveGatewayOpen(): Promise<HermesGateway | null> {
  if (g.activeKey === g.primaryProfile) {
    return g.primaryGateway
  }

  const entry = g.secondaries.get(g.activeKey)

  if (!entry) {
    return null
  }

  if (!isOpen(entry.gateway)) {
    await reconnectSecondary(entry)
  }

  if (!isOpen(entry.gateway)) {
    // A remote/registry secondary can still be ACTIVATING (backend waking,
    // socket dialing). Failing instantly turned a routine cold start into
    // "Hermes gateway is not connected" on the Sessions `+` action (#88880).
    // Wait a bounded beat for the in-flight activation instead of erroring;
    // a genuinely dead gateway still returns null when the window closes.
    const deadline = Date.now() + ACTIVE_GATEWAY_OPEN_WAIT_MS

    while (Date.now() < deadline && entry.wantOpen && g.secondaries.get(g.activeKey) === entry) {
      if (isOpen(entry.gateway)) {
        break
      }

      await new Promise(resolve => setTimeout(resolve, 250))
    }
  }

  return isOpen(entry.gateway) ? entry.gateway : null
}

// How long ensureActiveGatewayOpen waits out an in-flight secondary
// activation before reporting the gateway as unavailable.
const ACTIVE_GATEWAY_OPEN_WAIT_MS = 8_000

// Recovery signal: nudge every live secondary back open. Power-resume/network
// signals can force sockets that still report open to retire before redialing.
export function reconnectSecondaryGateways({ forceOpenSockets = false }: { forceOpenSockets?: boolean } = {}): void {
  for (const entry of g.secondaries.values()) {
    if (!entry.wantOpen) {
      continue
    }

    if (isOpen(entry.gateway)) {
      if (!forceOpenSockets) {
        continue
      }

      entry.gateway.close()
    }

    entry.reconnectAttempt = 0
    clearTimer(entry)
    void reconnectSecondary(entry)
  }
}

// Keep the idle reaper from killing a backend we still need: ping every live
// secondary. The active one is pinged separately (touchActiveGatewayBackend).
export function touchSecondaryGateways(): void {
  const desktop = window.hermesDesktop

  for (const entry of g.secondaries.values()) {
    if (entry.wantOpen) {
      void desktop?.touchBackend?.(entry.scope).catch(() => undefined)
    }
  }
}

// Tear a secondary down: stop its reconnect loop, detach listeners, close the
// socket. Caller handles removal from the map.
function disposeSecondary(entry: Secondary): void {
  entry.wantOpen = false
  clearTimer(entry)
  entry.offEvent()
  entry.offState()
  entry.gateway.close()
}

// Invariant restore for every eviction path: if the active key names a
// secondary that no longer exists, fall back to the primary EXPLICITLY (atoms
// and composer state follow) instead of leaving a dangling key that
// activeGateway() can no longer resolve. Without this, a soft gateway switch
// (closeSecondaryGateways in use-gateway-boot) left activeKey pointing at an
// evicted registry scope and every call silently hit the primary backend.
function restoreActiveToPrimaryIfEvicted(): void {
  if (g.activeKey !== g.primaryProfile && !g.secondaries.has(g.activeKey)) {
    setActive(g.primaryProfile)
  }
}

// Close + evict secondaries whose scope is neither active nor in `keep`
// (scopes with a running / needs-input session). Bounds cost to live work.
// `keep` carries PROFILE names for local/legacy entries and composite
// registryBackendScopeKey(connectionId, profile) scopes for registry-sourced live
// work. A registry-scoped entry matches ONLY on its composite key: every
// source exposes a 'default' profile, so matching a non-local entry on the
// bare profile name kept gateway B's 'default' socket alive off gateway A's
// 'default' activity (and vice versa) — cross-connection attribution.
//
// Live work is not the only thing worth a socket: an idle tile still holds a
// resumed runtime on its owner's socket, and closing that socket makes the
// backend detach and orphan-reap the runtime, whose `session.reclaimed`
// unbinds the tile and re-resumes it on a fresh socket that the next
// recompute closes again — a spinner loop with no terminal state (#93892).
// Foreground-bound scopes come from the registry's `foregroundScopes` hook
// (foregroundPinned), not from `keep`, so every dispose path sees the same
// pin. `entry.retained` is deliberately NOT consulted here (see the field's
// doc).
export function pruneSecondaryGateways(keep: Set<string>): void {
  const now = Date.now()

  for (const [key, entry] of [...g.secondaries]) {
    if (drainPendingConnectionRedial(entry)) {
      continue
    }

    if (
      key === g.activeKey ||
      keep.has(key) ||
      (!entry.connectionId && keep.has(entry.profile)) ||
      // Bot-relay retention (#93594): the relay pins its remote routes for
      // its whole active lifetime; the live-work pruner must not undo that
      // pin between drain ticks or the socket churn returns.
      relayRetained(entry) ||
      // A mounted tile / the primary thread is bound to a runtime on this
      // socket (#93892) — pinned for as long as that surface is mounted.
      foregroundPinned(entry) ||
      // Mid-dial activation target: the profile being switched TO is not yet
      // active and has no live work, so without this lease any recompute
      // during its cold spawn disposed the entry and the click died silently
      // (#89622). Number guard: dev-HMR entries predate the field. Bounded:
      // an orphaned lease expires on its own.
      (Number.isFinite(entry.activationLeaseUntil) && entry.activationLeaseUntil > now)
    ) {
      continue
    }

    // The route is no longer live work. Release turn leases first so their
    // counted request holds cannot outlive a disposed route or leave a stale
    // release closure attached to a later same-key socket.
    releaseTurnLeasesForScope(key)

    if (g.secondaries.get(key) !== entry) {
      continue
    }

    if (entry.activeRequests > 0) {
      continue
    }

    disposeSecondary(entry)
    g.secondaries.delete(key)
  }

  restoreActiveToPrimaryIfEvicted()
}

function closeSecondariesWhere(shouldClose: (entry: Secondary) => boolean): void {
  for (const [scope, entry] of [...g.secondaries]) {
    if (!shouldClose(entry)) {
      continue
    }

    disposeSecondary(entry)
    g.secondaries.delete(scope)
  }

  restoreActiveToPrimaryIfEvicted()
}

function isLegacySecondary(entry: Secondary): boolean {
  // Every v2 registry route is created with an explicit connection id,
  // including the registry's `local` source. A missing id is reserved for the
  // old profile-only pool; the loose null check also retires HMR entries from
  // builds that predate the field instead of leaving an old legacy socket
  // behind during a mode apply.
  return entry.connectionId == null
}

/**
 * Close only profile sockets that follow the legacy v1 connection config.
 *
 * A global mode apply re-homes the primary backend, but registered connection
 * sockets are independent sources in the v2 registry. Closing every secondary
 * here would detach their sessions and arm `ws_orphan_reap` even though those
 * sources remain valid and reusable. Legacy profile sockets still need to be
 * retired because their endpoint is derived from the v1 config being changed.
 */
export function closeLegacySecondaryGateways(): void {
  closeSecondariesWhere(isLegacySecondary)
}

export function closeSecondaryGateways(): void {
  // Full teardown releases every routed-turn lease (class-2 #94284) and the
  // renderer-generation ledger; the predicate close leaves live sources'
  // leases alone (their sockets stay open).
  for (const timer of g.turnLeaseReleaseTimers.values()) {
    clearTimeout(timer)
  }

  g.turnLeaseReleaseTimers.clear()

  for (const release of [...g.turnLeases.values()]) {
    release()
  }

  g.turnLeases.clear()

  closeSecondariesWhere(() => true)
  openedSecondaryScopes().clear()
}

// A local profile can have two renderer-owned sockets: the legacy bare
// profile scope and the explicit `local` registry scope. Profile deletion
// stops their Electron backend processes, but a retained Secondary otherwise
// sees that shutdown as a transient disconnect and starts its reconnect loop,
// resurrecting the backend that was just deleted. Retire both local scopes
// before the DELETE request while preserving same-named agents on remote,
// cloud, or SSH connections.
export function retireLocalProfileGateways(profile: string): void {
  const name = String(profile || '').trim()

  if (!name) {
    return
  }

  const key = normKey(name)
  const scopes = new Set([key, registryBackendScopeKey('local', key)])
  let activeInvalidated = false

  for (const scope of scopes) {
    const entry = g.secondaries.get(scope)

    if (!entry) {
      continue
    }

    activeInvalidated ||= scope === g.activeKey
    disposeSecondary(entry)
    g.secondaries.delete(scope)
  }

  restoreActiveToPrimaryIfEvicted()

  if (activeInvalidated) {
    g.config?.onActiveConnectionInvalidated?.(g.primaryProfile, gatewayActivationEpoch())
  }
}

// Registry lifecycle: a connection was removed or materially edited. Removal
// disposes every scoped secondary immediately (a removed remote/cloud source
// has no local process to die, so otherwise its WebSocket streams ghost
// events). A material edit redials each profile through the normal open path so
// fresh sockets target the NEW endpoint, but request/relay leases and mounted
// foreground runtimes keep their old socket until they drain; the active scope
// re-activates when its replacement is safe to publish.
export function disposeSecondariesForConnection(connectionId: string, opts: { redial?: boolean } = {}): void {
  const id = String(connectionId || '').trim()
  let activeInvalidated = false

  if (!id) {
    return
  }

  for (const [key, entry] of [...g.secondaries]) {
    if (entry.connectionId !== id) {
      continue
    }

    const wasActive = key === g.activeKey
    activeInvalidated ||= wasActive

    if (opts.redial && (entry.activeRequests > 0 || relayRetained(entry) || foregroundPinned(entry))) {
      entry.pendingConnectionRedial = true

      continue
    }

    disposeSecondary(entry)
    g.secondaries.delete(key)

    if (opts.redial) {
      const reopen = wasActive
        ? ensureGatewayForAgent(entry.connectionId, entry.profile)
        : openGatewayForAgent(entry.connectionId, entry.profile)

      void reopen.catch(() => undefined)
    }
  }

  if (activeInvalidated && !opts.redial) {
    setActive(g.primaryProfile)
    g.config?.onActiveConnectionInvalidated?.(g.primaryProfile, gatewayActivationEpoch())
  }
}

// Self-accept so editing this module (or a fan-out that lands here) is an
// in-place hot update instead of a full page reload — the live sockets in `g`
// survive the swap. Dev-only: production strips import.meta.hot.
if (import.meta.hot) {
  import.meta.hot.accept()
}
