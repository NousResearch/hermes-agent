import {
  type ConnectionState,
  type GatewayEvent,
  isGatewayReauthRequired,
  registryBackendScopeKey,
  resolveGatewayWsUrl
} from '@hermes/shared'
import { atom } from 'nanostores'

import type { HermesConnection } from '@/global'
import { HermesGateway, setApiRequestConnection } from '@/hermes'
import { reconnectBackoffDelayMs } from '@/lib/reconnect-backoff'
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
  activeRequests: number
  commandLeases: number
  offEvent: () => void
  offState: () => void
  openPromise: Promise<void> | null
  reconnectTimer: ReturnType<typeof setTimeout> | null
  reconnectAttempt: number
  reconnecting: Promise<void> | null
  /** True when a foreground/prewarmed consumer owns this entry beyond one RPC. */
  retained: boolean
  prunePending: boolean
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
  primaryProfile: string
  primaryGeneration: number
  /** Monotonic ownership token for async foreground activation intent. */
  activationGeneration: number
  activeKey: string
  activationEpoch: number
  secondaries: Map<string, Secondary>
  keptProfiles: Set<string>
  primaryReconnect: {
    gateway: HermesGateway
    generation: number
    profile: string
    promise: Promise<GatewayConnection | null>
  } | null
  $gateway: ReturnType<typeof atom<HermesGateway | null>>
  $activeProfile: ReturnType<typeof atom<string>>
}

const STATE_KEY = Symbol.for('hermes.desktop.gatewayRegistryState')

function createRegistryState(): GatewayRegistryState {
  return {
    config: null,
    primaryGateway: null,
    primaryProfile: 'default',
    primaryGeneration: 0,
    activationGeneration: 0,
    activeKey: 'default',
    activationEpoch: 0,
    secondaries: new Map<string, Secondary>(),
    keptProfiles: new Set<string>(),
    primaryReconnect: null,
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

    return store[STATE_KEY]
  }

  return createRegistryState()
}

const g = gatewayState()

// Keep HMR-surviving entries compatible when this module's registry shape grows.
g.primaryReconnect ??= null
g.primaryGeneration ??= 0
g.activationGeneration ??= 0
g.keptProfiles ??= new Set<string>()

for (const entry of g.secondaries.values()) {
  entry.commandLeases ??= 0
  entry.openPromise ??= null
  entry.prunePending ??= false

  if (typeof entry.reconnecting === 'boolean') {
    entry.reconnecting = null
  }
}

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
  const key = normKey(profile)

  if (g.primaryGateway !== gateway || g.primaryProfile !== key) {
    g.primaryGeneration += 1
    g.primaryReconnect = null
  }

  g.primaryGateway = gateway
  g.primaryProfile = key
}

/**
 * Invalidate work pinned to the primary's current physical connection before a
 * soft close/re-home reuses the same gateway object and profile. Object identity
 * alone cannot distinguish the old runtime namespace from the replacement.
 */
export function invalidatePrimaryGatewayGeneration(gateway: HermesGateway): void {
  if (g.primaryGateway !== gateway) {
    return
  }

  g.primaryGeneration += 1
  g.primaryReconnect = null
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
 * at — null for the local/legacy primary path and for profile-keyed (local)
 * secondaries. Event consumers pair this with the event's own `connectionId`
 * tag so "from the active profile" really means "from the active SOURCE":
 * two connected gateways can both expose a 'default' profile, and a bare
 * profile comparison attributed gateway B's 'default' activity to gateway A.
 */
export function activeGatewayConnectionId(): null | string {
  if (g.activeKey === g.primaryProfile) {
    return null
  }

  return g.secondaries.get(g.activeKey)?.connectionId ?? null
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

interface GatewayActivationClaim {
  generation: number
  routeEpoch: number
}

function claimActivation(): GatewayActivationClaim {
  g.activationGeneration += 1

  return {
    generation: g.activationGeneration,
    // Preserve current-main's eager route epoch: a registry edit can capture
    // and invalidate this intent while its descriptor/socket dial is pending.
    routeEpoch: beginGatewayActivation()
  }
}

function setActiveIfOwned(profile: string, activation: GatewayActivationClaim): boolean {
  // Opening/reconnecting the physical transport is still useful after the user
  // selects another agent, but its late completion no longer owns the global UI
  // pointer. Request leases (including captured slash/branch work) stay pinned
  // to their concrete transport and are intentionally independent of this token.
  if (g.activationGeneration !== activation.generation) {
    return false
  }

  return applyActive(profile, activation.routeEpoch)
}

type SecondaryLeaseKind = 'command' | 'request'

function secondaryIsKept(entry: Secondary): boolean {
  return g.keptProfiles.has(entry.scope) || (!entry.connectionId && g.keptProfiles.has(entry.profile))
}

function releaseSecondaryLease(entry: Secondary, kind: SecondaryLeaseKind): void {
  if (kind === 'command') {
    entry.commandLeases = Math.max(0, entry.commandLeases - 1)
  } else {
    entry.activeRequests = Math.max(0, entry.activeRequests - 1)
  }

  if (entry.commandLeases > 0 || entry.activeRequests > 0 || !entry.prunePending) {
    return
  }

  // Consume the deferred decision once when the final owner releases. A newer
  // activation/keep decision cancels that old prune; an unregistered or already
  // disposed entry must not be closed again by its stale async completion.
  entry.prunePending = false

  if (
    !entry.wantOpen ||
    g.secondaries.get(entry.scope) !== entry ||
    entry.scope === g.activeKey ||
    secondaryIsKept(entry)
  ) {
    return
  }

  disposeSecondary(entry)
  g.secondaries.delete(entry.scope)
}

function clearTimer(entry: Secondary): void {
  if (entry.reconnectTimer !== null) {
    clearTimeout(entry.reconnectTimer)
    entry.reconnectTimer = null
  }
}

async function openSecondary(entry: Secondary): Promise<void> {
  if (entry.openPromise) {
    return entry.openPromise
  }

  const attempt = (async () => {
    const desktop = window.hermesDesktop

    if (!desktop) {
      return
    }

    // Registry-scoped entries dial through getConnectionFor when the bridge has
    // it (feature-detected: an older Electron main lacks the door and those
    // entries simply can't exist yet — createSecondary guards creation).
    const conn =
      entry.connectionId && desktop.getConnectionFor
        ? await desktop.getConnectionFor({ connectionId: entry.connectionId, profile: entry.profile })
        : await desktop.getConnection(entry.profile)

    const wsDeps =
      entry.connectionId && desktop.getGatewayWsUrlFor
        ? {
            getGatewayWsUrl: () =>
              desktop.getGatewayWsUrlFor!({ connectionId: entry.connectionId, profile: entry.profile })
          }
        : entry.connectionId
          ? {}
          : desktop

    const wsUrl = await resolveGatewayWsUrl(wsDeps, conn)

    // Profile switches and teardown can run while descriptor/ticket lookup
    // awaits. Never resurrect an entry that was already disposed or replaced.
    if (!entry.wantOpen || g.secondaries.get(entry.scope) !== entry) {
      return
    }

    entry.connection = conn
    await entry.gateway.connect(wsUrl)

    // Teardown can also win while connect() itself is pending. Close the exact
    // orphaned socket rather than letting its late open transition resurrect a
    // removed registry entry.
    if (!entry.wantOpen || g.secondaries.get(entry.scope) !== entry) {
      entry.gateway.close()

      return
    }

    if (g.activeKey === entry.scope) {
      publishActiveConnection(conn)
    }

    void desktop.touchBackend?.(entry.scope).catch(() => undefined)
  })()

  entry.openPromise = attempt

  try {
    await attempt
  } finally {
    if (entry.openPromise === attempt) {
      entry.openPromise = null
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
  if (!entry.wantOpen || isOpen(entry.gateway)) {
    return
  }

  if (entry.reconnecting) {
    return entry.reconnecting
  }

  const reconnecting = (async () => {
    try {
      await openSecondary(entry)
        void import('@/store/session-states')
          .then(({ reconcileBusyStatesOnReconnect }) => reconcileBusyStatesOnReconnect(entry.scope))
          .catch(() => undefined)
      } catch (error) {
        // The registry no longer knows this connection (removed while we were
        // backing off), or Electron's deletion guard reports the profile itself
        // gone/mid-delete. Both are permanent for this scoped socket — retrying
        // forever can never succeed and hammers the spawn guard every backoff
        // tick (#88769). Fail-stop: dispose the entry and evict it instead of an
        // infinite 15s-cap retry loop.
        if ((entry.connectionId && isMissingConnectionError(error)) || isMissingProfileError(error)) {
          disposeSecondary(entry)

          if (g.secondaries.get(entry.scope) === entry) {
            g.secondaries.delete(entry.scope)
          }

          restoreActiveToPrimaryIfEvicted()

          return
        }
        // Other transport failure → fall through to the backoff below.
      }
    })()

    entry.reconnecting = reconnecting

    try {
      await reconnecting
    } finally {
    if (entry.reconnecting === reconnecting) {
      entry.reconnecting = null
    }

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
    activeRequests: 0,
    commandLeases: 0,
    offEvent: () => {},
    offState: () => {},
    openPromise: null,
    reconnectTimer: null,
    reconnectAttempt: 0,
    reconnecting: null,
    retained: false,
    prunePending: false,
    wantOpen: true,
    activationLeaseUntil: 0
  }

  // Events keep carrying the bare profile — session routing is profile-keyed
  // everywhere. connectionId rides along for surfaces that need the source.
  entry.offEvent = gateway.onEvent(event =>
    g.config?.onEvent({ ...event, profile, ...(connectionId ? { connectionId } : {}) })
  )
  entry.offState = gateway.onState(state => {
    reportGatewayState(scope, state)

    if (state === 'open') {
      entry.reconnectAttempt = 0
      clearTimer(entry)
    } else if ((state === 'closed' || state === 'error') && entry.wantOpen) {
      scheduleReconnect(entry)
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
    const conn = await desktop.getConnection(profile)

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
      releaseSecondaryLease(entry, 'request')

      if (
        g.secondaries.get(entry.scope) === entry &&
        entry.activeRequests === 0 &&
        !entry.retained &&
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
  params: Record<string, unknown> = {}
): Promise<T> {
  const key = normKey(profile)
  const scope = registryBackendScopeKey(connectionId, key)

  if (scope === key) {
    return requestGatewayForProfile<T>(key, method, params)
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

    return await entry.gateway.request<T>(method, params)
  } finally {
    releaseSecondaryLease(entry, 'request')

    if (
      g.secondaries.get(entry.scope) === entry &&
      entry.activeRequests === 0 &&
      !entry.retained &&
      g.activeKey !== entry.scope
    ) {
      disposeSecondary(entry)

      if (g.secondaries.get(entry.scope) === entry) {
        g.secondaries.delete(entry.scope)
      }
    }
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

export async function openGatewayForAgent(connectionId: null | string, profile: string): Promise<void> {
  const scope = registryBackendScopeKey(connectionId, profile)

  if (scope === normKey(profile)) {
    return openGatewayForProfile(profile)
  }

  if (!window.hermesDesktop?.getConnectionFor) {
    throw new Error('This Desktop build cannot dial registry connections. Update Hermes Desktop.')
  }

  const entry = g.secondaries.get(scope) ?? createSecondary(profile, connectionId)
  entry.retained = true
  entry.wantOpen = true

  if (!isOpen(entry.gateway)) {
    await openSecondary(entry)
  }
}

async function activateGatewayForAgent(
  connectionId: null | string,
  profile: string,
  activation: GatewayActivationClaim
): Promise<boolean> {
  const scope = registryBackendScopeKey(connectionId, profile)

  if (scope === normKey(profile)) {
    return activateGatewayForProfile(profile, activation)
  }

  if (!window.hermesDesktop?.getConnectionFor) {
    throw new Error('This Desktop build cannot dial registry connections. Update Hermes Desktop.')
  }

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
  // Activation owns this exact scoped entry across descriptor/ticket lookup.
  // A concurrent live-session prune must not dispose it and let the eventual
  // completion activate a replacement or dangling scope.
  entry.commandLeases += 1

  try {
    if (!isOpen(entry.gateway)) {
      clearTimer(entry)
      entry.reconnectAttempt = 0

      try {
        await openSecondary(entry)
      } catch {
        scheduleReconnect(entry)
      }
    }

    // A source edit/remove may dispose this entry while its dial is still in
    // flight. Only the still-registered, still-owned activation may publish.
    const activated =
      entry.wantOpen &&
      g.secondaries.get(scope) === entry &&
      Boolean(entry.connection) &&
      setActiveIfOwned(scope, activation)

    if (activated && entry.connection) {
      publishActiveConnection(entry.connection)
    }

    return activated
  } finally {
    entry.activationLeaseUntil = 0
    releaseSecondaryLease(entry, 'command')
  }
}

export function ensureGatewayForAgent(connectionId: null | string, profile: string): Promise<boolean> {
  return activateGatewayForAgent(connectionId, profile, claimActivation())
}

// Make `profile` the active gateway, lazily opening its socket if needed. The
// primary is a no-op fast path. Background sockets are never closed here.
async function activateGatewayForProfile(profile: string, activation: GatewayActivationClaim): Promise<boolean> {
  const key = normKey(profile)

  if (key === g.primaryProfile) {
    return setActiveIfOwned(key, activation)
  }

  let entry = g.secondaries.get(key)

  if (!entry) {
    // Global-remote share (routing case 3): one remote host serves every
    // profile through the PRIMARY socket, scoped per request. Only classify
    // when there is no already-owned pooled entry: awaiting this lookup before
    // claiming an existing prewarm lets pruning dispose that entry mid-activate.
    if (await sharedPrimaryRoute(key)) {
      return setActiveIfOwned(g.primaryProfile, activation)
    }

    entry = g.secondaries.get(key) ?? createSecondary(key)
  }

  entry.retained = true
  entry.wantOpen = true
  // Lease the entry against the live-work pruner for the whole dial — the
  // profile-door twin of the agent path's lease above (#89622).
  entry.activationLeaseUntil = Date.now() + ACTIVATION_LEASE_MS
  // Activation is an authoritative claim, not speculative pre-warm. Claim the
  // shared entry before joining its openPromise so pruning cannot dispose it in
  // the await window and leave activeKey pointing at a removed transport.
  entry.commandLeases += 1

  try {
    if (!isOpen(entry.gateway)) {
      clearTimer(entry)
      entry.reconnectAttempt = 0

      try {
        await openSecondary(entry)
      } catch {
        scheduleReconnect(entry)
      }
    }

    const activated =
      entry.wantOpen &&
      g.secondaries.get(key) === entry &&
      Boolean(entry.connection) &&
      setActiveIfOwned(key, activation)

    if (activated && entry.connection) {
      publishActiveConnection(entry.connection)
    }

    return activated
  } finally {
    // The activation is settling either way — release both prune guards.
    entry.activationLeaseUntil = 0
    releaseSecondaryLease(entry, 'command')
  }
}

export async function ensureGatewayForProfile(profile: string): Promise<void> {
  await activateGatewayForProfile(profile, claimActivation())
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

export type GatewayRequester = <T>(
  method: string,
  params?: Record<string, unknown>,
  timeoutMs?: number,
  signal?: AbortSignal
) => Promise<T>

export interface GatewayRequestLease {
  request: GatewayRequester
  release(): void
}

interface LeasedGatewayOwner {
  gateway: HermesGateway
  generation: number
  profile: string
  secondary: Secondary | null
}

function ownerStillRegistered(owner: LeasedGatewayOwner): boolean {
  return owner.secondary
    ? owner.secondary.commandLeases > 0 &&
        owner.secondary.wantOpen &&
        g.secondaries.get(owner.secondary.scope) === owner.secondary &&
        owner.secondary.gateway === owner.gateway
    : g.primaryGateway === owner.gateway &&
        g.primaryProfile === owner.profile &&
        g.primaryGeneration === owner.generation
}

type GatewayConnection = Awaited<ReturnType<NonNullable<typeof window.hermesDesktop>['getConnection']>>

async function reconnectPrimary(owner: LeasedGatewayOwner): Promise<GatewayConnection | null> {
  const current = g.primaryReconnect

  if (
    current?.gateway === owner.gateway &&
    current.profile === owner.profile &&
    current.generation === owner.generation
  ) {
    return current.promise
  }

  const attempt = (async () => {
    const desktop = window.hermesDesktop

    if (!desktop || !ownerStillRegistered(owner)) {
      return null
    }

    // All primary reconnect consumers (boot, ordinary requests, and pinned
    // commands) share this cohort. Revalidation is cheap/no-op for local
    // gateways and prevents one caller from redialing a stale remote descriptor.
    await desktop.revalidateConnection?.().catch(() => undefined)
    const conn = await desktop.getConnection(owner.profile)
    const wsUrl = await resolveGatewayWsUrl(desktop, conn)

    if (!ownerStillRegistered(owner)) {
      return null
    }

    await owner.gateway.connect(wsUrl)

    return ownerStillRegistered(owner) && isOpen(owner.gateway) ? conn : null
  })()

  const reconnect = {
    gateway: owner.gateway,
    generation: owner.generation,
    profile: owner.profile,
    promise: attempt
  }

  g.primaryReconnect = reconnect

  try {
    return await attempt
  } finally {
    if (g.primaryReconnect === reconnect) {
      g.primaryReconnect = null
    }
  }
}

/** Reopen the registered primary through the same owner/generation cohort used by leases. */
export function reconnectPrimaryGateway(gateway: HermesGateway): Promise<GatewayConnection | null> {
  if (g.primaryGateway !== gateway) {
    return Promise.resolve(null)
  }

  return reconnectPrimary({
    gateway,
    generation: g.primaryGeneration,
    profile: g.primaryProfile,
    secondary: null
  })
}

async function recoverLeasedGateway(owner: LeasedGatewayOwner): Promise<HermesGateway | null> {
  if (!ownerStillRegistered(owner)) {
    return null
  }

  if (isOpen(owner.gateway)) {
    return owner.gateway
  }

  if (owner.secondary) {
    clearTimer(owner.secondary)
    owner.secondary.reconnectAttempt = 0
    await reconnectSecondary(owner.secondary)
  } else {
    await reconnectPrimary(owner)
  }

  return ownerStillRegistered(owner) && isOpen(owner.gateway) ? owner.gateway : null
}

/**
 * Pin a concrete profile/socket for one async command. The lease is acquired
 * synchronously, so profile-switch pruning cannot dispose the source transport
 * while an earlier metadata lookup is pending. Requests retain the normal
 * disconnected → exact-owner reconnect → one retry semantics without ever
 * consulting the newly active gateway.
 */
export function acquireGatewayRequestLease(gateway: HermesGateway, profile: string): GatewayRequestLease {
  const key = normKey(profile)

  // A profile can now have multiple registry-scoped sockets. The concrete
  // gateway captured by the invocation is the authority; the bare profile is
  // only validation/metadata and must never redirect the lease to another scope.
  const secondary =
    [...g.secondaries.values()].find(entry => entry.gateway === gateway && normKey(entry.profile) === key) ?? null

  const owner: LeasedGatewayOwner = {
    gateway,
    generation: secondary ? 0 : g.primaryGeneration,
    profile: key,
    secondary
  }

  if (
    (secondary && secondary.gateway !== gateway) ||
    (!secondary && (g.primaryGateway !== gateway || g.primaryProfile !== key))
  ) {
    throw new Error('Hermes source gateway unavailable')
  }

  if (secondary) {
    secondary.commandLeases += 1
  }

  let released = false

  return {
    request: async <T>(method: string, params = {}, timeoutMs?: number, signal?: AbortSignal): Promise<T> => {
      if (released || !ownerStillRegistered(owner)) {
        throw new Error('Hermes source gateway unavailable')
      }

      try {
        return await gateway.request<T>(method, params, timeoutMs, signal)
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)

        // JSON-RPC domain errors can contain phrases such as "model provider not
        // connected" while the WebSocket remains healthy. Replay only when the
        // transport itself authoritatively left the open state.
        if (isOpen(gateway) || !/not connected|connection closed/i.test(message)) {
          throw error
        }

        let recovered: HermesGateway | null

        try {
          recovered = await recoverLeasedGateway(owner)
        } catch (reconnectError) {
          if (isGatewayReauthRequired(reconnectError)) {
            throw reconnectError
          }

          throw error
        }

        if (!recovered || released || !ownerStillRegistered(owner)) {
          throw error
        }

        return recovered.request<T>(method, params, timeoutMs, signal)
      }
    },
    release: () => {
      if (released) {
        return
      }

      released = true

      if (secondary) {
        releaseSecondaryLease(secondary, 'command')
      }
    }
  }
}

// Wake signal (sleep/network/visibility): nudge every live secondary back open.
export function reconnectSecondaryGateways(): void {
  for (const entry of g.secondaries.values()) {
    if (!entry.wantOpen || isOpen(entry.gateway)) {
      continue
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
export function pruneSecondaryGateways(keep: Set<string>): void {
  const now = Date.now()
  g.keptProfiles = new Set([...keep].map(normKey))

  for (const [key, entry] of [...g.secondaries]) {
    if (
      key === g.activeKey ||
      secondaryIsKept(entry) ||
      // Mid-dial activation target: the profile being switched TO is not yet
      // active and has no live work, so without this lease any recompute
      // during its cold spawn disposed the entry and the click died silently
      // (#89622). Number guard: dev-HMR entries predate the field. Bounded:
      // an orphaned lease expires on its own.
      (Number.isFinite(entry.activationLeaseUntil) && entry.activationLeaseUntil > now)
    ) {
      entry.prunePending = false

      continue
    }

    if (entry.commandLeases > 0 || entry.activeRequests > 0) {
      entry.prunePending = true

      continue
    }

    disposeSecondary(entry)
    g.secondaries.delete(key)
  }

  restoreActiveToPrimaryIfEvicted()
}

export function closeSecondaryGateways(): void {
  for (const entry of g.secondaries.values()) {
    disposeSecondary(entry)
  }

  g.secondaries.clear()
  g.keptProfiles.clear()
  restoreActiveToPrimaryIfEvicted()
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

// Registry lifecycle: a connection was removed or materially edited. Dispose
// every secondary scoped to it (a removed remote/cloud source has no local
// process to die, so without this its WebSocket stays open streaming ghost
// events). With `redial` (the edit case) each disposed profile is re-dialed
// through the normal open path so the fresh socket targets the NEW endpoint;
// the active scope re-activates so the foreground keeps painting.
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
