import { type ConnectionState, type GatewayEvent, registryBackendScopeKey, resolveGatewayWsUrl } from '@hermes/shared'
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
  connectPromise: Promise<void> | null
  offEvent: () => void
  offState: () => void
  reconnectTimer: ReturnType<typeof setTimeout> | null
  reconnectAttempt: number
  reconnecting: boolean
  /** True when a foreground/prewarmed consumer owns this entry beyond one RPC. */
  retained: boolean
  // While true the entry auto-reconnects on drop; pruning flips it off so a
  // deliberate close doesn't trigger the backoff loop.
  wantOpen: boolean
}

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
  activeKey: string
  activationEpoch: number
  secondaries: Map<string, Secondary>
  $gateway: ReturnType<typeof atom<HermesGateway | null>>
  $activeProfile: ReturnType<typeof atom<string>>
}

const STATE_KEY = Symbol.for('hermes.desktop.gatewayRegistryState')

function createRegistryState(): GatewayRegistryState {
  return {
    config: null,
    primaryGateway: null,
    primaryProfile: 'default',
    activeKey: 'default',
    activationEpoch: 0,
    secondaries: new Map<string, Secondary>(),
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
  g.primaryGateway = gateway
  g.primaryProfile = normKey(profile)
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
    // Registry-scoped entries dial through getConnectionFor when the bridge has
    // it. Local/legacy entries retain the existing getConnection path.
    const conn =
      entry.connectionId && desktop.getConnectionFor
        ? await desktop.getConnectionFor({ connectionId: entry.connectionId, profile: entry.profile })
        : await desktop.getConnection(entry.profile)

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

    const wsUrl = await resolveGatewayWsUrl(wsDeps, conn)

    await entry.gateway.connect(wsUrl)

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
    activeRequests: 0,
    connectPromise: null,
    offEvent: () => {},
    offState: () => {},
    reconnectTimer: null,
    reconnectAttempt: 0,
    reconnecting: false,
    retained: false,
    wantOpen: true
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
      entry.activeRequests = Math.max(0, entry.activeRequests - 1)

      if (entry.activeRequests === 0 && !entry.retained && g.activeKey !== entry.scope) {
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
  params: Record<string, unknown> = {}
): Promise<T> {
  const route = await gatewayForProfile(profile, true)

  try {
    if (!route.gateway) {
      throw new Error(`Hermes gateway unavailable for profile "${route.key}"`)
    }

    const routedParams = route.scopeProfile ? { ...params, profile: route.key } : params

    return await route.gateway.request<T>(method, routedParams)
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
    entry.activeRequests = Math.max(0, entry.activeRequests - 1)

    if (entry.activeRequests === 0 && !entry.retained && g.activeKey !== entry.scope) {
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

// The agent-scoped analogue of prepareGatewayForProfile, and the same
// publication seam: dial the agent's socket without publishing anything, and
// hand back the synchronous activation thunk. A local/null connection falls
// through to the profile seam, so both doors into an activation share one
// atomicity contract instead of drifting apart.
//
// The thunk reports whether it actually published, preserving the `activated`
// contract callers rely on: a source edit/remove can dispose this entry while
// its dial is in flight, and a caller must be able to tell "switched" from
// "the target stopped existing" rather than assume the former.
export async function prepareGatewayForAgent(connectionId: null | string, profile: string): Promise<() => boolean> {
  const scope = registryBackendScopeKey(connectionId, profile)

  // Genuinely-local scope: the profile door owns this route, so hand back ITS
  // thunk unchanged. Wrapping it to return an unconditional `true` would have
  // reported a rejected activation as a successful one and let the agent
  // caller publish companion state for a switch that never happened.
  if (scope === normKey(profile)) {
    return prepareGatewayForProfile(profile)
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

  if (!isOpen(entry.gateway)) {
    clearTimer(entry)
    entry.reconnectAttempt = 0

    try {
      await openSecondary(entry)
    } catch {
      scheduleReconnect(entry)
    }
  }

  // Bind the entry this dial settled on; see prepareGatewayForProfile.
  const prepared = entry

  return () => {
    // A source edit/remove may dispose this entry while its dial is still in
    // flight. Only the still-registered, still-owned activation may publish.
    const activated =
      prepared.wantOpen &&
      g.secondaries.get(scope) === prepared &&
      Boolean(prepared.connection) &&
      applyActive(scope, activationEpoch)

    if (activated && prepared.connection) {
      publishActiveConnection(prepared.connection)
    }

    return activated
  }
}

export async function ensureGatewayForAgent(connectionId: null | string, profile: string): Promise<boolean> {
  return (await prepareGatewayForAgent(connectionId, profile))()
}

// Open `profile`'s socket if needed and hand back a synchronous activation
// thunk — the publication seam for atomic profile switches. The caller invokes
// the thunk in the same synchronous frame as its own atom writes (profile
// pointer, connection descriptor), so no subscriber can observe the active
// gateway pointing at one backend while companion state still describes
// another. Nothing is published until the thunk runs.
export async function prepareGatewayForProfile(profile: string): Promise<() => boolean> {
  const key = normKey(profile)
  const activationEpoch = beginGatewayActivation()

  if (key === g.primaryProfile) {
    return () => applyActive(key, activationEpoch)
  }

  // Global-remote share (routing case 3): one remote host serves every
  // profile through the PRIMARY socket, scoped per request. Activate the
  // primary instead of dialing a doomed duplicate socket at the same
  // descriptor - $activeGatewayProfile still moves to `key`, so request
  // scoping and profile-aware surfaces behave identically. Checked BEFORE
  // createSecondary so a shared-remote profile never mints a secondary
  // entry, and returned as a thunk like every other path here so this
  // switch publishes as atomically as a dedicated-socket one.
  if (await sharedPrimaryRoute(key)) {
    return () => applyActive(g.primaryProfile, activationEpoch)
  }

  let entry = g.secondaries.get(key)

  if (!entry) {
    entry = createSecondary(key)
  }

  entry.retained = true
  entry.wantOpen = true

  if (!isOpen(entry.gateway)) {
    clearTimer(entry)
    entry.reconnectAttempt = 0

    try {
      await openSecondary(entry)
    } catch {
      scheduleReconnect(entry)
    }
  }

  // Bind the entry the await settled on. `g.secondaries.get(key)` can be a
  // DIFFERENT object by the time the thunk runs (a teardown + redial between
  // prepare and publish), and publishing that one's descriptor would be the
  // very mismatch this seam exists to prevent, so the identity re-check below
  // compares against this exact entry.
  const prepared = entry

  // Reports whether the ACTIVATION was accepted, which is a different question
  // from whether a descriptor was published: an accepted activation with no
  // cached connection still moved the gateway, so the caller must still move
  // its companion state. Only a rejected activation (disposed entry, or an
  // epoch superseded by a newer switch while this one was dialing) must leave
  // every companion store alone.
  return () => {
    const activated = prepared.wantOpen && g.secondaries.get(key) === prepared && applyActive(key, activationEpoch)

    if (activated && prepared.connection) {
      publishActiveConnection(prepared.connection)
    }

    return activated
  }
}

// Make `profile` the active gateway, lazily opening its socket if needed. The
// primary is a no-op fast path. Background sockets are never closed here.
export async function ensureGatewayForProfile(profile: string): Promise<void> {
  ;(await prepareGatewayForProfile(profile))()
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
  for (const [key, entry] of [...g.secondaries]) {
    if (
      key === g.activeKey ||
      keep.has(key) ||
      (!entry.connectionId && keep.has(entry.profile)) ||
      entry.activeRequests > 0
    ) {
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
