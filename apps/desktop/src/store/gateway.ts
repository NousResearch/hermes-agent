import { backendScopeKey, type ConnectionState, type GatewayEvent, resolveGatewayWsUrl } from '@hermes/shared'
import { atom } from 'nanostores'

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
}

// ── Secondary (pool) backends ──────────────────────────────────────────────
interface Secondary {
  /** Scope key from backendScopeKey(connectionId, profile). */
  scope: string
  profile: string
  /** Registry connection serving this socket; null = the local/legacy path. */
  connectionId: null | string
  gateway: HermesGateway
  offEvent: () => void
  offState: () => void
  reconnectTimer: ReturnType<typeof setTimeout> | null
  reconnectAttempt: number
  reconnecting: boolean
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
  secondaries: Map<string, Secondary>
  $gateway: ReturnType<typeof atom<HermesGateway | null>>
}

const STATE_KEY = Symbol.for('hermes.desktop.gatewayRegistryState')

function createRegistryState(): GatewayRegistryState {
  return {
    config: null,
    primaryGateway: null,
    primaryProfile: 'default',
    activeKey: 'default',
    secondaries: new Map<string, Secondary>(),
    // The active gateway instance, exposed for inline message-stream
    // components (inline ClarifyTool, model overlays) that call gateway
    // methods without the instance threaded down through props.
    $gateway: atom<HermesGateway | null>(null)
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
  g.activeKey = normKey(profile)
  const gateway = activeGateway()
  g.$gateway.set(gateway)
  setGatewayState(gateway?.connectionState ?? 'closed')
  // Push the active scope's registry connection into the hermes module (null
  // for the local pool) so connection-building WS calls (pluginSocket) resolve
  // through the same source of truth every activation path maintains here —
  // registry-agent activations included, not just profile switches.
  setApiRequestConnection(activeGatewayConnectionId())
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

  // Registry-scoped entries dial through getConnectionFor when the bridge has
  // it (feature-detected: an older Electron main lacks the door and those
  // entries simply can't exist yet — createSecondary guards creation).
  const conn =
    entry.connectionId && desktop.getConnectionFor
      ? await desktop.getConnectionFor({ connectionId: entry.connectionId, profile: entry.profile })
      : await desktop.getConnection(entry.profile)

  const wsUrl = await resolveGatewayWsUrl(
    entry.connectionId && desktop.getGatewayWsUrlFor
      ? {
          getGatewayWsUrl: () =>
            desktop.getGatewayWsUrlFor!({ connectionId: entry.connectionId, profile: entry.profile })
        }
      : desktop,
    conn
  )

  await entry.gateway.connect(wsUrl)

  if (g.activeKey === entry.scope) {
    setConnection(conn)
  }

  void desktop.touchBackend?.(entry.scope).catch(() => undefined)
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
    // backing off). Retrying forever can never succeed — fail-stop: dispose
    // the entry and evict it instead of an infinite 15s-cap retry loop.
    if (entry.connectionId && isMissingConnectionError(error)) {
      entry.reconnecting = false
      disposeSecondary(entry)
      g.secondaries.delete(entry.scope)

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

function createSecondary(profile: string, connectionId: null | string = null): Secondary {
  const gateway = new HermesGateway()
  const scope = backendScopeKey(connectionId, profile)

  const entry: Secondary = {
    scope,
    profile,
    connectionId,
    gateway,
    offEvent: () => {},
    offState: () => {},
    reconnectTimer: null,
    reconnectAttempt: 0,
    reconnecting: false,
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

// Open `profile`'s socket WITHOUT making it active — the hover-intent pre-warm
// (store/profile). Runs the same spawn + connect chain as a real switch, so by
// click time ensureGatewayForProfile finds an open socket and just activates
// it. No scheduleReconnect on failure: a hover is speculative, so a dead
// backend must not start a background retry loop — the real switch owns retry
// and error UX. An already-open (or primary) profile is a no-op.
export async function openGatewayForProfile(profile: string): Promise<void> {
  const key = normKey(profile)

  if (key === g.primaryProfile) {
    return
  }

  if (await sharedPrimaryRoute(key)) {
    // Served by the primary backend — there is no per-profile socket to warm.
    return
  }

  const entry = g.secondaries.get(key) ?? createSecondary(key)
  entry.wantOpen = true

  if (!isOpen(entry.gateway)) {
    await openSecondary(entry)
  }
}

// ── Connection-scoped agents (multi-source roster) ─────────────────────────
// The (connectionId, profile) analogues of the profile functions above. A
// null/'local' connectionId falls straight through to the profile path, so
// callers can pass roster rows verbatim without special-casing the local
// source. Feature-detected: without the Electron getConnectionFor door these
// throw, and roster surfaces disable non-local rows instead.

export async function openGatewayForAgent(connectionId: null | string, profile: string): Promise<void> {
  const scope = backendScopeKey(connectionId, profile)

  if (scope === normKey(profile)) {
    return openGatewayForProfile(profile)
  }

  if (!window.hermesDesktop?.getConnectionFor) {
    throw new Error('This Desktop build cannot dial registry connections. Update Hermes Desktop.')
  }

  const entry = g.secondaries.get(scope) ?? createSecondary(profile, connectionId)
  entry.wantOpen = true

  if (!isOpen(entry.gateway)) {
    await openSecondary(entry)
  }
}

export async function ensureGatewayForAgent(connectionId: null | string, profile: string): Promise<void> {
  const scope = backendScopeKey(connectionId, profile)

  if (scope === normKey(profile)) {
    return ensureGatewayForProfile(profile)
  }

  if (!window.hermesDesktop?.getConnectionFor) {
    throw new Error('This Desktop build cannot dial registry connections. Update Hermes Desktop.')
  }

  let entry = g.secondaries.get(scope)

  if (!entry) {
    entry = createSecondary(profile, connectionId)
  }

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

  setActive(scope)
}

// Make `profile` the active gateway, lazily opening its socket if needed. The
// primary is a no-op fast path. Background sockets are never closed here.
export async function ensureGatewayForProfile(profile: string): Promise<void> {
  const key = normKey(profile)

  if (key === g.primaryProfile) {
    setActive(key)

    return
  }

  // Global-remote share (routing case 3): one remote host serves every
  // profile through the PRIMARY socket, scoped per request. Activate the
  // primary instead of dialing a doomed duplicate socket at the same
  // descriptor — $activeGatewayProfile still moves to `key`, so request
  // scoping and profile-aware surfaces behave identically.
  if (await sharedPrimaryRoute(key)) {
    setActive(g.primaryProfile)

    return
  }

  let entry = g.secondaries.get(key)

  if (!entry) {
    entry = createSecondary(key)
  }

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

  setActive(key)
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

  return isOpen(entry.gateway) ? entry.gateway : null
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
// backendScopeKey(connectionId, profile) scopes for registry-sourced live
// work. A registry-scoped entry matches ONLY on its composite key: every
// source exposes a 'default' profile, so matching a non-local entry on the
// bare profile name kept gateway B's 'default' socket alive off gateway A's
// 'default' activity (and vice versa) — cross-connection attribution.
export function pruneSecondaryGateways(keep: Set<string>): void {
  for (const [key, entry] of [...g.secondaries]) {
    if (key === g.activeKey || keep.has(key) || (!entry.connectionId && keep.has(entry.profile))) {
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

// Registry lifecycle: a connection was removed or materially edited. Dispose
// every secondary scoped to it (a removed remote/cloud source has no local
// process to die, so without this its WebSocket stays open streaming ghost
// events). With `redial` (the edit case) each disposed profile is re-dialed
// through the normal open path so the fresh socket targets the NEW endpoint;
// the active scope re-activates so the foreground keeps painting.
export function disposeSecondariesForConnection(connectionId: string, opts: { redial?: boolean } = {}): void {
  const id = String(connectionId || '').trim()

  if (!id) {
    return
  }

  for (const [key, entry] of [...g.secondaries]) {
    if (entry.connectionId !== id) {
      continue
    }

    const wasActive = key === g.activeKey

    disposeSecondary(entry)
    g.secondaries.delete(key)

    if (opts.redial) {
      const reopen = wasActive
        ? ensureGatewayForAgent(entry.connectionId, entry.profile)
        : openGatewayForAgent(entry.connectionId, entry.profile)

      void reopen.catch(() => undefined)
    }
  }
}

// Self-accept so editing this module (or a fan-out that lands here) is an
// in-place hot update instead of a full page reload — the live sockets in `g`
// survive the swap. Dev-only: production strips import.meta.hot.
if (import.meta.hot) {
  import.meta.hot.accept()
}
