/**
 * cloud-agent-auth.ts
 *
 * Per-agent bearer lifecycle for Hermes Cloud connections. A cloud agent's
 * dashboard token comes from the portal token exchange (§5) and is stored as
 * that connection's native bearer, so every existing bearer transport (REST,
 * `POST /api/auth/ws-ticket`, downloads, media) is reused unchanged.
 *
 * Trust model: portal `/api/agents` discovery is the only authority for
 * "this URL is a Hermes Cloud agent" and for the exchange audience. Each
 * discovery is reconciled into the persisted REGISTRY as an authoritative
 * snapshot (dashboard URL → { agentId, confirmedAt }). A registry row is a
 * routing hint only; an exchange needs a binding confirmed within
 * CLOUD_BINDING_MAX_AGE_MS, or a live discovery run now (confirmedAgentIdFor).
 * Neither a renderer-supplied agent id nor any field of a stored token set
 * (which a remote gateway can write) decides whether — or for which agent —
 * the desktop exchanges the user's portal session. The coordinator's refresh
 * strategy consults the same registry (native-token-coordinator-deps.ts).
 */

import { cloudDiscoveryUnavailableError, isCloudLoginRequired } from './cloud-auth-errors'
import type { NativeTokenSet } from './native-oauth'
import { CLOUD_AGENT_TOKEN_PROVIDER } from './portal-oauth'

const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '[::1]'])

/**
 * Hermes Cloud dashboards are https. Plain http is allowed only on loopback,
 * for the local portal stand-in used in development and tests.
 */
export function cloudDashboardUrlAllowed(rawUrl: string): boolean {
  let parsed: URL

  try {
    parsed = new URL(String(rawUrl || '').trim())
  } catch {
    return false
  }

  return parsed.protocol === 'https:' || (parsed.protocol === 'http:' && LOOPBACK_HOSTS.has(parsed.hostname))
}

export interface CloudAgentRegistryIo {
  /** Throws when the file is absent — treated as empty. */
  readText: () => string
  writeText: (text: string) => void
}

/** One portal-sourced binding: the agent id and when discovery last confirmed it (epoch ms; 0 = never). */
export interface CloudAgentBinding {
  agentId: string
  confirmedAt: number
}

interface CloudAgentRegistryState {
  orgId: null | string
  agents: Record<string, CloudAgentBinding>
}

const CLOUD_AGENT_REGISTRY_VERSION = 2

/**
 * dashboardUrl → { agentId, confirmedAt }, persisted. Not a secret (the id is
 * the public audience). Written only from portal discovery snapshots.
 *
 * A row is a ROUTING hint ("this URL is a Hermes Cloud agent, so it renews by
 * re-exchange"); it never authorizes an exchange on its own. The exchange
 * audience must come from a binding confirmed by a recent portal discovery
 * (see createCloudAgentAuth().confirmedAgentIdFor).
 *
 * The file also records the org (`org_id` of the desktop token, non-secret)
 * the entries were discovered under, so a sign-in to another org — even
 * after a sign-out left no previous token to compare — drops them. On disk:
 * `{ "version": 2, "orgId": string | null, "agents": { url: { agentId, confirmedAt } } }`.
 * Earlier builds wrote `{ orgId, agents: { url: id } }` or a bare `{ url: id }`
 * map (org unknown); both read as UNCONFIRMED rows (confirmedAt 0).
 */
export function createCloudAgentRegistry(io: CloudAgentRegistryIo, normalizeBaseUrl: (url: string) => string) {
  let cache: CloudAgentRegistryState | null = null

  const readBindings = (value: unknown): Record<string, CloudAgentBinding> => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return {}
    }

    const rows: Record<string, CloudAgentBinding> = {}

    for (const [url, row] of Object.entries(value as Record<string, unknown>)) {
      if (typeof row === 'string' && row) {
        // Pre-versioned formats: never confirmed.
        rows[url] = { agentId: row, confirmedAt: 0 }
      } else if (row && typeof row === 'object') {
        const { agentId, confirmedAt } = row as { agentId?: unknown; confirmedAt?: unknown }

        if (typeof agentId === 'string' && agentId) {
          rows[url] = {
            agentId,
            confirmedAt: typeof confirmedAt === 'number' && Number.isFinite(confirmedAt) ? confirmedAt : 0
          }
        }
      }
    }

    return rows
  }

  const read = (): CloudAgentRegistryState => {
    if (cache) {
      return cache
    }

    try {
      const parsed: unknown = JSON.parse(io.readText())
      const shaped = parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as any) : null

      if (shaped && 'agents' in shaped) {
        const versioned = shaped.version === CLOUD_AGENT_REGISTRY_VERSION
        const agents = readBindings(shaped.agents)

        cache = {
          orgId: typeof shaped.orgId === 'string' && shaped.orgId ? shaped.orgId : null,
          agents: versioned
            ? agents
            : Object.fromEntries(Object.entries(agents).map(([url, row]) => [url, { ...row, confirmedAt: 0 }]))
        }
      } else {
        cache = {
          orgId: null,
          agents: Object.fromEntries(
            Object.entries(readBindings(shaped)).map(([url, row]) => [url, { ...row, confirmedAt: 0 }])
          )
        }
      }
    } catch {
      cache = { orgId: null, agents: {} }
    }

    return cache
  }

  const write = (next: CloudAgentRegistryState) => {
    cache = next

    try {
      io.writeText(JSON.stringify({ version: CLOUD_AGENT_REGISTRY_VERSION, ...next }))
    } catch {
      // Best-effort: the in-memory map still serves this run.
    }
  }

  const keyFor = (url: string): null | string => {
    try {
      return normalizeBaseUrl(url) || null
    } catch {
      return null
    }
  }

  const bindingFor = (url: string): CloudAgentBinding | null => {
    const key = keyFor(url)
    const row = key ? read().agents[key] : undefined

    return row ? { ...row } : null
  }

  return {
    /** Routing hint only: the last agent id discovery recorded for this URL. */
    agentIdFor(url: string): null | string {
      return bindingFor(url)?.agentId ?? null
    },
    bindingFor,
    /**
     * Replace every row with an authoritative discovery snapshot (keys are
     * already normalized by the caller). Returns the URLs that were dropped
     * or re-bound to another agent.
     */
    replaceAll(agents: Record<string, CloudAgentBinding>): string[] {
      const current = read()

      const changed = Object.keys(current.agents).filter(
        url => !(url in agents) || agents[url]!.agentId !== current.agents[url]!.agentId
      )

      write({ ...current, agents: { ...agents } })

      return changed
    },
    forget(url: string): void {
      const key = keyFor(url)
      const current = read()

      if (key && key in current.agents) {
        const { [key]: _dropped, ...rest } = current.agents
        write({ ...current, agents: rest })
      }
    },
    clear(): void {
      write({ ...read(), agents: {} })
    },
    urls(): string[] {
      return Object.keys(read().agents)
    },
    /** The org the entries were discovered under; null = unknown. */
    orgId(): null | string {
      return read().orgId
    },
    /** Record the org; the entries are the caller's to clear first. */
    setOrgId(orgId: null | string): void {
      const current = read()

      if (current.orgId !== orgId) {
        write({ ...current, orgId })
      }
    }
  }
}

export type CloudAgentRegistry = ReturnType<typeof createCloudAgentRegistry>

export interface CloudAgentAuthDeps {
  normalizeBaseUrl: (url: string) => string
  registry: CloudAgentRegistry
  /** portal-session exchangeForAgent (§5). */
  exchangeForAgent: (agentId: string) => Promise<NativeTokenSet>
  /** Explicit mutations go through the native coordinator (epoch-fenced). */
  storeAgentTokens: (baseUrl: string, tokens: NativeTokenSet) => void
  clearAgentTokens: (baseUrl: string) => void
  /** Every key in the native token store (for sign-out cleanup). */
  listStoredTokenUrls: () => string[]
  loadStoredTokens: (baseUrl: string) => NativeTokenSet | null
  clearPortalSession: () => void
  /** Live portal discovery: the only source of a confirmed binding. */
  discoverAgents?: () => Promise<Array<{ id: string; dashboardUrl: null | string }>>
  /** Diagnostics; never passed a token or an agent id. */
  log?: (line: string) => void
  /** Wall clock (epoch ms): binding confirmation stamps and the discovery throttle. */
  nowMs?: () => number
}

/**
 * A binding confirmed by portal discovery within this window may authorize
 * an exchange without another round trip; anything older (or never
 * confirmed, e.g. migrated from an earlier build) needs a live discovery.
 */
export const CLOUD_BINDING_MAX_AGE_MS = 5 * 60_000

/**
 * Discoveries run by confirmedAgentIdFor happen at most this often (explicit
 * sign-ins excepted). Every row a discovery returns is stamped then, so a
 * throttled caller either has a fresh binding or none — never a stale one.
 */
export const CLOUD_REDISCOVERY_MIN_INTERVAL_MS = 60_000

export interface DiscoveryTicket {
  seq: number
  orgAtStart: null | string
  startedAt: number
}

export const CLOUD_AGENT_NOT_FOUND_MESSAGE =
  'Could not find this agent in your Hermes Cloud account. Refresh the agent list in Settings → Gateway and pick it again.'

export function createCloudAgentAuth(deps: CloudAgentAuthDeps) {
  const log = deps.log ?? (() => undefined)
  const nowMs = deps.nowMs ?? (() => Date.now())
  // The rediscovery throttle runs on a monotonic clock so a wall-clock
  // rollback cannot suppress discovery; binding ages stay on the wall clock
  // because they are persisted across restarts.
  const monotonicMs = deps.nowMs ? nowMs : () => performance.now()
  let discovery: null | Promise<void> = null
  let lastDiscoveryAt = Number.NEGATIVE_INFINITY
  // Discoveries can overlap (Settings refresh vs. a background confirm), and
  // responses can land out of order. Each discovery takes a sequence number
  // when it STARTS; a result older than the last one reconciled is dropped,
  // so a slow stale snapshot can never overwrite a newer one.
  let discoverySeq = 0
  let lastReconciledSeq = 0

  /**
   * Start a discovery: records its order, the org it runs under and its start
   * time (rows are stamped confirmed at the START, never later than the
   * portal actually answered for).
   */
  function beginDiscovery(): DiscoveryTicket {
    return { seq: ++discoverySeq, orgAtStart: deps.registry.orgId(), startedAt: nowMs() }
  }

  const keyFor = (url: string): null | string => {
    try {
      return deps.normalizeBaseUrl(url) || null
    } catch {
      return null
    }
  }

  const hasStoredTokens = (url: string): boolean => {
    try {
      return deps.loadStoredTokens(url) !== null
    } catch {
      // Unreadable: clear it rather than guess.
      return true
    }
  }

  /**
   * Reconcile a portal discovery result as an authoritative SNAPSHOT: the
   * registry becomes exactly these rows (anything the portal no longer lists
   * is removed, and its agent bearer cleared), each stamped confirmed now.
   *
   *   - `ticket` comes from beginDiscovery() when the request STARTED: if a
   *     sign-in to another org landed meanwhile, or a discovery that started
   *     later was already reconciled, the whole result is dropped; rows are
   *     stamped confirmed at the start time;
   *   - a malformed / non-https row is skipped on its own;
   *   - two or more rows that normalize to the same URL fail closed: that URL
   *     is left out entirely, so no exchange can pick either agent for it.
   */
  function reconcileDiscovered(
    agents: Array<{ id: string; dashboardUrl: null | string }>,
    ticket: DiscoveryTicket = beginDiscovery()
  ): void {
    if (deps.registry.orgId() !== ticket.orgAtStart) {
      log('[cloud] dropped a discovery result that started under a different org')

      return
    }

    if (ticket.seq < lastReconciledSeq) {
      log('[cloud] dropped a discovery result older than one already applied')

      return
    }

    lastReconciledSeq = ticket.seq
    const confirmedAt = ticket.startedAt
    const byUrl = new Map<string, string[]>()

    for (const agent of Array.isArray(agents) ? agents : []) {
      if (!agent?.id || typeof agent.id !== 'string' || !agent.dashboardUrl) {
        continue
      }

      if (!cloudDashboardUrlAllowed(agent.dashboardUrl)) {
        log('[cloud] skipped a discovered agent: dashboard URL is not https')

        continue
      }

      const key = keyFor(agent.dashboardUrl)

      if (!key) {
        log('[cloud] skipped a discovered agent: dashboard URL is not valid')

        continue
      }

      byUrl.set(key, [...(byUrl.get(key) ?? []), agent.id])
    }

    const snapshot: Record<string, CloudAgentBinding> = {}

    for (const [url, ids] of byUrl) {
      if (ids.length > 1) {
        log(`[cloud] ${ids.length} discovered agents share one dashboard URL; refusing to bind it`)

        continue
      }

      snapshot[url] = { agentId: ids[0]!, confirmedAt }
    }

    // Dropped or re-bound URLs: a bearer minted for the old binding must not
    // keep flowing to a URL the portal no longer vouches for. The clear is
    // epoch-fenced, so a refresh already in flight for such a URL cannot
    // store a bearer for the old binding after it (it fails "auth changed";
    // the retry exchanges from the fresh binding). URLs with nothing stored
    // are left alone so their in-flight bootstrap is not disturbed.
    for (const url of deps.registry.replaceAll(snapshot)) {
      if (hasStoredTokens(url)) {
        deps.clearAgentTokens(url)
      }
    }
  }

  /**
   * One live portal discovery, shared by concurrent callers. Unless
   * `explicit`, at most one per CLOUD_REDISCOVERY_MIN_INTERVAL_MS (a
   * throttled call resolves false without a round trip). Rejects with a
   * transient cloudDiscoveryUnavailable error when the portal call fails.
   */
  async function runDiscovery(explicit: boolean): Promise<boolean> {
    if (!deps.discoverAgents) {
      return false
    }

    if (!discovery) {
      if (!explicit && monotonicMs() - lastDiscoveryAt < CLOUD_REDISCOVERY_MIN_INTERVAL_MS) {
        return false
      }

      lastDiscoveryAt = monotonicMs()
      const discoverAgents = deps.discoverAgents
      const ticket = beginDiscovery()

      discovery = (async () => {
        try {
          reconcileDiscovered(await discoverAgents(), ticket)
        } catch (error) {
          log(`[cloud] agent discovery failed: ${error instanceof Error ? error.message : String(error)}`)

          throw cloudDiscoveryUnavailableError(error)
        } finally {
          discovery = null
        }
      })()
    }

    await discovery

    return true
  }

  /** Whether the latest reconciled snapshot still binds `url` to `agentId`. */
  function isBindingCurrent(url: string, agentId: string): boolean {
    return deps.registry.bindingFor(url)?.agentId === agentId
  }

  function freshAgentIdFor(url: string): null | string {
    const binding = deps.registry.bindingFor(url)

    if (!binding || binding.confirmedAt <= 0) {
      return null
    }

    const age = nowMs() - binding.confirmedAt

    return age >= 0 && age < CLOUD_BINDING_MAX_AGE_MS ? binding.agentId : null
  }

  /**
   * The ONLY source of an exchange audience. Returns the agent id for this
   * dashboard URL when portal discovery confirmed it within
   * CLOUD_BINDING_MAX_AGE_MS; otherwise runs one live discovery (shared,
   * throttled), reconciles it, and returns the fresh binding — or null when
   * the portal does not list the URL (or lists it ambiguously), or when the
   * throttle blocks and nothing fresh is on record. Fails closed: a
   * discovery failure rejects with a transient cloudDiscoveryUnavailable
   * error and nothing is exchanged.
   */
  async function confirmedAgentIdFor(dashboardUrl: string, options: { explicit?: boolean } = {}) {
    if (!cloudDashboardUrlAllowed(dashboardUrl)) {
      return null
    }

    const fresh = freshAgentIdFor(dashboardUrl)

    if (fresh) {
      return fresh
    }

    if (!(await runDiscovery(Boolean(options.explicit)))) {
      return null
    }

    return freshAgentIdFor(dashboardUrl)
  }

  /**
   * A sign-in succeeded with a desktop token pinned to `orgId`. When the
   * registry was populated under another org (or an unknown one), every
   * agent bearer and registry entry belongs to that org: drop them, so none
   * of them is ever exchanged with the new session.
   */
  function adoptSessionOrg(orgId: null | string): void {
    const previous = deps.registry.orgId()

    if (previous !== orgId && (previous !== null || deps.registry.urls().length > 0 || agentTokenUrls().length > 0)) {
      log("[cloud] Hermes Cloud session is pinned to a different org; dropping the previous org's agents")
      forgetAgents()
    }

    deps.registry.setOrgId(orgId)
  }

  /**
   * Silent per-agent sign-in: exchange the desktop token for this agent's
   * bearer and store it as the connection's native token set. Refuses any
   * URL that is not an https dashboard a recent portal discovery confirmed.
   * A renderer agent-id hint is never an audience: a hint that disagrees with
   * the confirmed binding forces one live discovery, whose answer wins.
   */
  async function signIn(dashboardUrl: string, agentIdHint?: null | string) {
    if (!cloudDashboardUrlAllowed(dashboardUrl)) {
      throw new Error('Hermes Cloud agents are only reachable over https.')
    }

    const baseUrl = deps.normalizeBaseUrl(dashboardUrl)
    let agentId: null | string

    try {
      agentId = freshAgentIdFor(baseUrl)

      if (!agentId || (agentIdHint && agentIdHint !== agentId)) {
        await runDiscovery(true)
        agentId = freshAgentIdFor(baseUrl)
      }
    } catch (error) {
      // An explicit sign-in with no portal session should prompt a sign-in.
      const cause = (error as { cause?: unknown })?.cause

      throw isCloudLoginRequired(cause) ? cause : error
    }

    if (!agentId) {
      throw new Error(CLOUD_AGENT_NOT_FOUND_MESSAGE)
    }

    const tokens = await deps.exchangeForAgent(agentId)

    // A discovery that landed while the exchange was in flight may have
    // re-bound or dropped this URL: never store a bearer for a binding the
    // latest snapshot no longer vouches for.
    if (!isBindingCurrent(baseUrl, agentId)) {
      throw new Error(CLOUD_AGENT_NOT_FOUND_MESSAGE)
    }

    deps.storeAgentTokens(baseUrl, tokens)

    return { baseUrl, connected: true }
  }

  /**
   * Every URL that may hold a cloud agent bearer: registry entries plus any
   * stored set minted by the exchange (the registry write is best-effort).
   * The provider marker is used for cleanup only, never for trust.
   */
  function agentTokenUrls(): string[] {
    const urls = new Set(deps.registry.urls())

    for (const url of deps.listStoredTokenUrls()) {
      try {
        if (deps.loadStoredTokens(url)?.provider === CLOUD_AGENT_TOKEN_PROVIDER) {
          urls.add(url)
        }
      } catch {
        // An unreadable entry cannot be used either.
      }
    }

    return [...urls]
  }

  function clearAllAgentTokens(): void {
    for (const url of agentTokenUrls()) {
      deps.clearAgentTokens(url)
    }
  }

  /**
   * Sign out of Hermes Cloud: the portal session and every derived agent
   * bearer. The registry stays, so saved connections re-exchange after the
   * next sign-in to the same org.
   */
  function logout(): void {
    deps.clearPortalSession()
    clearAllAgentTokens()
  }

  /**
   * The portal session moved to another org: every agent bearer and registry
   * entry belongs to the old org. Drop them so a saved connection re-resolves
   * through discovery instead of surfacing "You no longer have access".
   */
  function forgetAgents(): void {
    clearAllAgentTokens()
    deps.registry.clear()
  }

  return {
    beginDiscovery,
    reconcileDiscovered,
    confirmedAgentIdFor,
    isBindingCurrent,
    adoptSessionOrg,
    signIn,
    logout,
    forgetAgents
  }
}
