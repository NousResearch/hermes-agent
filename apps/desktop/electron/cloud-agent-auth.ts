/**
 * cloud-agent-auth.ts
 *
 * Per-agent bearer lifecycle for Hermes Cloud connections. A cloud agent's
 * dashboard token comes from the portal token exchange (§5) and is stored as
 * that connection's native bearer, so every existing bearer transport (REST,
 * `POST /api/auth/ws-ticket`, downloads, media) is reused unchanged.
 *
 * Trust model: the cloud agent REGISTRY (dashboard URL → agent id) is the
 * only authority for "this URL is a Hermes Cloud agent" and for the exchange
 * audience. It is written solely from portal `/api/agents` discovery results.
 * Neither a renderer-supplied agent id nor any field of a stored token set
 * (which a remote gateway can write) decides whether — or for which agent —
 * the desktop exchanges the user's portal session. The coordinator's refresh
 * strategy consults the same registry (native-token-coordinator-deps.ts).
 */

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

/**
 * dashboardUrl → AgentInstance id, persisted. Not a secret (the id is the
 * public audience). Written only from portal discovery; it lets reconnect
 * after a restart (or a sign-out/sign-in) re-exchange without a round trip.
 *
 * The file also records the org (`org_id` of the desktop token, non-secret)
 * the entries were discovered under, so a sign-in to another org — even
 * after a sign-out left no previous token to compare — drops them. On disk:
 * `{ "orgId": string | null, "agents": { url: id } }`; a bare `{ url: id }`
 * map (earlier builds) reads as org unknown.
 */
export function createCloudAgentRegistry(io: CloudAgentRegistryIo, normalizeBaseUrl: (url: string) => string) {
  let cache: null | { orgId: null | string; agents: Record<string, string> } = null

  const onlyStringEntries = (value: unknown): Record<string, string> =>
    value && typeof value === 'object' && !Array.isArray(value)
      ? Object.fromEntries(
          Object.entries(value as Record<string, unknown>).filter(
            (entry): entry is [string, string] => typeof entry[1] === 'string' && entry[1].length > 0
          )
        )
      : {}

  const read = () => {
    if (cache) {
      return cache
    }

    try {
      const parsed: unknown = JSON.parse(io.readText())
      const shaped = parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as any) : null

      if (shaped && 'agents' in shaped) {
        cache = {
          orgId: typeof shaped.orgId === 'string' && shaped.orgId ? shaped.orgId : null,
          agents: onlyStringEntries(shaped.agents)
        }
      } else {
        cache = { orgId: null, agents: onlyStringEntries(shaped) }
      }
    } catch {
      cache = { orgId: null, agents: {} }
    }

    return cache
  }

  const write = (next: { orgId: null | string; agents: Record<string, string> }) => {
    cache = next

    try {
      io.writeText(JSON.stringify(next))
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

  return {
    agentIdFor(url: string): null | string {
      const key = keyFor(url)

      return key ? (read().agents[key] ?? null) : null
    },
    /** Throws on a URL normalizeBaseUrl rejects; callers skip such rows. */
    remember(url: string, agentId: string): void {
      const key = normalizeBaseUrl(url)
      const current = read()

      if (!key || !agentId || current.agents[key] === agentId) {
        return
      }

      write({ ...current, agents: { ...current.agents, [key]: agentId } })
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
  /** Live portal discovery, for a URL the registry does not (yet) know. */
  discoverAgents?: () => Promise<Array<{ id: string; dashboardUrl: null | string }>>
  /** Diagnostics; never passed a token. */
  log?: (line: string) => void
  /** Clock for the rediscovery throttle (ms). */
  nowMs?: () => number
}

/** Background rediscovery (bootstrap of a saved connection) runs at most this often. */
export const CLOUD_REDISCOVERY_MIN_INTERVAL_MS = 60_000

export const CLOUD_AGENT_NOT_FOUND_MESSAGE =
  'Could not find this agent in your Hermes Cloud account. Refresh the agent list in Settings → Gateway and pick it again.'

export function createCloudAgentAuth(deps: CloudAgentAuthDeps) {
  const log = deps.log ?? (() => undefined)
  const nowMs = deps.nowMs ?? (() => Date.now())
  let rediscovery: null | Promise<void> = null
  let lastRediscoveryAt = Number.NEGATIVE_INFINITY

  /** Record portal discovery rows; a malformed row is skipped on its own. */
  function rememberDiscovered(agents: Array<{ id: string; dashboardUrl: null | string }>): void {
    for (const agent of agents) {
      if (!agent?.id || !agent.dashboardUrl) {
        continue
      }

      if (!cloudDashboardUrlAllowed(agent.dashboardUrl)) {
        log(`[cloud] skipped agent ${agent.id}: dashboard URL is not https`)

        continue
      }

      try {
        deps.registry.remember(agent.dashboardUrl, agent.id)
      } catch {
        log(`[cloud] skipped agent ${agent.id}: dashboard URL is not valid`)
      }
    }
  }

  /**
   * The agent id for a dashboard URL, from the registry only. A renderer
   * hint is accepted only when it matches; a missing or disagreeing entry
   * earns one live discovery, after which the portal's answer wins.
   */
  async function resolveAgentId(baseUrl: string, hint: null | string): Promise<string> {
    const known = deps.registry.agentIdFor(baseUrl)

    if (known && (!hint || hint === known)) {
      return known
    }

    if (deps.discoverAgents) {
      rememberDiscovered(await deps.discoverAgents())
      const discovered = deps.registry.agentIdFor(baseUrl)

      if (discovered) {
        return discovered
      }
    }

    throw new Error(CLOUD_AGENT_NOT_FOUND_MESSAGE)
  }

  /**
   * Background fallback for a saved cloud connection the registry does not
   * know (e.g. after org A→B→A cleared it): ONE live portal discovery,
   * shared by concurrent callers and throttled across calls, then the
   * registry — i.e. the portal's answer — decides. A discovery failure is
   * logged and reads as "unknown" (null), never thrown.
   */
  async function rediscoverAgentId(dashboardUrl: string): Promise<null | string> {
    const known = deps.registry.agentIdFor(dashboardUrl)

    if (known || !deps.discoverAgents || !cloudDashboardUrlAllowed(dashboardUrl)) {
      return known
    }

    if (!rediscovery) {
      if (nowMs() - lastRediscoveryAt < CLOUD_REDISCOVERY_MIN_INTERVAL_MS) {
        return null
      }

      lastRediscoveryAt = nowMs()
      const discoverAgents = deps.discoverAgents

      rediscovery = (async () => {
        try {
          rememberDiscovered(await discoverAgents())
        } catch (error) {
          log(`[cloud] background agent discovery failed: ${error instanceof Error ? error.message : String(error)}`)
        } finally {
          rediscovery = null
        }
      })()
    }

    await rediscovery

    return deps.registry.agentIdFor(dashboardUrl)
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
   * URL that is not an https dashboard portal discovery returned.
   */
  async function signIn(dashboardUrl: string, agentIdHint?: null | string) {
    if (!cloudDashboardUrlAllowed(dashboardUrl)) {
      throw new Error('Hermes Cloud agents are only reachable over https.')
    }

    const baseUrl = deps.normalizeBaseUrl(dashboardUrl)
    const agentId = await resolveAgentId(baseUrl, agentIdHint || null)
    const tokens = await deps.exchangeForAgent(agentId)

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

  return { rememberDiscovered, rediscoverAgentId, adoptSessionOrg, signIn, logout, forgetAgents }
}
