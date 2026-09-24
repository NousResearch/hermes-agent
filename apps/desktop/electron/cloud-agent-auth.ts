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
 */
export function createCloudAgentRegistry(io: CloudAgentRegistryIo, normalizeBaseUrl: (url: string) => string) {
  let cache: null | Record<string, string> = null

  const read = (): Record<string, string> => {
    if (cache) {
      return cache
    }

    try {
      const parsed: unknown = JSON.parse(io.readText())

      cache =
        parsed && typeof parsed === 'object' && !Array.isArray(parsed)
          ? Object.fromEntries(
              Object.entries(parsed as Record<string, unknown>).filter(
                (entry): entry is [string, string] => typeof entry[1] === 'string' && entry[1].length > 0
              )
            )
          : {}
    } catch {
      cache = {}
    }

    return cache
  }

  const write = (next: Record<string, string>) => {
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

      return key ? (read()[key] ?? null) : null
    },
    /** Throws on a URL normalizeBaseUrl rejects; callers skip such rows. */
    remember(url: string, agentId: string): void {
      const key = normalizeBaseUrl(url)
      const current = read()

      if (!key || !agentId || current[key] === agentId) {
        return
      }

      write({ ...current, [key]: agentId })
    },
    forget(url: string): void {
      const key = keyFor(url)
      const current = read()

      if (key && key in current) {
        const { [key]: _dropped, ...rest } = current
        write(rest)
      }
    },
    clear(): void {
      write({})
    },
    urls(): string[] {
      return Object.keys(read())
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
}

export const CLOUD_AGENT_NOT_FOUND_MESSAGE =
  'Could not find this agent in your Hermes Cloud account. Refresh the agent list in Settings → Gateway and pick it again.'

export function createCloudAgentAuth(deps: CloudAgentAuthDeps) {
  const log = deps.log ?? (() => undefined)

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

  return { rememberDiscovered, signIn, logout, forgetAgents }
}
