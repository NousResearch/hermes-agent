/**
 * cloud-agent-auth.ts
 *
 * Per-agent bearer lifecycle for Hermes Cloud connections. A cloud agent's
 * dashboard token comes from the portal token exchange (§5) and is stored as
 * that connection's native bearer, so every existing bearer transport (REST,
 * `POST /api/auth/ws-ticket`, downloads, media) is reused unchanged.
 *
 * The exchange issues no refresh token, so "refresh" for a cloud connection is
 * a RE-EXCHANGE (fresh portal access token → exchange again), never the
 * gateway's `/auth/native/refresh`. createNativeTokenRefresher() is the single
 * per-connection strategy switch the native access-token coordinator calls.
 *
 * The agent id (the exchange audience) is recoverable after a restart from:
 * the token set itself (`userId` slot), the persisted dashboardUrl → agent id
 * registry, and finally one live discovery.
 */

import { isCloudAgentAccessLost, isCloudLoginRequired } from './cloud-auth-errors'
import type { NativeTokenSet } from './native-oauth'
import { CLOUD_AGENT_TOKEN_PROVIDER } from './portal-oauth'

export function isCloudAgentTokenSet(tokens: NativeTokenSet | null | undefined): boolean {
  return Boolean(tokens && tokens.provider === CLOUD_AGENT_TOKEN_PROVIDER)
}

/** Coordinator `canRefresh`: cloud agent bearers renew by re-exchange. */
export function canRefreshNativeTokenSet(tokens: NativeTokenSet): boolean {
  return isCloudAgentTokenSet(tokens) || Boolean(tokens.refreshToken)
}

/**
 * Auth verdicts from a re-exchange that mean "this connection is signed out"
 * (as opposed to a transient failure that should keep the token and retry).
 */
export function isCloudAuthLoss(error: unknown): boolean {
  return isCloudLoginRequired(error) || isCloudAgentAccessLost(error)
}

/** The per-connection refresh strategy: cloud → re-exchange; others unchanged. */
export function createNativeTokenRefresher(deps: {
  refreshGatewayTokens: (baseUrl: string, tokens: NativeTokenSet) => Promise<NativeTokenSet>
  reexchangeCloudAgent: (baseUrl: string, tokens: NativeTokenSet) => Promise<NativeTokenSet>
}) {
  return (baseUrl: string, tokens: NativeTokenSet): Promise<NativeTokenSet> =>
    isCloudAgentTokenSet(tokens)
      ? deps.reexchangeCloudAgent(baseUrl, tokens)
      : deps.refreshGatewayTokens(baseUrl, tokens)
}

export interface CloudAgentRegistryIo {
  /** Throws when the file is absent — treated as empty. */
  readText: () => string
  writeText: (text: string) => void
}

/**
 * dashboardUrl → AgentInstance id, persisted. Not a secret (the id is the
 * public audience); it exists so reconnect / re-sign-in after a restart does
 * not need a discovery round trip first.
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

  return {
    agentIdFor(url: string): null | string {
      return read()[normalizeBaseUrl(url)] ?? null
    },
    remember(url: string, agentId: string): void {
      const key = normalizeBaseUrl(url)
      const current = read()

      if (!key || !agentId || current[key] === agentId) {
        return
      }

      cache = { ...current, [key]: agentId }

      try {
        io.writeText(JSON.stringify(cache))
      } catch {
        // Best-effort: the in-memory map still serves this run, and the token
        // set carries the id too.
      }
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
  clearPortalSession: () => void
  /** Last-resort agent-id lookup for a URL nothing remembers. */
  discoverAgents?: () => Promise<Array<{ id: string; dashboardUrl: null | string }>>
}

export function createCloudAgentAuth(deps: CloudAgentAuthDeps) {
  function rememberDiscovered(agents: Array<{ id: string; dashboardUrl: null | string }>): void {
    for (const agent of agents) {
      if (agent?.id && agent.dashboardUrl) {
        deps.registry.remember(agent.dashboardUrl, agent.id)
      }
    }
  }

  async function resolveAgentId(baseUrl: string, hint?: null | string): Promise<string> {
    if (hint) {
      return hint
    }

    const known = deps.registry.agentIdFor(baseUrl)

    if (known) {
      return known
    }

    if (deps.discoverAgents) {
      rememberDiscovered(await deps.discoverAgents())
      const discovered = deps.registry.agentIdFor(baseUrl)

      if (discovered) {
        return discovered
      }
    }

    throw new Error(
      'Could not find this agent in your Hermes Cloud account. Refresh the agent list in Settings → Gateway and pick it again.'
    )
  }

  /**
   * Silent per-agent sign-in: exchange the desktop token for this agent's
   * bearer and store it as the connection's native token set.
   */
  async function signIn(dashboardUrl: string, agentIdHint?: null | string) {
    const baseUrl = deps.normalizeBaseUrl(dashboardUrl)
    const agentId = await resolveAgentId(baseUrl, agentIdHint)
    const tokens = await deps.exchangeForAgent(agentId)

    deps.registry.remember(baseUrl, agentId)
    deps.storeAgentTokens(baseUrl, tokens)

    return { baseUrl, connected: true }
  }

  /** Coordinator refresh for a cloud connection. The coordinator stores it. */
  async function reexchange(baseUrl: string, tokens: NativeTokenSet): Promise<NativeTokenSet> {
    return deps.exchangeForAgent(await resolveAgentId(baseUrl, tokens.userId || null))
  }

  /** Sign out of Hermes Cloud: the portal session and every derived agent bearer. */
  function logout(): void {
    deps.clearPortalSession()

    for (const url of deps.registry.urls()) {
      deps.clearAgentTokens(url)
    }
  }

  return { rememberDiscovered, signIn, reexchange, logout }
}
