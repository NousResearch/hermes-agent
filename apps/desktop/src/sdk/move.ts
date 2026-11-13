/**
 * The Move-to-Hermes-Cloud door: the one place a plugin may call the `/api/move/*` routes on a
 * named connection (the local backend that builds the bundle, a Hermes Cloud agent that receives
 * it), plus the Hermes Cloud discovery the Settings panel already uses. Narrow on purpose: the
 * generic plugin REST door stays namespace-scoped, and this one reaches exactly one route family.
 */

import type { DesktopCloudAgent, DesktopCloudDiscoverResult, DesktopCloudStatus } from '@/global'
import { hermesApi } from '@/hermes'

const MOVE_PATH = /^\/api\/move\/(bundle|uploads)(\/[A-Za-z0-9._-]+(\/(chunk|finish|apply|undo))?)?(\?.*)?$/

export interface MoveRequest {
  /** `'local'` for this computer's backend, else a registry connection id. */
  connectionId: string
  path: string
  method?: 'DELETE' | 'GET' | 'POST'
  body?: unknown
  /** Remote profile on that connection (default: its default profile). */
  profile?: string
  timeoutMs?: number
}

function bridge() {
  const desktop = typeof window === 'undefined' ? undefined : window.hermesDesktop

  if (!desktop?.api) {
    throw new Error('Hermes desktop bridge unavailable')
  }

  return desktop
}

export const move = {
  request: async <T>({ connectionId, path, method = 'GET', body, profile, timeoutMs }: MoveRequest): Promise<T> => {
    if (!MOVE_PATH.test(path)) {
      throw new Error(`move.request: ${path} is not a move route`)
    }

    bridge()

    return hermesApi<T>({
      body,
      connectionId,
      method,
      path,
      priority: 'foreground',
      timeoutMs: timeoutMs ?? 120_000,
      ...(profile ? { profile } : {})
    })
  },

  cloudStatus: async (): Promise<DesktopCloudStatus> => {
    const cloud = bridge().cloud

    if (!cloud) {
      throw new Error('This Desktop build cannot reach Hermes Cloud. Update Hermes Desktop.')
    }

    return cloud.status()
  },

  cloudLogin: async (): Promise<DesktopCloudStatus> => bridge().cloud.login(),

  discoverCloudAgents: async (org?: string): Promise<DesktopCloudDiscoverResult> => bridge().cloud.discover(org),

  /** Sign in to one cloud agent silently through the portal session and save it as a gateway, so
   *  `request` can reach it. Returns the saved connection id. */
  connectCloudAgent: async (agent: DesktopCloudAgent, org?: string): Promise<string> => {
    const desktop = bridge()

    if (!agent.dashboardUrl || !desktop.connections) {
      throw new Error('This agent is still being set up. Try again in a minute.')
    }

    const url = agent.dashboardUrl.replace(/\/+$/, '')
    const registry = await desktop.connections.list()
    const same = (candidate?: string) => (candidate ?? '').replace(/\/+$/, '').toLowerCase() === url.toLowerCase()
    const saved = registry.connections.find(connection => connection.kind === 'cloud' && same(connection.url))

    const signIn = await desktop.cloud.agentSignIn(url)

    if (!signIn.connected) {
      throw new Error('Sign-in to this agent did not finish. Sign in to Hermes Cloud again.')
    }

    if (saved) {
      return saved.id
    }

    const result = await desktop.connections.save({
      authMode: 'oauth',
      kind: 'cloud',
      label: agent.name,
      url,
      ...(org ? { org } : {})
    })

    return result.connection.id
  }
}
