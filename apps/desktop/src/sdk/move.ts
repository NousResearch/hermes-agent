/**
 * The Move-to-Hermes-Cloud door: the one place a plugin may call the `/api/move/*` routes on a
 * named connection (the local backend that builds the bundle, a Hermes Cloud agent that receives
 * it), plus the Hermes Cloud discovery the Settings panel already uses. Narrow on purpose: the
 * generic plugin REST door stays namespace-scoped, and this one reaches exactly one route family.
 */

import type { DesktopCloudAgent, DesktopCloudDiscoverResult, DesktopCloudStatus } from '@/global'
import { hermesApi } from '@/hermes'
import { servedByHermesCloud } from '@/lib/hermes-cloud-host'

const MOVE_PATH = /^\/api\/move\/(bundle|uploads)(\/[A-Za-z0-9._-]+(\/(chunk|finish|apply|undo))?)?(\?.*)?$/
const BACKUP_POLL_MS = 1_000

export interface BackupResult {
  /** Where the archive sits on the backend that wrote it. */
  archive: string
  /** True once the user saved a copy (a canceled save dialog leaves it on the backend only). */
  saved: boolean
}

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
  /** Moving this agent into a Hermes Cloud agent is offered everywhere except a Webapp Hermes
   *  Cloud itself serves (the same rule that hides the updater there): that tab IS a cloud agent. */
  transferAvailable: (): boolean => !servedByHermesCloud(),

  /** `hermes backup` of the whole Hermes home (every profile; credentials included, so the user
   *  keeps the file private), then a save of the archive to this computer. `onLine` gets the
   *  command's output as it runs. */
  backup: async (onLine?: (lines: string[]) => void): Promise<BackupResult> => {
    const started = await hermesApi<{ archive?: string; name: string }>({
      body: {},
      method: 'POST',
      path: '/api/ops/backup'
    })

    if (!started.archive) {
      throw new Error('The backup did not say where it was written.')
    }

    for (;;) {
      await new Promise(resolve => setTimeout(resolve, BACKUP_POLL_MS))

      const status = await hermesApi<{ exit_code: null | number; lines: string[]; running: boolean }>({
        path: `/api/actions/${encodeURIComponent(started.name)}/status?lines=40`
      })

      onLine?.(status.lines)

      if (!status.running) {
        if (status.exit_code !== 0) {
          throw new Error(status.lines.slice(-3).join('\n') || 'The backup failed.')
        }

        break
      }
    }

    const save = bridge().saveGatewayFile

    if (!save) {
      return { archive: started.archive, saved: false }
    }

    const result = await save({ path: started.archive, suggestedName: started.archive.split(/[\\/]/).pop() })

    return { archive: started.archive, saved: result.saved }
  },

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
