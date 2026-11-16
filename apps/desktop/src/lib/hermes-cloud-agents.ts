/**
 * Hermes Cloud for Settings ▸ Backup & Transfer: the Nous sign-in, the agent list, and a silent
 * sign-in to one agent saved as a gateway connection. All through `window.hermesDesktop.cloud`
 * (native: the portal browser session in Electron; Webapp: `lib/browser-cloud.ts`) and
 * `window.hermesDesktop.connections`, so a test rig stubs those bridge methods.
 */

import { hermesApiAs } from '@/api/client'
import type { DesktopCloudAgent, DesktopCloudDiscoverResult, DesktopCloudStatus } from '@/global'
import type { TransferEndpoint } from '@/lib/agent-transfer'
import { servedByHermesCloud } from '@/lib/hermes-cloud-host'

export const PORTAL_AGENTS_URL = 'https://portal.nousresearch.com/agents'

const BACKUP_POLL_MS = 1_000

export interface BackupResult {
  /** Where the archive sits on the backend that wrote it. */
  archive: string
  /** True once the user saved a copy (a canceled save dialog leaves it on the backend only). */
  saved: boolean
}

function bridge() {
  const desktop = typeof window === 'undefined' ? undefined : window.hermesDesktop

  if (!desktop?.api) {
    throw new Error('Hermes desktop bridge unavailable')
  }

  return desktop
}

/** Transfer is offered everywhere except a Webapp Hermes Cloud itself serves: that tab IS a cloud
 *  agent. */
export const transferAvailable = (): boolean => !servedByHermesCloud()

/** `hermes backup` of the whole Hermes home on `owner`, then a native save of the archive. */
export async function backup(owner: TransferEndpoint, onLine?: (lines: string[]) => void): Promise<BackupResult> {
  const started = await hermesApiAs<{ archive?: string; name: string }>(owner, {
    body: {},
    method: 'POST',
    path: '/api/ops/backup'
  })

  if (!started.archive) {
    throw new Error('The backup did not say where it was written.')
  }

  for (;;) {
    await new Promise(resolve => setTimeout(resolve, BACKUP_POLL_MS))

    const status = await hermesApiAs<{ exit_code: null | number; lines: string[]; running: boolean }>(owner, {
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

  const suggestedName = started.archive.split(/[\\/]/).pop()
  const result = await save({ ...owner, path: started.archive, suggestedName })

  return { archive: started.archive, saved: result.saved }
}

export const cloudStatus = (): Promise<DesktopCloudStatus> => bridge().cloud.status()

export const cloudLogin = (): Promise<DesktopCloudStatus> => bridge().cloud.login()

export const discoverCloudAgents = (org?: string): Promise<DesktopCloudDiscoverResult> => bridge().cloud.discover(org)

const trimUrl = (url?: string) => (url ?? '').replace(/\/+$/, '').toLowerCase()

/** Sign in to one cloud agent silently through the portal session and save it as a gateway, so
 *  REST can reach it. Returns the saved connection id. */
export async function connectCloudAgent(agent: DesktopCloudAgent, org?: string): Promise<string> {
  const desktop = bridge()

  if (!agent.dashboardUrl || !desktop.connections) {
    throw new Error('This agent is still being set up. Try again in a minute.')
  }

  const url = agent.dashboardUrl.replace(/\/+$/, '')
  const registry = await desktop.connections.list()
  const saved = registry.connections.find(c => c.kind === 'cloud' && trimUrl(c.url) === trimUrl(url))
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
