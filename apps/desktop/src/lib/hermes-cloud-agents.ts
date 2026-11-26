/**
 * Hermes Cloud for Settings ▸ Backup & Transfer: the Nous sign-in, the agent list, and a silent
 * sign-in to one agent saved as a gateway connection. All through `window.hermesDesktop.cloud`
 * (native: the portal browser session in Electron; Webapp: `lib/browser-cloud.ts`) and
 * `window.hermesDesktop.connections`, so a test rig stubs those bridge methods.
 */

import { hermesApiAs } from '@/api/client'
import type { DesktopCloudAgent, DesktopCloudDiscoverResult, DesktopCloudStatus } from '@/global'
import { runBackup, type TransferEndpoint, type TransferRequestFn } from '@/lib/agent-transfer'

export const PORTAL_AGENTS_URL = 'https://portal.nousresearch.com/agents'

export const transferRequest: TransferRequestFn = (owner, req) => hermesApiAs(owner, req)

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

/** `hermes backup` of the whole Hermes home on `owner`, then a save of the archive. */
export async function backup(owner: TransferEndpoint, onLine?: (line: string) => void): Promise<BackupResult> {
  const archive = await runBackup(transferRequest, owner, onLine)
  const save = bridge().saveGatewayFile

  if (!save) {
    return { archive, saved: false }
  }

  // The backup route serves the archive from the host that wrote it; /api/fs/download would read
  // the profile's SSH workspace instead.
  const result = await save({
    ...owner,
    downloadPath: `/api/ops/backup/download?archive=${encodeURIComponent(archive)}`,
    path: archive,
    suggestedName: archive.split(/[\\/]/).pop()
  })

  return { archive, saved: result.saved }
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
  const signIn = await desktop.cloud.agentSignIn(url)

  if (!signIn.connected) {
    throw new Error('Sign-in to this agent did not finish. Sign in to Hermes Cloud again.')
  }

  if (signIn.connectionId) {
    return signIn.connectionId
  }

  // Read after the sign-in: on the Webapp the sign-in itself saves the connection.
  const registry = await desktop.connections.list()
  const saved = registry.connections.find(c => c.kind === 'cloud' && trimUrl(c.url) === trimUrl(url))

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
