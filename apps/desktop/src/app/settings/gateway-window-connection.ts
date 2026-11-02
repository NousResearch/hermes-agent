import type { DesktopConnectionsRegistry, DesktopRegistryConnection } from '@/global'

import type { GatewaySettingsState } from './gateway-settings'

function sameRemoteUrl(a: string, b: string | undefined): boolean {
  const normalize = (url: string) => url.trim().replace(/\/+$/, '').toLowerCase()

  return Boolean(b) && normalize(a) === normalize(b ?? '')
}

function seededFrom(saved: GatewaySettingsState, connection: DesktopRegistryConnection): GatewaySettingsState {
  if (connection.kind === 'local') {
    return { ...saved, mode: 'local' }
  }

  if (connection.kind === 'ssh') {
    return {
      ...saved,
      mode: 'ssh',
      sshHost: connection.host ?? '',
      sshUser: connection.user ?? '',
      sshPort: connection.port ?? null,
      sshKeyPath: connection.keyPath ?? '',
      sshRemoteHermesPath: connection.remoteHermesPath ?? '',
      sshRemoteProfile: connection.remoteProfile ?? ''
    }
  }

  const url = connection.url ?? ''
  // The saved token belongs to the saved URL. Saving another gateway's URL
  // without a new token would carry it there, so only a matching URL keeps it.
  const keepsToken = saved.mode !== 'local' && sameRemoteUrl(url, saved.remoteUrl)

  return {
    ...saved,
    mode: connection.kind,
    remoteUrl: url,
    remoteAuthMode: connection.authMode ?? 'token',
    cloudOrg: connection.org ?? '',
    remoteOauthConnected: keepsToken && saved.remoteOauthConnected,
    remoteTokenSet: keepsToken && saved.remoteTokenSet,
    remoteTokenPreview: keepsToken ? saved.remoteTokenPreview : null,
    remoteTokenPlainText: keepsToken && saved.remoteTokenPlainText
  }
}

/** Settings > Gateways > This window describes the gateway the window is on.
 *  The saved config names the registry primary; after a pick (or a last-used
 *  launch) the window can be on another registered gateway, and the cards must
 *  show that one instead of claiming the primary. An env override pins the
 *  window to the saved route, so it is shown as saved. */
export function windowConnectionSettings(
  saved: GatewaySettingsState,
  registry: DesktopConnectionsRegistry | null,
  activeConnectionId: null | string
): GatewaySettingsState {
  if (saved.envOverride || !registry || !activeConnectionId || activeConnectionId === registry.primary) {
    return saved
  }

  const connection = registry.connections.find(candidate => candidate.id === activeConnectionId)

  return connection ? seededFrom(saved, connection) : saved
}
