import type { DesktopRegistryConnection } from '@/global'

import { normalizeGatewayUrl } from './connections-registry'

// `org` values are the refs NAS echoes (`slug ?? id`) and are compared as
// opaque strings. If a team gains or changes its slug, a connection saved under
// the old ref reads as moved once: the reconnect below re-authenticates and
// rewrites the ref, so the mismatch self-heals at the cost of one silent cascade.
export function cloudTeamChanged(connection: DesktopRegistryConnection | undefined, org: string | null): boolean {
  return Boolean(connection?.kind === 'cloud' && org && connection.org !== org)
}

// Only called after the user chooses an agent returned by NAS for this team.
// Keep the connection id and user label; the normal registry save invalidates
// its old pooled routes when org changes. Authenticate before committing it.
// Resolves false when the silent cascade did not land a gateway session (the
// caller warns the same way it does for a fresh connect); true once saved.
export async function reconnectMovedCloudAgent(
  desktop: NonNullable<Window['hermesDesktop']>,
  connection: DesktopRegistryConnection & { url: string },
  org: string,
  isCurrent: () => boolean
): Promise<boolean> {
  const { url } = connection
  await desktop.oauthLogoutConnectionConfig(url)

  if (!isCurrent()) {
    return false
  }

  const result = await desktop.cloud.agentSignIn(url)

  if (!isCurrent() || !result.connected) {
    return false
  }

  await desktop.connections.save({
    id: connection.id,
    kind: 'cloud',
    label: connection.label,
    url,
    authMode: 'oauth',
    org
  })

  // Legacy settings remember a team independently of the registry. Update it
  // only for this same gateway; selecting a secondary must not change defaults.
  const config = await desktop.getConnectionConfig()

  if (!isCurrent()) {
    return false
  }

  if (config.mode === 'cloud' && normalizeGatewayUrl(config.remoteUrl) === normalizeGatewayUrl(url)) {
    await desktop.saveConnectionConfig({ mode: 'cloud', remoteUrl: config.remoteUrl, cloudOrg: org })
  }

  return isCurrent()
}
