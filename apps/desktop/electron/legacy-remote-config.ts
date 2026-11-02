import { normalizeRemoteBaseUrl } from './connection-config'
import { type ConnectionRegistry, reconcileAppliedGlobalConnection } from './connection-registry'
import { type RemoteGatewayTunnelConfig, remoteGatewayTunnelFields } from './remote-gateway-tunnel'

interface TunnelFields {
  sshTunnel?: RemoteGatewayTunnelConfig
}

export function legacyRemoteTunnelFields(
  kind: 'cloud' | 'remote',
  url: unknown,
  incoming: unknown,
  existing?: RemoteGatewayTunnelConfig
): TunnelFields {
  if (incoming === null) {
    return {}
  }

  return remoteGatewayTunnelFields(kind, url, incoming === undefined ? existing : incoming)
}

export function legacyRemoteTunnelFieldsForMode(
  mode: unknown,
  url: unknown,
  incoming: unknown,
  existing?: RemoteGatewayTunnelConfig
): TunnelFields {
  if (mode === 'cloud' && incoming) {
    // An explicit tunnel on Cloud is rejected the same way the registry rejects it.
    return legacyRemoteTunnelFields('cloud', url, incoming)
  }

  return mode === 'remote' ? legacyRemoteTunnelFields('remote', url, incoming, existing) : {}
}

export function safeLegacyRemoteTunnelFields(mode: unknown, url: unknown, incoming: unknown): TunnelFields {
  try {
    return legacyRemoteTunnelFieldsForMode(mode, url, incoming)
  } catch {
    return {}
  }
}

export function sanitizedLegacyRemoteTunnel(mode: unknown, incoming: unknown): RemoteGatewayTunnelConfig | null {
  return mode === 'remote' && incoming && typeof incoming === 'object'
    ? ({ host: String((incoming as Record<string, unknown>).host || '') } as RemoteGatewayTunnelConfig)
    : null
}

export function buildLegacyRemoteBlock({
  remoteUrl,
  authMode,
  token,
  tokenPresent,
  kind = 'remote',
  sshTunnel,
  existingTunnel,
  headers,
  org,
  name,
  remoteProfile
}: {
  remoteUrl: unknown
  authMode: string
  token: unknown
  tokenPresent: boolean
  kind?: 'cloud' | 'remote'
  sshTunnel?: unknown
  existingTunnel?: RemoteGatewayTunnelConfig
  headers?: object
  org?: string
  name?: string
  remoteProfile?: string
}): Record<string, unknown> {
  const url = normalizeRemoteBaseUrl(remoteUrl)

  if (authMode !== 'oauth' && !tokenPresent) {
    throw new Error('Remote gateway session token is required.')
  }

  return {
    url,
    authMode,
    token,
    ...legacyRemoteTunnelFields(kind, url, sshTunnel, existingTunnel),
    ...(headers ? { headers } : {}),
    ...(org ? { org } : {}),
    ...(name ? { name } : {}),
    ...(remoteProfile ? { remoteProfile } : {})
  }
}

/**
 * Registry id of the saved tunnelled v1 Remote gateway at `rawUrl`, or ''.
 * URL-only cookie calls (sign-out, session status) carry the saved gateway URL, which a tunnelled
 * connection never dials; without the id they would read the shared jar, not the one its login wrote.
 */
export function savedLegacyTunnelConnectionId(
  rawUrl: unknown,
  config: Record<string, any> | null | undefined,
  registry: ConnectionRegistry
): string {
  const remote = config?.mode === 'remote' ? config.remote : null

  if (!remote?.sshTunnel) {
    return ''
  }

  try {
    if (normalizeRemoteBaseUrl(rawUrl) !== normalizeRemoteBaseUrl(remote.url)) {
      return ''
    }

    const reconciled = reconcileAppliedGlobalConnection(registry, { mode: 'remote', remote })

    return registry.connections.some(connection => connection.id === reconciled.primary) ? reconciled.primary : ''
  } catch {
    return ''
  }
}
