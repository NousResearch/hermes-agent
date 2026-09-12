import type { DesktopConnectionsRegistry, DesktopRegistryConnection } from '@/global'
import type { GatewayConnectRequest } from '@/lib/gateway-connect-link'

type DesktopBridge = Window['hermesDesktop']

function existingConnection(registry: DesktopConnectionsRegistry, url: string): DesktopRegistryConnection | undefined {
  return registry.connections.find(
    connection =>
      (connection.kind === 'remote' || connection.kind === 'cloud') && connection.url?.replace(/\/+$/, '') === url
  )
}

export async function connectLinkedGateway(
  request: GatewayConnectRequest,
  bridge: Pick<DesktopBridge, 'connections' | 'oauthLoginConnectionConfig'>,
  activate: (registry: DesktopConnectionsRegistry, id: string) => Promise<void>,
  signInIncomplete: string
): Promise<void> {
  let registry = await bridge.connections.list()
  let connection = existingConnection(registry, request.url)

  if (!connection || connection.authMode === 'oauth') {
    const signedIn = await bridge.oauthLoginConnectionConfig(request.url)

    if (!signedIn.connected) {
      throw new Error(signInIncomplete)
    }
  }

  // Sign-in can outlive a registry edit in another window. Re-read before saving.
  registry = await bridge.connections.list()
  connection = existingConnection(registry, request.url)

  if (!connection) {
    const names = new Set(registry.connections.map(item => item.label.toLowerCase()))
    let label = request.name
    let suffix = 2

    while (names.has(label.toLowerCase())) {
      label = `${request.name} (${suffix++})`
    }

    const saved = await bridge.connections.save({ kind: 'remote', label, url: request.url, authMode: 'oauth' })

    registry = saved.registry
    connection = saved.connection
  }

  await activate(registry, connection.id)
}
