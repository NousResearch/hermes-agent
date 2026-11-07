import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import type { DesktopConnectionsRegistry } from '@/global'
import { normalizeProfileKey } from '@/store/profile'

import type { TerminalGateway } from './terminal-gateway'

/** Where a shell for `connectionId` runs: an SSH gateway runs it on its own
 *  host; This device, URL/token and cloud gateways (no shell transport) run it
 *  here and say so. Undefined while the registry is unknown (or on a host with
 *  none, like the Webapp), so main keeps routing by the window. */
export function shellGatewayFor(
  connectionId: null | string | undefined,
  profile: null | string | undefined,
  registry: DesktopConnectionsRegistry | null | undefined
): TerminalGateway | undefined {
  const source = connectionId ? registry?.connections.find(connection => connection.id === connectionId) : undefined

  if (!source) {
    return undefined
  }

  return source.kind === 'ssh'
    ? { connectionId: source.id, profile: normalizeProfileKey(profile) }
    : { connectionId: LOCAL_CONNECTION_ID }
}

