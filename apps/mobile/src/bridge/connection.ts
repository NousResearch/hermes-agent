/**
 * connection.ts — the connection-descriptor half of the bridge.
 *
 * Implements `getConnection` / `getGatewayWsUrl` from the window.hermesDesktop
 * contract. The vendored desktop `store/gateway.ts` + `lib/gateway-ws-url.ts`
 * call these before every connect; `getGatewayWsUrl` mints a FRESH single-use
 * ws-ticket each time (30s TTL, single-use), exactly as the desktop main
 * process does.
 */

import type { HermesConnection } from '@/global'

import { mintWsTicket } from './auth'
import { buildGatewayWsUrl, buildGatewayWsUrlWithTicket } from './connection-config'
import { currentTarget } from './state'

function requireTarget() {
  const target = currentTarget()
  if (!target) throw new Error('Not connected to a gateway.')
  return target
}

/** Build the WS URL the gateway client connects with. OAuth/password gateways
 *  mint a fresh ticket; token gateways embed the long-lived token. */
export async function getGatewayWsUrl(): Promise<string> {
  const target = requireTarget()
  if (target.authMode === 'oauth') {
    const ticket = await mintWsTicket(target.baseUrl)
    return buildGatewayWsUrlWithTicket(target.baseUrl, ticket)
  }
  // token mode: the (currently empty) token rides in the URL. Token-auth
  // gateways aren't the primary target here but the path is kept for parity.
  return buildGatewayWsUrl(target.baseUrl, '')
}

/** Resolve a full connection descriptor. `wsUrl` is minted fresh so a cached
 *  descriptor is always connectable; the vendored re-mint path also re-calls
 *  getGatewayWsUrl on each connect. */
export async function getConnection(): Promise<HermesConnection> {
  const target = requireTarget()
  const wsUrl = await getGatewayWsUrl()
  return {
    baseUrl: target.baseUrl,
    isFullscreen: false,
    mode: 'remote',
    authMode: target.authMode,
    nativeOverlayWidth: 0,
    source: 'settings',
    token: '',
    wsUrl,
    logs: [],
    windowButtonPosition: null,
  }
}
