import { translateNow } from '@/i18n'
import {
  dialPrimaryRoute,
  type GatewayReconnectHandler,
  type GatewayReconnectOptions,
  registerGatewayPrimaryRoute
} from '@/store/gateway-supervisor'

/**
 * The window's reconnect *request* surface: who owns the primary route, and how
 * a caller asks for it to be re-dialed (#C5).
 *
 * The state this module used to hold — the single in-flight dial — lives in
 * `store/gateway-supervisor.ts` now, with the epoch, the backoff ladder and the
 * wake action. Nothing here keeps a promise or a timer: it maps the public
 * "reconnect the window's gateway" request onto the supervisor's primary route,
 * which is also the route a wake dials.
 */

export function registerGatewayReconnect(
  handler: GatewayReconnectHandler,
  ports: { reportsOpen?: () => boolean } = {}
): () => void {
  return registerGatewayPrimaryRoute({ dial: handler, reportsOpen: ports.reportsOpen })
}

export function reconnectGateway(options?: GatewayReconnectOptions): Promise<void> {
  return dialPrimaryRoute(options)
}

/** Toast button that re-dials the active connection — attached wherever a
 *  send fails because Hermes is offline (sudo/secret/approval prompts …). */
export function reconnectAction(): { label: string; onClick: () => void } {
  return {
    label: translateNow('prompts.reconnect'),
    onClick: () => void reconnectGateway().catch(() => undefined)
  }
}
