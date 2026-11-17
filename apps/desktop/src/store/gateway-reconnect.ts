import { translateNow } from '@/i18n'

export type GatewayReconnectSource = 'manual' | 'restart-followthrough'

export interface GatewayReconnectOptions {
  /**
   * Why the reconnect is running. `'manual'` (default) is an explicit user
   * recovery — the handler may unconditionally re-dial, including retrying a
   * credential that requires sign-in. `'restart-followthrough'` is the
   * automatic hand-off after a gateway restart: the socket often SURVIVED
   * (the restart targets the messaging gateway, not this client's backend),
   * so the handler probes first instead of force-closing a healthy socket.
   */
  source?: GatewayReconnectSource
}

type GatewayReconnectHandler = (options?: GatewayReconnectOptions) => Promise<void> | void

let activeHandler: GatewayReconnectHandler | null = null
let inFlight: Promise<void> | null = null

export function registerGatewayReconnect(handler: GatewayReconnectHandler): () => void {
  activeHandler = handler

  return () => {
    if (activeHandler === handler) {
      activeHandler = null
    }
  }
}

export function reconnectGateway(options?: GatewayReconnectOptions): Promise<void> {
  if (inFlight) {
    return inFlight
  }

  const handler = activeHandler

  if (!handler) {
    return Promise.reject(new Error('Gateway reconnect is unavailable'))
  }

  inFlight = Promise.resolve()
    .then(() => handler(options))
    .finally(() => {
      inFlight = null
    })

  return inFlight
}

/** Toast button that re-dials the active connection — attached wherever a
 *  send fails because Hermes is offline (sudo/secret/approval prompts …). */
export function reconnectAction(): { label: string; onClick: () => void } {
  return {
    label: translateNow('prompts.reconnect'),
    onClick: () => void reconnectGateway().catch(() => undefined)
  }
}
