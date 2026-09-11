type GatewayReconnectHandler = () => Promise<void> | void

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

export function reconnectGateway(): Promise<void> {
  if (inFlight) {
    return inFlight
  }

  const handler = activeHandler

  if (!handler) {
    return Promise.reject(new Error('Gateway reconnect is unavailable'))
  }

  inFlight = Promise.resolve()
    .then(handler)
    .finally(() => {
      inFlight = null
    })

  return inFlight
}
