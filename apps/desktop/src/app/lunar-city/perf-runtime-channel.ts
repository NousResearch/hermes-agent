interface RuntimeRequest {
  action: string
  payload: unknown
  requestId: string
  type: 'request'
}

interface RuntimePort {
  onmessage: ((event: MessageEvent<unknown>) => void) | null
  postMessage(value: unknown): void
  start?(): void
}

let runtimePort: RuntimePort | undefined
let responder: ((action: string, payload: unknown) => Promise<unknown>) | undefined
let claimed = false

function isRuntimeRequest(value: unknown): value is RuntimeRequest {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }

  const candidate = value as Partial<RuntimeRequest>

  return candidate.type === 'request' && typeof candidate.action === 'string' && typeof candidate.requestId === 'string'
}

function bindPort(port: RuntimePort): void {
  if (runtimePort) {
    return
  }

  runtimePort = port

  port.onmessage = event => {
    if (!responder || !isRuntimeRequest(event.data)) {
      return
    }

    const request = event.data
    void responder(request.action, request.payload)
      .then(value => port.postMessage({ requestId: request.requestId, type: 'response', value }))
      .catch(error =>
        port.postMessage({
          requestId: request.requestId,
          type: 'response',
          value: { error: error instanceof Error ? error.message : String(error) }
        })
      )
  }

  port.start?.()

  if (responder) {
    port.postMessage({ type: 'ready' })
  }
}

if (typeof window !== 'undefined' && window.__LUNAR_CITY_PERF_AUTHORIZED__ === true) {
  window.addEventListener(
    'message',
    event => {
      if (
        event.source === window &&
        event.data?.type === 'hermes:lunar-city-perf-runtime-port-v1' &&
        event.ports.length === 1
      ) {
        bindPort(event.ports[0])
      }
    },
    { capture: true }
  )
}

export const lunarCityPerfRuntimeEndpoint = {
  onRequest(callback: (action: string, payload: unknown) => Promise<unknown>): () => void {
    if (claimed) {
      throw new Error('Lunar City performance runtime responder already claimed')
    }

    claimed = true
    responder = callback
    runtimePort?.postMessage({ type: 'ready' })

    return () => {
      responder = undefined
      runtimePort = undefined
    }
  }
}
