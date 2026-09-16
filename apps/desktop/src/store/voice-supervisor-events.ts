import {
  type GatewayEvent,
  type ServerRequest,
  type VoiceSupervisorSurfaceEvent,
  voiceSupervisorSurfaceEvent,
  voiceSupervisorSurfaceRequest
} from '@hermes/shared'

type Listener = (event: VoiceSupervisorSurfaceEvent) => void

const listenersBySession = new Map<string, Set<Listener>>()

export function subscribeVoiceSupervisorEvents(sessionId: string, listener: Listener): () => void {
  const listeners = listenersBySession.get(sessionId) ?? new Set<Listener>()

  listeners.add(listener)
  listenersBySession.set(sessionId, listeners)

  return () => {
    listeners.delete(listener)

    if (listeners.size === 0) {
      listenersBySession.delete(sessionId)
    }
  }
}

function deliver(sessionId: string | undefined, surfaceEvent: VoiceSupervisorSurfaceEvent | null): void {
  const listeners = sessionId ? listenersBySession.get(sessionId) : undefined

  if (!listeners?.size || !surfaceEvent) {
    return
  }

  for (const listener of listeners) {
    listener(surfaceEvent)
  }
}

export function routeVoiceSupervisorGatewayEvent(event: GatewayEvent): void {
  if (!event.session_id || !listenersBySession.has(event.session_id)) {
    return
  }

  deliver(event.session_id, voiceSupervisorSurfaceEvent(event))
}

/** Observe-only: a blocking prompt (approval / sudo / secret / clarify) raised for
 *  a voice-supervised session becomes a spoken notice. The request itself is
 *  answered by the app's own handler; a reconnect replay is not re-announced. */
export function routeVoiceSupervisorServerRequest(request: Pick<ServerRequest, 'method' | 'params' | 'replayed'>): void {
  if (request.replayed) {
    return
  }

  deliver(request.params.session_id, voiceSupervisorSurfaceRequest(request.method))
}
