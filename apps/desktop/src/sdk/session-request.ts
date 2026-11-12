import { $gateway } from '@/store/gateway'
import { ambientRequestFor } from '@/store/session-gone-latch'
import { requestForOwnedSession } from '@/store/session-states'

/** Gateway JSON-RPC routed to the backend that OWNS a chat session (its
 *  runtime id), under that session's profile scope — the routing the app's
 *  own session RPCs use, so a tile on another profile or connection is
 *  answered by its owner, not the active socket. `null` = the active
 *  gateway. `signal` stops waiting; it does not cancel work the backend has
 *  already started. Feature-detect on older hosts:
 *  `host.requestForSession?.(…)`. */
export async function requestForSession<T>(
  sessionId: null | string,
  method: string,
  params: Record<string, unknown> = {},
  options: { signal?: AbortSignal; timeoutMs?: number } = {}
): Promise<T> {
  const gateway = $gateway.get()

  if (!gateway) {
    throw new Error('Hermes gateway unavailable')
  }

  return requestForOwnedSession<T>(sessionId, ambientRequestFor(gateway), method, params, options.timeoutMs, options.signal)
}
