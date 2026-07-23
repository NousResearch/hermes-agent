/**
 * The plugin-facing gateway event tap. The wiring fans every inbound gateway
 * event through here BEFORE its own dispatch; plugins subscribe by type (or
 * `'*'`) via `host.onEvent`. Listeners are isolated — a throwing plugin can
 * never break the app's event handling — and emit is zero-cost when nobody
 * listens.
 */

import type { RpcEvent } from '@/types/hermes'

export type GatewayEventListener = (event: RpcEvent) => void

const listeners = new Map<string, Set<GatewayEventListener>>()

/** Subscribe to gateway events by `type` (`'*'` = everything). Returns a disposer. */
export function onGatewayEvent(type: string, listener: GatewayEventListener): () => void {
  const set = listeners.get(type) ?? new Set()
  set.add(listener)
  listeners.set(type, set)

  return () => {
    set.delete(listener)

    if (set.size === 0) {
      listeners.delete(type)
    }
  }
}

/** Fan an event to subscribers (wiring-side; call before app dispatch). */
export function emitGatewayEvent(event: RpcEvent): void {
  if (listeners.size === 0) {
    return
  }

  for (const type of [event.type, '*']) {
    for (const listener of listeners.get(type) ?? []) {
      try {
        listener(event)
      } catch (error) {
        console.error('[plugins] gateway event listener failed', error)
      }
    }
  }
}
