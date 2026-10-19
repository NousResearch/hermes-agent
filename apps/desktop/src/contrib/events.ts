/**
 * The plugin-facing gateway event tap. The wiring fans every inbound gateway
 * event through here BEFORE its own dispatch; plugins subscribe by type (or
 * `'*'`) via `host.onEvent`. Listeners are isolated — a throwing plugin can
 * never break the app's event handling — and emit is zero-cost when nobody
 * listens.
 */

import type { GatewayEvent } from '@hermes/shared'

export type GatewayEventListener = (event: GatewayEvent) => void

const listeners = new Map<string, Set<GatewayEventListener>>()
let activeDisposerTracker: ((dispose: () => void) => void) | null = null

/** Run a plugin registration while associating its event listeners with its
 * unload disposer list. Runtime plugins are evaluated afresh on every reload,
 * so their host subscriptions must leave with the previous module instance. */
export function trackGatewayEventDisposers<T>(track: (dispose: () => void) => void, register: () => T): T {
  const previous = activeDisposerTracker
  activeDisposerTracker = track

  try {
    return register()
  } finally {
    activeDisposerTracker = previous
  }
}

/** Subscribe to gateway events by `type` (`'*'` = everything). Returns a disposer. */
export function onGatewayEvent(type: string, listener: GatewayEventListener): () => void {
  const set = listeners.get(type) ?? new Set()
  set.add(listener)
  listeners.set(type, set)

  const dispose = () => {
    set.delete(listener)

    if (set.size === 0) {
      listeners.delete(type)
    }
  }

  activeDisposerTracker?.(dispose)

  return dispose
}

/** Fan an event to subscribers (wiring-side; call before app dispatch). */
export function emitGatewayEvent(event: GatewayEvent): void {
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
