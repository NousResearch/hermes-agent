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

// The plugin register() in flight, if any. The loaders wrap each plugin's
// register() in withEventSubscriptionScope() so bare host.onEvent() calls a
// plugin makes while registering get the plugin's unload disposer for free —
// otherwise every hot reload strands one live listener per subscription.
let activeScope: ((dispose: () => void) => void) | null = null

/** Run `fn` with subscriptions auto-tracked by `onDispose` (the caller's unload hook). */
export function withEventSubscriptionScope(onDispose: (dispose: () => void) => void, fn: () => void): void {
  const previous = activeScope
  activeScope = onDispose

  try {
    fn()
  } finally {
    activeScope = previous
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

  // A plugin subscribing mid-register gets unload-tracked without touching
  // its own disposer — double-dispose is a no-op (Set.delete is idempotent).
  activeScope?.(dispose)

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
