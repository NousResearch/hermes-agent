/**
 * The plugin-facing server→client request tap, the request sibling of
 * `contrib/events.ts`: a plugin can ANSWER a backend request method
 * (`server_requests.send(method, ...)`) without the app owning the logic.
 *
 * The app's own dispatch (`use-message-stream/gateway-event/server-requests.ts`)
 * tries built-in handlers first, then consults this registry; a plugin handler
 * that returns `true` claims the request and MUST answer it (`request.respond`),
 * exactly like a built-in. Listener-style isolation: a throwing plugin can't
 * break app request handling — its error is answered to the backend instead of
 * the answer it owed, so the blocked tool returns now rather than at deadline.
 *
 * Subscriptions made while a plugin's `register()` runs are retired with the
 * plugin on unload/reload/disable through the same disposer list that tracks
 * `host.onEvent` (`trackGatewayEventDisposers` wraps the whole register call).
 */

import { JSON_RPC_INTERNAL_ERROR } from '@hermes/shared'

import type { ScopedServerRequest } from '@/store/gateway'

/** A plugin's answer to one request method: claim the request (answer it) or ignore it. */
export type PluginServerRequestHandler = (request: ScopedServerRequest) => boolean | void

const handlers = new Map<string, Set<PluginServerRequestHandler>>()

/** Register a handler for one request method. Returns a disposer. */
export function onPluginServerRequest(method: string, handler: PluginServerRequestHandler): () => void {
  const set = handlers.get(method) ?? new Set()
  set.add(handler)
  handlers.set(method, set)

  return () => {
    set.delete(handler)

    if (set.size === 0) {
      handlers.delete(method)
    }
  }
}

/** Does any plugin want this method? Cheap pre-check for the dispatcher. */
export function pluginWantsServerRequest(method: string): boolean {
  return handlers.has(method)
}

/**
 * Hand an unclaimed request to the plugin handlers for its method. True when
 * a plugin claimed it; a plugin throw answers an error to the backend (the
 * request owes an answer no matter what).
 */
export function dispatchPluginServerRequest(request: ScopedServerRequest): boolean {
  const set = handlers.get(request.method)

  if (!set) {
    return false
  }

  for (const handler of set) {
    let claimed: boolean | void

    try {
      claimed = handler(request)
    } catch (error) {
      console.error('[plugins] server request handler failed', request.method, error)
      request.fail(JSON_RPC_INTERNAL_ERROR, error instanceof Error ? error.message : String(error))

      return true
    }

    if (claimed !== false && claimed !== undefined) {
      return true
    }
  }

  return false
}
