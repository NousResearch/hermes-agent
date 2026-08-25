/**
 * Pure routing helpers for the contrib wiring controller.
 *
 * Kept out of wiring.tsx so they can be unit-tested without importing the whole
 * React/Electron controller module.
 */

/**
 * Resolve a runtime session id back to its stored id by reverse-scanning the
 * stored->runtime binding map — the same ladder use-session-tile-delegate's
 * `storedSessionIdForRuntime` uses. Returns undefined when the id isn't a known
 * runtime id, so the caller can treat it as already a stored id.
 */
export function findStoredIdForRuntimeId(bindings: Map<string, string>, runtimeId: string): string | undefined {
  for (const [storedId, mapped] of bindings) {
    if (mapped === runtimeId) {
      return storedId
    }
  }

  return undefined
}

/**
 * The stored session id a session-scoped RPC should route by.
 *
 * Route by the session the RPC TARGETS (its `session_id` param), not by the
 * window's focused tile: `requestGateway` is one shared closure for every
 * session RPC, so keying off the focused tile sent a non-focused tile's RPC
 * (a bot chat while another pane is active) to the focused tile's backend — the
 * Bot Mode misroute. `session_id` is a RUNTIME id while tiles/rows key on the
 * STORED id, so translate via the state cache, then the reverse binding scan;
 * an unknown id is already a stored id (several RPCs pass stored ids directly).
 * With no `session_id` at all (ambient/config calls) fall back to the focused
 * then selected tile.
 */
export function resolveRoutingSessionId(args: {
  paramSessionId: string | undefined
  storedIdForRuntime: (runtimeId: string) => string | undefined
  focusedStoredSessionId: null | string
  selectedStoredSessionId: null | string
}): null | string {
  const { focusedStoredSessionId, paramSessionId, selectedStoredSessionId, storedIdForRuntime } = args

  if (paramSessionId) {
    return storedIdForRuntime(paramSessionId) ?? paramSessionId
  }

  return focusedStoredSessionId ?? selectedStoredSessionId
}
