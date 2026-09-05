// Route-less (Sessions-list) tiles carry no ownerRoute until their resume
// proves the owning backend, so neither foreground pin rung (route / runtime
// event scope) can hold their socket open. This leaf records, per stored
// session id, the composite gateway scope a session.resume dialed; the
// foreground rung in session-states consults it to pin a mounted tile that
// still has no ownerRoute. Otherwise the resume lease's finally disposes the
// fresh socket and the backend reaps the runtime (reclaim → re-dial loop,
// #93892 shape). The map is renderer-lifetime and keyed by stored id; the pin
// still lives on the tile, so it never latches — this is only attribution.
// Dependency-free leaf on purpose: the gateway must not import the
// session/tile stores.
const ownerScopeBySessionId = new Map<string, string>()

export function recordSessionOwnerScope(sessionId: string, scope: string): void {
  ownerScopeBySessionId.set(sessionId, scope)
}

export function sessionOwnerScopeFor(sessionId: string): string | undefined {
  return ownerScopeBySessionId.get(sessionId)
}
