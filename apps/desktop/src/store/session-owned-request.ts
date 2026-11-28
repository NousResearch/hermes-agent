/**
 * `requestForOwnedSession` — dispatch a session-scoped RPC through the OWNER of
 * a session, failing closed when nobody can name that owner.
 *
 * This is the client half of #91684: approval.respond (and siblings) sent on
 * the ambient socket land on whatever backend is active, which for a
 * cross-profile session is a backend that never held the approval. An UNKNOWN
 * owner fails closed with an explicit SessionOwnerResolutionError unless the
 * ambient gateway is provably the only backend (legacy single-profile, no
 * registry source).
 *
 * It lives in a topical sibling of `session-states.ts` rather than inside that
 * module, which is already over its code-health line cap. The owner ladder is
 * INJECTED (`setSessionOwnedRequestOwnerResolver`) instead of imported: the
 * ladder walks this module's sibling, so importing it back would close a cycle.
 * `session-states.ts` registers the resolver and re-exports this surface, so
 * every existing caller keeps importing from the same place.
 */
import {
  ambientGatewayOwnsEverySession,
  assertSessionOwnerResolved,
  sessionOwnerIsKnown
} from './session-owner-resolution'
import { requestForSessionProfile, type SessionOwnerScope } from './session-request-router'

/**
 * Cross-profile async owner probe (the REST ladder in
 * `use-session-actions/utils`), registered by the RPC dispatcher when the
 * window wires up. It cannot be imported here: that module imports
 * `store/session-states`, so the import would close a cycle. Null until the
 * dispatcher is created — the sync ladder still answers in that window and a
 * genuinely unknown owner still fails closed, exactly as before.
 */
type SessionOwnerProbe = (sessionId: string) => Promise<SessionOwnerScope>

type KnownOwnerResolver = (sessionId: null | string | undefined) => SessionOwnerScope

let knownOwnerResolver: KnownOwnerResolver | null = null
let sessionOwnerProbe: SessionOwnerProbe | null = null

export function setSessionOwnedRequestOwnerResolver(resolver: KnownOwnerResolver | null): void {
  knownOwnerResolver = resolver
}

export function setSessionOwnerProbe(probe: SessionOwnerProbe | null): void {
  sessionOwnerProbe = probe
}

/**
 * Dispatch a session-scoped RPC through the session's owner (tile route → hint
 * → connection-tagged row / known profile).
 *
 * The sync ladder only sees the loaded sidebar slices. A session that lives in
 * none of them — the ARCHIVED rows are a separate backend query, and a row can
 * also sit outside every loaded page — used to fail closed here even though an
 * id-only REST read names its owner. Run the same cross-profile probe the RPC
 * dispatcher uses before giving up: a hit routes the request (this is what lets
 * session control work on an archived conversation), a miss still throws.
 *
 * The KNOWN-owner path stays SYNCHRONOUS on purpose. `requestForOwnedSession`
 * is awaited by the background pollers, and the gone-latch they time is
 * sensitive to one extra microtask: making this function `async` outright
 * re-fired `process.list` on remount
 * (`status-stack/polling-guard.test.tsx` pins that shape). Only the
 * unknown-owner detour below is async.
 */
export function requestForOwnedSession<T>(
  sessionId: null | string | undefined,
  ambientRequest: <R>(
    method: string,
    params?: Record<string, unknown>,
    timeoutMs?: number,
    signal?: AbortSignal
  ) => Promise<R>,
  method: string,
  params: Record<string, unknown> = {},
  timeoutMs?: number,
  signal?: AbortSignal
): Promise<T> {
  const owner = knownOwnerResolver?.(sessionId)

  if (!sessionId || sessionOwnerIsKnown(owner) || ambientGatewayOwnsEverySession() || !sessionOwnerProbe) {
    try {
      assertSessionOwnerResolved(owner, { method, sessionId })
    } catch (error) {
      return Promise.reject(error)
    }

    return requestForSessionProfile<T>(owner, ambientRequest, method, params, timeoutMs, signal)
  }

  return dispatchViaOwnerProbe<T>(sessionId, owner, sessionOwnerProbe, ambientRequest, method, params, timeoutMs, signal)
}

/**
 * The async rung: name an owner the sync ladder cannot see, then dispatch
 * through it. A probe miss still fails closed — the ambient gateway is
 * presentation state, so routing there would be a guess.
 */
async function dispatchViaOwnerProbe<T>(
  sessionId: string,
  knownOwner: SessionOwnerScope,
  probe: SessionOwnerProbe,
  ambientRequest: <R>(method: string, params?: Record<string, unknown>, timeoutMs?: number, signal?: AbortSignal) => Promise<R>,
  method: string,
  params: Record<string, unknown>,
  timeoutMs?: number,
  signal?: AbortSignal
): Promise<T> {
  const probed = await probe(sessionId).catch(() => undefined)
  const owner = probed ?? knownOwner

  assertSessionOwnerResolved(owner, { method, sessionId })

  return requestForSessionProfile<T>(owner, ambientRequest, method, params, timeoutMs, signal)
}
