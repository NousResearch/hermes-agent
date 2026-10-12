import type { ClientSessionState } from '../../../types'

/** The busy-relevant slice of a cached runtime state. */
export type ResumeBusyState = Pick<ClientSessionState, 'awaitingResponse' | 'busy' | 'sawAssistantPayload'>

/**
 * The busy value a resume/activate response should land with (#70449).
 *
 * `running` in a `session.activate` / `session.resume` payload is a snapshot
 * taken when the RPC was issued. A turn that started — or streamed — after
 * that snapshot has already marked the runtime busy in the live cache, so a
 * stale `running: false` must never rewind it: that is exactly how opening an
 * in-progress chat cleared its working indicator while the agent was still
 * going.
 *
 * The snapshot is only stale relative to state written AFTER the RPC was
 * issued. `atRequest` is the runtime's cached state when the request went out
 * (undefined when the cache held none): a busy claim that is still that very
 * object is older than the snapshot, and `running: false` is the backend's
 * authoritative word that the turn is over. Those are the claims whose
 * terminal events were lost — a socket closed mid-turn by a connection switch
 * or a reconnect — and preserving them pinned the spinner forever, because
 * nothing else would ever settle them. A prompt this window submitted that the
 * backend has not started yet (awaiting its first payload) is the exception:
 * the backend honestly reports it idle, and the local claim is the newer fact.
 *
 * A snapshot that says `running: true` always wins — adopting a live turn is
 * never stale. One that does not report `running` at all (older backends)
 * cannot clear anything.
 */
export function resolveResumedBusy(
  snapshotRunning: boolean | null | undefined,
  latest: ResumeBusyState | undefined,
  atRequest: ResumeBusyState | undefined
): boolean {
  if (snapshotRunning) {
    return true
  }

  if (!latest?.busy) {
    return false
  }

  if (snapshotRunning === undefined || snapshotRunning === null) {
    return true
  }

  return latest !== atRequest || Boolean(latest.awaitingResponse && !latest.sawAssistantPayload)
}

/** Busy resolution for one `session.resume` request, which may hand back a
 *  parked runtime the cache already holds. `capture` records the
 *  conversation's cached runtime states as the request goes out — the
 *  `atRequest` side of resolveResumedBusy. A call that joined another caller's
 *  in-flight resume never captures: it cannot know that request's time, so it
 *  cannot prove any cached claim older than the snapshot. */
export function resumeRequestBaseline(
  cache: { readonly current: ReadonlyMap<string, ClientSessionState> },
  storedSessionId: string
) {
  let atRequest: Map<string, ClientSessionState> | null = null

  return {
    capture<T>(request: () => T): T {
      atRequest = new Map([...cache.current].filter(([, state]) => state.storedSessionId === storedSessionId))

      return request()
    },
    resolve: (runtimeId: string, snapshotRunning: boolean | null | undefined) =>
      resolveResumedBusy(snapshotRunning, cache.current.get(runtimeId), atRequest?.get(runtimeId))
  }
}
