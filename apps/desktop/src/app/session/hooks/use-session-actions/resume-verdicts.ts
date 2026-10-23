// A "session genuinely doesn't exist" failure (deleted, or an id from a wiped /
// rotated backend) — the REST transcript 404s with `Session not found`. Distinct
// from a transient/wedged backend (ECONNREFUSED, timeout), which must still
// retry rather than discard the id.
export function isSessionGoneError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err ?? '')

  return message.includes('404') || /session not found/i.test(message)
}

/**
 * What to do when a resume's RPC and REST fallback BOTH came back
 * gone-looking (#88540).
 *
 * A 404 is only proof of deletion when it came from the backend that owns
 * the session. During (or moments after) a profile/connection switch the
 * request can land on a backend that has never heard of the id — the
 * cross-profile Bots-pane open is the reproducer: the route is written
 * correctly, the resume races the gateway swap, both lookups 404 on the
 * wrong backend, and the "genuinely gone" branch yanks the window to the
 * blank new-chat route while the target session is perfectly alive.
 *
 * `'retry'` keeps the route and arms the bounded auto-retry (which re-runs
 * the resume once the swap settles); `'draft'` is reserved for a session
 * that is verifiably gone in calm conditions.
 */
export function goneSessionVerdict(options: {
  /** The session was created by this window in this run — never discard. */
  createdThisRun: boolean
  /** A post-failure re-resolve still finds the row on SOME profile. */
  stillListed: boolean
  /** A profile swap or connection switch is in flight (or just targeted). */
  switchInFlight: boolean
}): 'draft' | 'retry' {
  return options.createdThisRun || options.stillListed || options.switchInFlight ? 'retry' : 'draft'
}

/**
 * The busy value a resume/activate response should land with (#70449).
 *
 * `running` in a `session.activate` / `session.resume` payload is a snapshot
 * taken when the RPC was issued. A turn that started — or streamed — after
 * that snapshot has already marked the runtime busy in the live cache, so a
 * stale `running: false` must never rewind it: that is exactly how opening an
 * in-progress chat cleared its working indicator while the agent was still
 * going. Preserving the newer live busy is safe, because the turn's own
 * terminal signal (running:false via session.info / the settle path) remains
 * the only authority that ends it, and the background-sync reaper clears
 * truly lost turns.
 *
 * A snapshot that says `running: true` always wins — adopting a live turn is
 * never stale.
 */
export function resolveResumedBusy(snapshotRunning: boolean | null | undefined, liveBusy: boolean): boolean {
  return Boolean(snapshotRunning) || liveBusy
}
