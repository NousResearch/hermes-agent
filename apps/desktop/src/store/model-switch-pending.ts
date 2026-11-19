import { atom } from 'nanostores'

/**
 * In-flight model switches, by live session runtime id.
 *
 * `config.set model` is a gateway round trip: until its answer lands, the
 * composer pill must not keep presenting a model name the switch may still
 * roll back — it shows a spinner instead. Draft picks (no live session) never
 * reach the gateway and never mark anything.
 *
 * A COUNT, not a boolean: two picks on one session can overlap, and the older
 * answer landing first must not clear the newer pick's spinner.
 */
export const $modelSwitchPending = atom<Readonly<Record<string, number>>>({})

/** One `config.set model` request for this session is now waiting on the gateway. */
export function beginModelSwitchPending(sessionId: string): void {
  const pending = $modelSwitchPending.get()

  $modelSwitchPending.set({ ...pending, [sessionId]: (pending[sessionId] ?? 0) + 1 })
}

/** That request settled (applied, deferred, refused, confirmed or failed); the last one clears the mark. */
export function endModelSwitchPending(sessionId: string): void {
  const pending = $modelSwitchPending.get()
  const count = pending[sessionId] ?? 0

  if (count === 0) {
    return
  }

  const next = { ...pending }

  if (count > 1) {
    next[sessionId] = count - 1
  } else {
    delete next[sessionId]
  }

  $modelSwitchPending.set(next)
}
