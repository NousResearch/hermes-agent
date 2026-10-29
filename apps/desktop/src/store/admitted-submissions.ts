// The input ids (client submission ids) the owner's pending snapshots list per composer queue key,
// as this page saw them. A send the owner admitted but has not run yet (queued behind a running
// turn or an unresolved one) has no stored row, so only this listing shows that it landed.
const listed = new Map<string, Set<string>>()
const listeners = new Set<(key: string, ids: Set<string>) => void>()

/** Record one pending snapshot of `key` (called by the pending-submissions reconcile). */
export function noteListedSubmissions(key: string, inputIds: Iterable<string>): void {
  const ids = new Set(inputIds)
  listed.set(key, ids)
  listeners.forEach(listener => listener(key, ids))
}

/** Call `listener` with every later snapshot of `key`; returns the unsubscribe. */
export function onListedSubmissions(key: string, listener: (ids: Set<string>) => void): () => void {
  const scoped = (snapshotKey: string, ids: Set<string>) => void (snapshotKey === key && listener(ids))
  listeners.add(scoped)

  return () => void listeners.delete(scoped)
}

/** The ids `key`'s pending snapshot lists, waiting up to `timeoutMs` for this page's first one;
 *  `undefined` when none arrived (an owner that sends no snapshot): the caller then knows nothing. */
export function listedSubmissions(key: string, timeoutMs: number): Promise<Set<string> | undefined> {
  const now = listed.get(key)

  if (now) {
    return Promise.resolve(now)
  }

  return new Promise(resolve => {
    let stop = () => {}

    const timer = setTimeout(() => {
      stop()
      resolve(undefined)
    }, timeoutMs)

    stop = onListedSubmissions(key, ids => {
      stop()
      clearTimeout(timer)
      resolve(ids)
    })
  })
}
