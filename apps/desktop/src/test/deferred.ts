/** A promise plus the handle that settles it.
 *
 *  For tests that need a call to stay in flight — asserting the pending UI,
 *  proving a second call is coalesced or ignored, ordering two resolutions —
 *  and then release it on their own schedule. */
export function deferred<T = void>() {
  let resolve!: (value: PromiseLike<T> | T) => void
  let reject!: (reason?: unknown) => void

  const promise = new Promise<T>((done, fail) => {
    resolve = done
    reject = fail
  })

  return { promise, reject, resolve }
}
