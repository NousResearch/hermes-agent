/** Cancel a startup wait, and check ownership before invoking its continuation. */
export async function runBackendStartStep<T>(signal: AbortSignal | undefined, run: () => T | Promise<T>): Promise<T> {
  signal?.throwIfAborted()

  if (!signal) {
    return run()
  }

  let onAbort: () => void = () => {}

  const cancelled = new Promise<never>((_resolve, reject) => {
    onAbort = () => reject(signal.reason)
    signal.addEventListener('abort', onAbort, { once: true })
  })

  try {
    const result = await Promise.race([run(), cancelled])
    signal.throwIfAborted()

    return result
  } finally {
    signal.removeEventListener('abort', onAbort)
  }
}
