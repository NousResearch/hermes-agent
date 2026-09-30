// Shared always-reply guard for hermes:* IPC handlers. Electron cannot
// structured-clone arbitrary handler rejection values; an uncloneable
// rejection surfaces renderer-side as the opaque "reply was never sent"
// instead of the real cause (401/404/timeout). Every channel maps the
// normalized failure message to its own serializable fallback — a result
// object ({saved, error}), `false`, an empty string, or a cloneable rethrown
// Error when the renderer has a catch that toasts the reason.

export function cloneableError(error: unknown): Error {
  return new Error(error instanceof Error ? error.message : String(error))
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** Cap a promise (backend resolution, network fetch) with a legible timeout.
 *  A late failure of the wrapped work after the timeout already replied is
 *  unobservable, so it is swallowed rather than left unhandled. */
export function withTimeout<T>(work: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const guard = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${message} (timed out after ${Math.round(timeoutMs / 1000)}s)`)), timeoutMs)
  })

  return Promise.race([work, guard]).finally(() => {
    clearTimeout(timer)
    work.catch(() => undefined)
  }) as Promise<T>
}

/** Run `work`, never rejecting: on failure, `onFail` builds the channel's
 *  serializable fallback from the normalized failure message. With
 *  `timeoutMs`, an unresolved `work` (dead backend, unbounded fetch) is
 *  treated as the same failure once the cap elapses. */
export async function replyAlways<T>(
  run: () => Promise<T>,
  onFail: (message: string) => T,
  { timeoutMs, timeoutMessage }: { timeoutMs?: number; timeoutMessage?: string } = {}
): Promise<T> {
  try {
    return await (timeoutMs ? withTimeout(run(), timeoutMs, timeoutMessage ?? `Timed out after ${Math.round(timeoutMs / 1000)}s`) : run())
  } catch (error) {
    return onFail(errorMessage(error))
  }
}
