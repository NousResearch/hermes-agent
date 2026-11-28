export const SUPERSEDED_BACKEND_START = 'Hermes backend start was superseded by a newer connection attempt.'

export function isSupersededBackendStartError(error: unknown): boolean {
  return error instanceof Error && error.message === SUPERSEDED_BACKEND_START
}

/**
 * A stale startHermes attempt must follow the attempt that replaced it.
 * Returning that promise lets the original IPC resolve with the live
 * connection instead of becoming the "Hermes couldn't start" dialog.
 */
export async function replacementBackendPromise<T>(
  own: Promise<T>,
  getCurrent: () => Promise<T> | null,
  waitMs = 1500,
  sleep: (ms: number) => Promise<void> = ms => new Promise(resolve => setTimeout(resolve, ms))
): Promise<T | null> {
  const deadline = Date.now() + waitMs

  for (;;) {
    const current = getCurrent()

    if (current && current !== own) {
      return current
    }

    if (Date.now() >= deadline) {
      return null
    }

    await sleep(Math.min(50, Math.max(0, deadline - Date.now())))
  }
}
