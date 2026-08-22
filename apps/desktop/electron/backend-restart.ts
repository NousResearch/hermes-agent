export type LocalBackendRestartResult =
  { ok: true; mode: 'local' } | { ok: false; reason: 'restart-failed'; message: string }

export type BackendChildProcess = {
  exitCode: number | null
  signalCode: string | null
  kill: (signal?: number | string) => unknown
  once: (event: 'exit', listener: () => void) => unknown
}

export function isBackendExitPending(child: Pick<BackendChildProcess, 'exitCode' | 'signalCode'> | null | undefined) {
  return Boolean(child && child.exitCode === null && child.signalCode === null)
}

/**
 * Wait for observed child exit. This helper never escalates by itself;
 * callers that own termination must provide `onTimeout`. Without it, the
 * helper resolves after the bounded grace period even if the child remains
 * alive.
 */
export async function waitForBackendExit(
  child: BackendChildProcess | null | undefined,
  {
    timeoutMs = 5000,
    escalationGraceMs = 2000,
    onTimeout
  }: { timeoutMs?: number; escalationGraceMs?: number; onTimeout?: () => void } = {}
): Promise<void> {
  if (!child || child.exitCode !== null || child.signalCode !== null) {
    return
  }

  await new Promise<void>(resolve => {
    let settled = false
    let timer: ReturnType<typeof setTimeout>
    let graceTimer: ReturnType<typeof setTimeout> | undefined

    const finish = () => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)
      clearTimeout(graceTimer)
      resolve()
    }

    child.once('exit', finish)
    timer = setTimeout(() => {
      try {
        onTimeout?.()
      } catch {
        // Escalation is best effort; the exit event remains the completion gate.
      }

      if (child.exitCode !== null || child.signalCode !== null) {
        finish()

        return
      }

      // The escalated kill normally produces an 'exit' event. Never block the
      // restart flow forever when it does not arrive.
      graceTimer = setTimeout(finish, escalationGraceMs)
    }, timeoutMs)
  })
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export async function restartLocalBackend({
  teardown,
  start,
  notifyApplied
}: {
  teardown: () => Promise<void>
  start: () => Promise<unknown>
  notifyApplied: () => void
}): Promise<LocalBackendRestartResult> {
  try {
    await teardown()
    await start()
    notifyApplied()

    return { ok: true, mode: 'local' }
  } catch (error) {
    // Keep the renderer re-home notification on startup failure: it clears
    // stale session state and lets the boot/reconnect path report recovery
    // failure without treating this notification as readiness.
    notifyApplied()

    return { ok: false, reason: 'restart-failed', message: errorMessage(error) }
  }
}
