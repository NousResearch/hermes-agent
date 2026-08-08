export type LocalBackendRestartResult =
  { ok: true; mode: 'local' } | { ok: false; reason: 'restart-failed'; message: string }

type BackendChildProcess = {
  exitCode: number | null
  signalCode: string | null
  kill: (signal?: number | string) => unknown
  once: (event: 'exit', listener: () => void) => unknown
}

export function isBackendExitPending(child: Pick<BackendChildProcess, 'exitCode' | 'signalCode'> | null | undefined) {
  return Boolean(child && child.exitCode === null && child.signalCode === null)
}

export async function waitForBackendExit(
  child: BackendChildProcess | null | undefined,
  { timeoutMs = 5000, onTimeout }: { timeoutMs?: number; onTimeout?: () => void } = {}
): Promise<void> {
  if (!child || child.exitCode !== null || child.signalCode !== null) {
    return
  }

  await new Promise<void>(resolve => {
    let settled = false
    let timer: ReturnType<typeof setTimeout>

    const finish = () => {
      if (settled) {
        return
      }

      settled = true
      clearTimeout(timer)
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
      }
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
    notifyApplied()

    return { ok: false, reason: 'restart-failed', message: errorMessage(error) }
  }
}
