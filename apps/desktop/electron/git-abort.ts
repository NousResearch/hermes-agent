/**
 * Kill a git child when its AbortSignal fires, and escalate if it will not die.
 *
 * The bundle-skew probe aborts its git at a 10s timeout, because a treeless
 * partial clone can lazy-fetch trees for minutes. SIGTERM alone is not enough
 * to end that: a git blocked in a fetch can take as long again to unwind. So
 * the abort sends SIGTERM, and a short grace period later SIGKILL, which the
 * kernel cannot defer.
 *
 * Split out of main.ts as a small pure function of a child-like object, so the
 * escalation is provable without spawning a process or booting Electron.
 */

/** How long a git gets to close on SIGTERM before SIGKILL follows. */
export const GIT_KILL_GRACE_MS = 2_000

/**
 * The slice of ChildProcess this needs. Structural, so a real ChildProcess
 * fits and a test can pass a fake that records signals and fires 'close'.
 */
export interface AbortKillableChild {
  kill(signal?: NodeJS.Signals | number): boolean
  once(event: string | symbol, listener: (...args: any[]) => void): unknown
  removeListener?(event: string | symbol, listener: (...args: any[]) => void): unknown
}

/**
 * Wire `signal` to `child`: SIGTERM on abort, SIGKILL if 'close' or 'error'
 * has not arrived within `graceMs`. Idempotent teardown — once the child is
 * gone, both the escalation timer and the listeners are dropped, so a later
 * abort cannot signal a dead (or reused) child.
 *
 * An already-aborted signal kills immediately, because the spawn happened
 * before the abort was observed.
 */
export function killChildOnAbort(
  child: AbortKillableChild,
  signal: AbortSignal,
  graceMs: number = GIT_KILL_GRACE_MS
): void {
  let escalation: ReturnType<typeof setTimeout> | null = null
  // Set by teardown. Checked before every kill and inside the timer, so a child
  // that is already gone is never signalled again (its pid may even be reused).
  let closed = false

  const teardown = (): void => {
    closed = true

    if (escalation) {
      clearTimeout(escalation)
      escalation = null
    }

    signal.removeEventListener('abort', onAbort)
    child.removeListener?.('close', teardown)
    child.removeListener?.('error', teardown)
  }

  const onAbort = (): void => {
    if (closed) {
      return
    }

    // This runs inside controller.abort(), where an escaped throw is an
    // uncaught exception in whatever callback aborted (the probe's timeout).
    // A kill that throws — an already-reaped pid, EPERM — must not escape.
    try {
      child.kill('SIGTERM')
    } catch {
      // Deliberately swallowed; the escalation below still gets its chance.
    }

    // A child that closed the instant it was signalled needs no SIGKILL.
    if (closed) {
      return
    }

    escalation = setTimeout(() => {
      escalation = null

      if (closed) {
        return
      }

      try {
        child.kill('SIGKILL')
      } catch {
        // The timer must never throw either.
      }
    }, graceMs)
  }

  // 'close', not 'exit', matches how runGit resolves: the stdio pipes must
  // drain, and a git that closed its pipes is one that will not need SIGKILL.
  child.once('close', teardown)
  child.once('error', teardown)

  if (signal.aborted) {
    onAbort()
  } else {
    signal.addEventListener('abort', onAbort)
  }
}
