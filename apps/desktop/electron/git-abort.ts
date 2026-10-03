/**
 * Kill a git child when its AbortSignal fires, and escalate if it will not die.
 *
 * The bundle-skew probe aborts its git at a 10s timeout, because a treeless
 * partial clone can lazy-fetch trees for minutes. SIGTERM alone is not enough
 * to end that: a git blocked in a fetch can take as long again to unwind. So
 * the abort sends SIGTERM, and a short grace period later SIGKILL, which the
 * kernel cannot defer.
 *
 * The signal must reach the whole PROCESS TREE, not the top pid: a git that
 * lazy-fetches spawns `fetch -> index-pack -> pack-objects` as its own
 * children, and those survive a plain child.kill(), reparented to PID 1
 * (#125243). runGit spawns the probe's git detached on POSIX, so the child
 * leads its own process group and a negative-pgid signal reaches the
 * descendants; on Windows forceKillProcessTree (taskkill /T /F) follows the
 * ancestry instead. Both are injectable, mirroring stopBackendChild in
 * backend-child.ts, so the group/tree semantics stay provable in tests.
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
  /**
   * The process id, set only once the spawn succeeded. Undefined means the
   * spawn never happened (or failed before a process existed).
   */
  pid?: number
  /** Set once the process exited; null while it runs. Optional for fakes. */
  exitCode?: number | null
  /** Set once the process was signalled; null while it runs. Optional for fakes. */
  signalCode?: NodeJS.Signals | null
  kill(signal?: NodeJS.Signals | number): boolean
  on(event: string | symbol, listener: (...args: any[]) => void): unknown
  removeListener?(event: string | symbol, listener: (...args: any[]) => void): unknown
}

/**
 * Wire `signal` to `child`: SIGTERM on abort, SIGKILL if the child is not
 * proven gone within `graceMs`. Idempotent teardown — once the child is gone,
 * both the escalation timer and the listeners are dropped, so a later abort
 * cannot signal a dead (or reused) child.
 *
 * "Gone" has three proofs and any one of them cancels the escalation: an
 * 'exit' or a 'close' event, or a non-null exitCode/signalCode. 'close' waits
 * for the stdio pipes to drain and can lag a killed process, so a pid that has
 * already exited may otherwise be signalled again — and reused by then.
 *
 * An already-aborted signal kills immediately, because the spawn happened
 * before the abort was observed.
 */
export function killChildOnAbort(
  child: AbortKillableChild,
  signal: AbortSignal,
  graceMs: number = GIT_KILL_GRACE_MS,
  deps: {
    /** POSIX: signal the whole process group (negative pgid). Real: process.kill. */
    killGroup?: (pgid: number, signal: NodeJS.Signals) => void
    /** Windows: taskkill /T /F by pid ancestry. Real: forceKillProcessTree in main.ts. */
    forceKillProcessTree?: (pid: number) => void
    /** Defaults to the real platform check; injectable for tests. */
    isWindows?: boolean
  } = {}
): void {
  const isWindows = deps.isWindows ?? process.platform === 'win32'
  const killGroup = deps.killGroup ?? ((pgid: number, sig: NodeJS.Signals): boolean => process.kill(pgid, sig))
  // On Windows the tree-kill helper is required (no process groups); on POSIX
  // the group signal subsumes it and the fallback stays child.kill().
  const killTree = deps.forceKillProcessTree ?? (isWindows ? () => {} : undefined)

  /**
   * Signal the child AND its descendants. The top pid first (a group signal
   * needs the group alive to mean anything), then the group on POSIX or the
   * ancestry tree on Windows. Best-effort at every rung: a thrown kill must
   * not stop the escalation below from firing.
   */
  const killTreeNow = (): void => {
    if (child.pid !== undefined) {
      if (isWindows) {
        killTree?.(child.pid)
      } else {
        try {
          killGroup(-child.pid, 'SIGKILL')
        } catch {
          /* the group may already be gone; the top pid's own kill follows */
        }
      }
    }
  }
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
    child.removeListener?.('exit', teardown)
    child.removeListener?.('error', onError)
  }

  /**
   * True when Node's own fields prove the process has stopped: exitCode is set
   * once it exited, signalCode once it was signalled; null means still running.
   * A fake that carries neither field offers no proof here, and the 'exit' and
   * 'close' listeners carry it instead.
   */
  const hasExited = (): boolean =>
    (child.exitCode !== undefined && child.exitCode !== null) ||
    (child.signalCode !== undefined && child.signalCode !== null)

  /**
   * Node emits 'error' on two very different things: a spawn that failed, and
   * a kill (SIGTERM/SIGKILL) that failed. Only the first proves the child is
   * gone. After a failed kill the process may still be running, so tearing
   * down here would cancel the SIGKILL escalation and leave a git alive. The
   * pid is the test: a child that never spawned has none, so it is torn down;
   * a spawned child keeps listening, and only 'exit'/'close' or a set
   * exitCode/signalCode prove it exited.
   *
   * Registered with `on`, not `once`: a failed SIGTERM and the later failed
   * SIGKILL each emit 'error', and a spent once-listener would leave the
   * second one unhandled — which throws in the Electron main process. The
   * listener is dropped in teardown, once the child is provably gone.
   */
  const onError = (): void => {
    if (child.pid === undefined) {
      teardown()
    }
  }

  const onAbort = (): void => {
    if (closed) {
      return
    }

    // This runs inside controller.abort(), where an escaped throw is an
    // uncaught exception in whatever callback aborted (the probe's timeout).
    // A kill that throws — an already-reaped pid, EPERM — must not escape.
    //
    // An already-exited child is checked BEFORE the SIGTERM, not only before
    // the SIGKILL: Node sets exitCode/signalCode the moment the process is
    // gone, and its pid may already be reused by a later spawn. A signal to a
    // reused pid is a signal to an unrelated process.
    if (hasExited()) {
      teardown()

      return
    }

    try {
      child.kill('SIGTERM')
      // SIGTERM reaches only the top pid; git's fetch/index-pack descendants
      // keep running under it. The tree kill is best-effort here and hardens
      // into SIGKILL below if the child has not closed by the grace period.
      killTreeNow()
    } catch {
      // Deliberately swallowed; the escalation below still gets its chance.
    }

    // A child that closed the instant it was signalled needs no SIGKILL, and
    // neither does one whose fields already say it exited.
    if (closed || hasExited()) {
      teardown()

      return
    }

    escalation = setTimeout(() => {
      escalation = null

      if (closed || hasExited()) {
        teardown()

        return
      }

      killTreeNow()

      try {
        child.kill('SIGKILL')
      } catch {
        // The timer must never throw either.
      }
    }, graceMs)

    // A pending kill timer must not hold the process open on its own.
    ;(escalation as { unref?: () => void }).unref?.()
  }

  // 'close' is how runGit resolves: the stdio pipes must drain. 'exit' fires
  // earlier and is also proof the process is gone, so both tear down.
  child.on('close', teardown)
  child.on('exit', teardown)
  child.on('error', onError)

  if (signal.aborted) {
    onAbort()
  } else {
    signal.addEventListener('abort', onAbort)
  }
}
