/**
 * backend-release-gate.ts
 *
 * The Windows pre-update unlock gate: after the desktop tree-kills its own
 * backends, decide when it is actually safe to hand off to the updater.
 *
 * Why this exists (#74805): `taskkill /T /F` returns once termination is
 * INITIATED, not completed. A dying `python.exe -m hermes_cli.main serve`
 * stays in the process table while it unmaps .pyd files (AV / NTFS filter
 * drivers stretch this out), and it need not hold the venv `hermes.exe` shim
 * at all — so a gate that only probes the shim can pass on its very first
 * iteration, with zero dwell, while the killed pythons are still
 * terminating. The venv-blocker scan downstream has no liveness filter; it
 * enumerates those dying processes as holders and aborts the hand-off.
 * Result: the FIRST update attempt from the footbar always failed, and the
 * manual retry (by which time the table had settled) succeeded.
 *
 * The gate therefore requires BOTH: the shim unlocked AND every PID we have
 * ever signalled to have actually left the process table. On deadline, the
 * old shim-only criterion is kept as the escape hatch — lingering PIDs past
 * 15s are the venv-blocker re-scan's job, not a new failure mode.
 *
 * Extracted into its own dependency-free module (no electron import) so the
 * gate's decision logic can be asserted directly with fake clocks and fake
 * process tables, following the backend-child.ts pattern.
 */

export interface ReleaseGateDeps {
  /** Probe the venv hermes.exe shim (real: O_RDWR open attempt). */
  isShimLocked: () => boolean
  /** True while `pid` is still enumerable in the process table. */
  isPidAlive: (pid: number) => boolean
  /**
   * Re-collect PIDs that may have (re)spawned since the initial sweep —
   * the supervised primary backend and pool entries. Called every pass.
   */
  collectStragglerPids: () => number[]
  /** Tree-kill (real: taskkill /PID n /T /F). */
  killProcessTree: (pid: number) => void
  /** Async sleep; injectable so tests run on a fake clock. */
  sleep: (ms: number) => Promise<void>
  /** Monotonic-enough clock; injectable for tests. */
  now: () => number
  /** Log sink (real: rememberLog). */
  log: (line: string) => void
}

export interface ReleaseGateResult {
  unlocked: boolean
  /** PIDs we signalled that were still enumerable when the gate resolved. */
  lingeringPids: number[]
}

export const RELEASE_GATE_DEADLINE_MS = 15000
export const RELEASE_GATE_POLL_MS = 300

/**
 * Wait until the install is genuinely releasable: shim unlocked AND every
 * signalled PID gone — or the deadline passes.
 *
 * `initialPids` are the PIDs the caller already signalled (primary backend +
 * pool) before invoking the gate; stragglers collected on each pass are
 * killed and added to the same watch set.
 */
export async function waitForBackendRelease(
  initialPids: number[],
  deps: ReleaseGateDeps,
  tag: string,
  deadlineMs: number = RELEASE_GATE_DEADLINE_MS
): Promise<ReleaseGateResult> {
  const killedPids = new Set<number>(initialPids.filter(pid => Number.isInteger(pid) && pid > 0))

  const deadline = deps.now() + deadlineMs

  while (deps.now() < deadline) {
    const lingering = [...killedPids].filter(pid => deps.isPidAlive(pid))

    if (!deps.isShimLocked() && lingering.length === 0) {
      deps.log(`[${tag}] venv shim unlocked and ${killedPids.size} signalled backend PID(s) exited; safe to proceed`)

      return { unlocked: true, lingeringPids: [] }
    }

    // A supervised backend can respawn between kill and check (grandchildren,
    // pool entries registered mid-teardown). Re-collect and re-kill each pass
    // instead of trusting the initial sweep.
    for (const pid of deps.collectStragglerPids()) {
      if (Number.isInteger(pid) && pid > 0) {
        killedPids.add(pid)
        deps.killProcessTree(pid)
      }
    }

    await deps.sleep(RELEASE_GATE_POLL_MS)
  }

  // Deadline reached. Keep the pre-#74805 success criterion — an unlocked
  // shim — rather than inventing a new failure mode for PIDs that linger
  // past the deadline; the venv-blocker re-scan downstream covers that
  // residue (and a REAL foreign holder still fails the shim probe).
  const lingering = [...killedPids].filter(pid => deps.isPidAlive(pid))

  if (!deps.isShimLocked()) {
    deps.log(
      `[${tag}] proceeding after deadline: venv shim unlocked, but ${lingering.length} signalled PID(s) still enumerable`
    )

    return { unlocked: true, lingeringPids: lingering }
  }

  return { unlocked: false, lingeringPids: lingering }
}

/**
 * Liveness probe for a PID on Windows. `process.kill(pid, 0)` delivers
 * nothing; it only probes existence: EPERM ⇒ exists but inaccessible (still
 * alive), ESRCH ⇒ gone.
 */
export function isPidAliveWindows(pid: number): boolean {
  try {
    process.kill(pid, 0)

    return true
  } catch (err: any) {
    return Boolean(err) && err.code === 'EPERM'
  }
}
