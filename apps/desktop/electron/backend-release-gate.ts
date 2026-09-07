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
  isShimLocked: () => boolean | Promise<boolean>
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

    if (!(await deps.isShimLocked()) && lingering.length === 0) {
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

  if (!(await deps.isShimLocked())) {
    deps.log(
      `[${tag}] proceeding after deadline: venv shim unlocked, but ${lingering.length} signalled PID(s) still enumerable`
    )

    return { unlocked: true, lingeringPids: lingering }
  }

  return { unlocked: false, lingeringPids: lingering }
}

/** Restart Manager attribution budget for the gate; discovery keeps its own, larger one. */
export const RELEASE_GATE_ATTRIBUTION_BUDGET_MS = 4_000

export interface InstallLockGateProbeDeps {
  /** Exclusive-open sweep over the mutation set, stopping after `limit` findings. */
  probeLocks: (limit: number) => { definite: readonly string[]; shared: readonly string[] }
  /** Per-process module attribution for uv-shared hard links; resolves to the holder count. */
  countAttributedHolders: (budgetMs: number) => Promise<number>
  now?: () => number
  attributionBudgetMs?: number
  /** Minimum interval between attributions; defaults to one per gate run. */
  attributionCooldownMs?: number
}

/**
 * The lock probe `waitForBackendRelease` polls, bounded so it can be asked
 * every RELEASE_GATE_POLL_MS.
 *
 * Two costs had to come off the poll:
 *
 *  - the exclusive-open sweep, which visited the whole mutation set (~270
 *    synchronous openSync calls on a real install: ~27 ms median on the main
 *    thread, seconds when a filter driver is cold). One lock is a complete
 *    answer for this question, so the sweep stops at the first one (~0.3 ms
 *    while the install is held). If that first hit is a uv-shared hard link,
 *    the attribution below re-probes the full set, so nothing is missed.
 *  - the Restart Manager attribution, a PowerShell child with a 12 s budget,
 *    which ran on EVERY poll that saw only uv-shared hard links: up to 50
 *    spawns inside a 15 s gate, each able to outlive the gate itself. It now
 *    runs at most once per cooldown, which defaults to the gate deadline, so
 *    the polling loop attributes once and the post-deadline check may refresh.
 *
 * Until attribution has answered, a shared lock counts as locked (fail
 * closed), and an attribution that throws leaves it locked.
 */
export function createInstallLockGateProbe(deps: InstallLockGateProbeDeps): () => Promise<boolean> {
  const now = deps.now ?? Date.now
  const budgetMs = deps.attributionBudgetMs ?? RELEASE_GATE_ATTRIBUTION_BUDGET_MS
  const cooldownMs = deps.attributionCooldownMs ?? RELEASE_GATE_DEADLINE_MS
  let attributedAt: number | null = null
  let attributedLocked = true

  return async () => {
    const locks = deps.probeLocks(1)

    if (locks.definite.length > 0) {return true}

    if (locks.shared.length === 0) {return false}

    if (attributedAt !== null && now() - attributedAt < cooldownMs) {return attributedLocked}

    attributedAt = now()

    try {
      attributedLocked = (await deps.countAttributedHolders(budgetMs)) > 0
    } catch {
      attributedLocked = true
    }

    return attributedLocked
  }
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
