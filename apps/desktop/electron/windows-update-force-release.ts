/**
 * Windows update force-release orchestration.
 *
 * Selecting Update authorizes terminating every process that currently holds or
 * executes files inside the target install. This module:
 *   1. discovers holders via the existing scanner + Windows Restart Manager;
 *   2. binds each target to PID + create-time (+ resource evidence);
 *   3. terminates leaf-first with a SuperF4-style TerminateProcess path;
 *   4. stays inside a five-second non-elevated budget;
 *   5. escalates to an elevated helper only on access-denied survivors.
 *
 * Behavioral reference (not copied): stefansundin/superf4 @ 6b677d4, superf4.c
 * OpenProcess/TerminateProcess sequence. Implement from Win32 docs only.
 */

import type { DesktopPluginServiceProcess } from './venv-blocker-scan'

export type ForceReleaseHolder = {
  pid: number
  createdAt: number
  name: string
  cmdline: string
  source: 'scanner' | 'restart-manager'
  /**
   * ONE exact file this process currently holds, as proven by Restart
   * Manager. The exact-terminate script re-proves ownership of this path
   * before it terminates anything, so it must never be a fabricated default
   * or a joined list (both made every termination fail with
   * TERMINATION_CURRENT_LOCK_OWNERSHIP_MISMATCH on 2026-09-03/04).
   */
  resource?: string
  /** Every locked path attributed to this process, for messages and logs. */
  resources?: string[]
  parentPid?: number
  wrapperPid?: number
  role?: 'worker' | 'wrapper' | 'other'
  /**
   * How to stop this holder. A Desktop plugin service is stopped as a unit
   * through the scanner (supervisor, wrapper, workers); the generic path
   * terminates exactly one PID and would leave the supervisor to respawn it.
   */
  terminateVia?: 'exact-process' | 'desktop-plugin-service'
  service?: DesktopPluginServiceProcess
}

export type ForceReleaseTerminateResult =
  | { kind: 'terminated' }
  | { kind: 'already-gone' }
  | { kind: 'create-time-mismatch' }
  | { kind: 'access-denied'; win32Error: number; detail?: string }
  | { kind: 'protected'; win32Error: number }
  | { kind: 'failed'; detail: string; win32Error?: number }

export type WindowsUpdateForceReleaseDeps = {
  now?: () => number
  wait?: (delayMs: number) => Promise<void>
  isResourceLocked: () => Promise<boolean>
  /**
   * Discover scanner holders. `budgetMs` is the hard remaining wall-clock
   * budget; implementations must honor it (or the orchestrator races them).
   */
  listScannerHolders: (budgetMs: number) => Promise<ForceReleaseHolder[]>
  listRestartManagerHolders: (budgetMs: number) => Promise<ForceReleaseHolder[]>
  /**
   * Terminate one holder. Must honor `signal` by cancelling any child work and
   * guaranteeing no process-mutation side effect after abort settles.
   */
  terminateHolder: (
    holder: ForceReleaseHolder,
    budgetMs: number,
    signal?: AbortSignal,
    /** One absolute deadline shared with the native mutation boundary. */
    deadlineAt?: number
  ) => Promise<ForceReleaseTerminateResult>
  excludePids?: ReadonlySet<number>
  /** Non-elevated wall-clock budget for discovery plus identity-verified termination. */
  deadlineMs?: number
  settleMs?: number
  /** Forensics: every discovery pass, so a refusal names what it saw and how long it took. */
  onDiscovery?: (info: { pass: number; scanner: number; restartManager: number; elapsedMs: number }) => void
  /** Forensics: the outcome of every termination attempt. */
  onHolderOutcome?: (holder: ForceReleaseHolder, result: ForceReleaseTerminateResult, elapsedMs: number) => void
}

export type WindowsUpdateForceReleaseOutcome =
  | { kind: 'clear' }
  | { kind: 'needs-elevation'; holders: ForceReleaseHolder[]; message: string }
  | { kind: 'blocked'; holders: ForceReleaseHolder[]; message: string }
  | { kind: 'timeout'; holders: ForceReleaseHolder[]; message: string }

const DEFAULT_DEADLINE_MS = 5_000
const DEFAULT_SETTLE_MS = 150
/** RM emits integer-second create times; scanner may be fractional. */
export const HOLDER_CREATE_TIME_MATCH_SECONDS = 1.5

function wait(delayMs: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, delayMs))
}

export function holdersMatchIdentity(
  left: Pick<ForceReleaseHolder, 'pid' | 'createdAt'>,
  right: Pick<ForceReleaseHolder, 'pid' | 'createdAt'>,
  toleranceSeconds = HOLDER_CREATE_TIME_MATCH_SECONDS
): boolean {
  if (left.pid !== right.pid) {return false}

  if (!Number.isFinite(left.createdAt) || !Number.isFinite(right.createdAt)) {return false}

  return Math.abs(left.createdAt - right.createdAt) <= toleranceSeconds
}

/**
 * Leaf-first order: deepest descendants first, then workers before wrappers.
 * Stable for unrelated peers (original relative order among ties).
 */
export function orderHoldersLeafFirst(holders: readonly ForceReleaseHolder[]): ForceReleaseHolder[] {
  const byPid = new Map(holders.map(entry => [entry.pid, entry] as const))
  const depthMemo = new Map<number, number>()

  const depthOf = (entry: ForceReleaseHolder, stack: Set<number> = new Set()): number => {
    const cached = depthMemo.get(entry.pid)

    if (cached !== undefined) {
      return cached
    }

    if (stack.has(entry.pid)) {
      return 0
    }

    stack.add(entry.pid)

    const parentRef =
      (entry.parentPid && byPid.get(entry.parentPid)) ||
      (entry.wrapperPid && byPid.get(entry.wrapperPid)) ||
      null

    let depth = 0

    if (parentRef) {
      depth = depthOf(parentRef, stack) + 1
    } else if (entry.role === 'worker') {
      depth = 1
    } else if (entry.role === 'wrapper') {
      depth = 0
    }

    depthMemo.set(entry.pid, depth)
    stack.delete(entry.pid)

    return depth
  }

  return holders
    .map((entry, index) => ({
      entry,
      index,
      depth: depthOf(entry),
      roleBias: entry.role === 'worker' ? 1 : entry.role === 'wrapper' ? -1 : 0
    }))
    .sort(
      (left, right) =>
        right.depth - left.depth || right.roleBias - left.roleBias || left.index - right.index
    )
    .map(item => item.entry)
}

/**
 * When holders carry parent/wrapper edges among the set, annotate roles so
 * orderHoldersLeafFirst can drain leaves before roots in production mappings.
 */
export function attachHolderTreeRelationships(
  holders: readonly ForceReleaseHolder[]
): ForceReleaseHolder[] {
  if (holders.length === 0) {return []}

  const byPid = new Map(holders.map(entry => [entry.pid, entry] as const))
  const childCount = new Map<number, number>()

  const withEdges = holders.map(entry => {
    const parentPid =
      (entry.parentPid && byPid.has(entry.parentPid) ? entry.parentPid : undefined) ??
      (entry.wrapperPid && byPid.has(entry.wrapperPid) ? entry.wrapperPid : undefined)

    if (parentPid != null) {
      childCount.set(parentPid, (childCount.get(parentPid) ?? 0) + 1)
    }

    return {
      ...entry,
      ...(parentPid != null
        ? {
            parentPid: entry.parentPid ?? parentPid,
            wrapperPid: entry.wrapperPid ?? (entry.wrapperPid === parentPid ? parentPid : entry.wrapperPid)
          }
        : {})
    }
  })

  return withEdges.map(entry => {
    const isParent = (childCount.get(entry.pid) ?? 0) > 0

    const hasParent =
      (entry.parentPid != null && byPid.has(entry.parentPid)) ||
      (entry.wrapperPid != null && byPid.has(entry.wrapperPid))

    if (entry.role === 'worker' || entry.role === 'wrapper') {
      return entry
    }

    if (hasParent) {
      return { ...entry, role: 'worker' as const }
    }

    if (isParent) {
      return { ...entry, role: 'wrapper' as const }
    }

    return entry
  })
}

export function mergeInstallHolders(
  holders: readonly ForceReleaseHolder[],
  excludePids: ReadonlySet<number> = new Set()
): ForceReleaseHolder[] {
  const merged: ForceReleaseHolder[] = []
  // PIDs whose primary resource already came from Restart Manager evidence.
  const provenResource = new Set<number>()

  for (const entry of holders) {
    if (!Number.isInteger(entry.pid) || entry.pid <= 0) {
      continue
    }

    if (!Number.isFinite(entry.createdAt) || entry.createdAt <= 0) {
      continue
    }

    if (excludePids.has(entry.pid)) {
      continue
    }

    const existingIndex = merged.findIndex(candidate => holdersMatchIdentity(candidate, entry))

    if (existingIndex < 0) {
      merged.push({ ...entry })

      if (entry.source === 'restart-manager' && entry.resource) {
        provenResource.add(entry.pid)
      }

      continue
    }

    const existing = merged[existingIndex]!

    const evidence = Array.from(
      new Set(
        [...(existing.resources ?? []), existing.resource, ...(entry.resources ?? []), entry.resource].filter(
          Boolean
        ) as string[]
      )
    )

    // Keep ONE exact resource for the ownership proof and prefer the first
    // Restart Manager attribution: it names a file the process really maps.
    // A joined "a; b" string is not a path and never verifies.
    const entryProvesResource = entry.source === 'restart-manager' && Boolean(entry.resource)

    const primaryResource =
      entryProvesResource && !provenResource.has(existing.pid) ? entry.resource : existing.resource ?? entry.resource

    if (entryProvesResource) {
      provenResource.add(existing.pid)
    }

    const preferredSource = existing.source === 'scanner' || entry.source === 'scanner' ? 'scanner' : entry.source

    // Prefer the more precise (fractional) create-time when both match within tolerance.
    const createdAt =
      !Number.isInteger(existing.createdAt) || Number.isInteger(entry.createdAt)
        ? existing.createdAt
        : entry.createdAt

    merged[existingIndex] = {
      ...existing,
      ...entry,
      createdAt,
      source: preferredSource,
      resource: primaryResource,
      ...(evidence.length ? { resources: evidence } : {}),
      parentPid: existing.parentPid ?? entry.parentPid,
      wrapperPid: existing.wrapperPid ?? entry.wrapperPid,
      role: existing.role === 'other' || existing.role == null ? entry.role ?? existing.role : existing.role,
      terminateVia: existing.terminateVia ?? entry.terminateVia,
      service: existing.service ?? entry.service
    }
  }

  return attachHolderTreeRelationships(merged)
}

export function formatHolderLine(holder: ForceReleaseHolder): string {
  const evidence = holder.resources?.length ? holder.resources : holder.resource ? [holder.resource] : []
  const resource = evidence.length ? ` resource=${evidence.join('; ')}` : ''

  return `PID ${holder.pid} ${holder.name}${resource}`
}

function elevationMessage(holders: readonly ForceReleaseHolder[]): string {
  const sample = holders
    .slice(0, 5)
    .map(formatHolderLine)
    .join('; ')

  return (
    'Update needs Administrator permission to stop processes still locking this Hermes install. ' +
    `Survivors: ${sample || 'unknown'}. Choose Force update (Administrator) to continue, or close those processes and retry.`
  )
}

function blockedMessage(holders: readonly ForceReleaseHolder[], detail: string): string {
  const sample = holders
    .slice(0, 5)
    .map(formatHolderLine)
    .join('; ')

  return (
    `Update aborted: ${detail}. ` +
    `Still holding the install: ${sample || 'unknown'}. ` +
    'The virtual environment was not modified.'
  )
}

/**
 * Bounds non-mutating probes against the remaining wall-clock budget. Mutating
 * termination owns and drains its native process/Job boundary and must never be
 * abandoned by this helper.
 */
export function raceWithBudget<T>(work: Promise<T>, budgetMs: number, onTimeout: () => T): Promise<T> {
  const budget = Math.trunc(budgetMs)

  if (budget <= 0) {return Promise.resolve(onTimeout())}

  return new Promise<T>((resolve, reject) => {
    let settled = false

    const timer = setTimeout(() => {
      if (settled) {return}
      settled = true
      resolve(onTimeout())
    }, budget)

    void work.then(
      value => {
        if (settled) {return}
        settled = true
        clearTimeout(timer)
        resolve(value)
      },
      error => {
        if (settled) {return}
        settled = true
        clearTimeout(timer)
        reject(error)
      }
    )
  })
}

export async function runWindowsUpdateForceRelease(
  deps: WindowsUpdateForceReleaseDeps
): Promise<WindowsUpdateForceReleaseOutcome> {
  const now = deps.now ?? Date.now
  const sleep = deps.wait ?? wait
  const deadlineMs = Math.max(0, deps.deadlineMs ?? DEFAULT_DEADLINE_MS)
  const settleMs = Math.max(0, deps.settleMs ?? DEFAULT_SETTLE_MS)
  const excludePids = deps.excludePids ?? new Set<number>()
  const started = now()
  const deadline = started + deadlineMs
  const remaining = () => Math.max(0, deadline - now())

  const resourceLockedWithinDeadline = async (): Promise<boolean> => {
    const budget = remaining()

    if (budget <= 0) {return true}

    return await raceWithBudget(deps.isResourceLocked(), budget, () => true)
  }

  const waitWithinDeadline = async (delayMs: number): Promise<void> => {
    const budget = Math.min(Math.max(0, delayMs), remaining())

    if (budget <= 0) {return}
    await raceWithBudget(sleep(budget), budget, () => undefined)
  }

  if (!(await resourceLockedWithinDeadline())) {
    return { kind: 'clear' }
  }

  let accessDenied: ForceReleaseHolder[] = []
  let protectedHolders: ForceReleaseHolder[] = []
  let mismatchHolders: ForceReleaseHolder[] = []
  let lastHolders: ForceReleaseHolder[] = []
  let restartManagerHolders: ForceReleaseHolder[] = []
  let pass = 0

  while (remaining() > 0) {
    if (!(await resourceLockedWithinDeadline())) {
      return { kind: 'clear' }
    }

    const passStarted = now()
    const scanBudget = remaining()

    if (scanBudget <= 0) {break}

    const scanned = await raceWithBudget(
      deps.listScannerHolders(scanBudget),
      scanBudget,
      () => [] as ForceReleaseHolder[]
    )

    if (remaining() <= 0) {break}

    // Restart Manager enumeration is the expensive step (about half a second
    // per holder for friendly-name resolution; 8 s measured on a loaded
    // workstation). Query it once up front, and again only when the scanner
    // sees nothing, so later passes spend the budget on termination.
    if (pass === 0 || scanned.length === 0) {
      const rmBudget = remaining()

      restartManagerHolders = await raceWithBudget(
        deps.listRestartManagerHolders(rmBudget),
        rmBudget,
        () => [] as ForceReleaseHolder[]
      )
    }

    const fromRm = restartManagerHolders

    if (remaining() <= 0) {
      lastHolders = orderHoldersLeafFirst(mergeInstallHolders([...scanned, ...fromRm], excludePids))

      break
    }

    const holders = orderHoldersLeafFirst(mergeInstallHolders([...scanned, ...fromRm], excludePids))
    lastHolders = holders

    try {
      deps.onDiscovery?.({ pass, scanner: scanned.length, restartManager: fromRm.length, elapsedMs: now() - passStarted })
    } catch {
      void 0
    }

    pass += 1

    if (holders.length === 0) {
      // Locked with no enumerable holders: fail closed rather than mutate.
      const pause = Math.min(settleMs || 50, remaining())

      if (pause > 0) {
        await waitWithinDeadline(pause)
      }

      if (!(await resourceLockedWithinDeadline())) {
        return { kind: 'clear' }
      }

      break
    }

    accessDenied = []
    protectedHolders = []
    mismatchHolders = []

    for (const target of holders) {
      const budget = remaining()

      if (budget <= 0) {
        break
      }

      // The native boundary receives the same absolute deadline and does not
      // resolve until its wrapper, Jobs, helpers, and authenticated descendants
      // are terminal. Never race/abandon this mutating operation.
      const controller = new AbortController()
      const abortTimer = setTimeout(() => controller.abort(), budget)
      const terminateStarted = now()
      let result: ForceReleaseTerminateResult

      try {
        result = await deps.terminateHolder(target, budget, controller.signal, deadline)
      } finally {
        clearTimeout(abortTimer)
      }

      try {
        deps.onHolderOutcome?.(target, result, now() - terminateStarted)
      } catch {
        void 0
      }

      switch (result.kind) {
        case 'terminated':

        case 'already-gone':
          break

        case 'access-denied':
          accessDenied.push(target)

          break

        case 'protected':
          protectedHolders.push(target)

          break

        case 'create-time-mismatch':
          mismatchHolders.push(target)

          break

        case 'failed':
          // Keep trying others; final lock proof decides.
          break
      }
    }

    const settle = Math.min(Math.max(settleMs, 1), remaining())

    if (settle > 0) {
      await waitWithinDeadline(settle)
    }

    if (!(await resourceLockedWithinDeadline())) {
      return { kind: 'clear' }
    }

    // Access-denied survivors need elevation — stop the quick path promptly.
    if (accessDenied.length > 0) {
      return {
        kind: 'needs-elevation',
        holders: accessDenied,
        message: elevationMessage(accessDenied)
      }
    }

    if (protectedHolders.length > 0) {
      return {
        kind: 'blocked',
        holders: protectedHolders,
        message: blockedMessage(
          protectedHolders,
          'a protected or unkillable process still holds install files (Win32 access denied / protected process)'
        )
      }
    }

    if (mismatchHolders.length > 0 && holders.every(h => mismatchHolders.some(m => m.pid === h.pid))) {
      return {
        kind: 'blocked',
        holders: mismatchHolders,
        message: blockedMessage(
          mismatchHolders,
          'PID create-time no longer matches (PID reuse); refusing to terminate a different process'
        )
      }
    }

    // Guarantee forward progress even when wait() is a no-op in tests.
    if (now() <= passStarted && deps.now) {
      // Fake clocks only advance in wait(); ensure we cannot spin forever.
      break
    }
  }

  if (!(await resourceLockedWithinDeadline())) {
    return { kind: 'clear' }
  }

  if (accessDenied.length > 0) {
    return {
      kind: 'needs-elevation',
      holders: accessDenied,
      message: elevationMessage(accessDenied)
    }
  }

  if (protectedHolders.length > 0) {
    return {
      kind: 'blocked',
      holders: protectedHolders,
      message: blockedMessage(
        protectedHolders,
        'a protected or unkillable process still holds install files (Win32 access denied / protected process)'
      )
    }
  }

  if (mismatchHolders.length > 0) {
    return {
      kind: 'blocked',
      holders: mismatchHolders,
      message: blockedMessage(
        mismatchHolders,
        'PID create-time no longer matches (PID reuse); refusing to terminate a different process'
      )
    }
  }

  return {
    kind: 'timeout',
    holders: lastHolders,
    message: blockedMessage(
      lastHolders,
      'install file locks could not be cleared within the force-release budget'
    )
  }
}
