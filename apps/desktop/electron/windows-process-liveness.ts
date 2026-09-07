/**
 * Read-only liveness and identity probes for the Windows termination boundary.
 *
 * A process is proved live or absent only by an authenticated child-process
 * exit status; timeouts, access errors and spawn failures are 'unknown' and
 * never licence a claim of absence. Identity is (pid, createdAt) so a reused
 * PID cannot be mistaken for the process we killed. Every probe here is
 * side-effect free — the mutating half lives in windows-process-terminate.ts —
 * and every one of them is bounded by the caller's shared deadline.
 */

import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

import { windowsPowerShellExecutable } from './windows-powershell-path'
import { buildProcessTreeSnapshotScript } from './windows-process-terminate-scripts'

const execFileAsync = promisify(execFile)

export type ProcessIdentity = { pid: number; createdAt?: number }

export async function readProcessCreatedAt(
  pid: number,
  timeoutMs = 2_000
): Promise<number | null> {
  if (!Number.isInteger(pid) || pid <= 0) {return null}
  const budget = Math.trunc(timeoutMs)

  if (budget <= 0) {return null}

  if (process.platform !== 'win32') {return null}

  try {
    const { stdout } = await execFileAsync(
      windowsPowerShellExecutable(),
      [
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        `$p = Get-Process -Id ${Math.trunc(pid)} -ErrorAction Stop; ` +
          'if ($p.HasExited -or $p.WaitForExit(0)) { exit 3 }; ' +
          '[DateTimeOffset]::new($p.StartTime.ToUniversalTime()).ToUnixTimeSeconds()'
      ],
      { encoding: 'utf8', windowsHide: true, timeout: budget }
    )

    const value = Number(String(stdout).trim())

    return Number.isFinite(value) && value > 0 ? value : null
  } catch {
    return null
  }
}

export type ProcessLiveness = 'live' | 'absent' | 'unknown'

/**
 * Discriminated liveness probe outcome.
 * Only authenticated child-process exits may prove live/absent.
 * Error metadata (timeout/access/spawn) is never treated as an exit code.
 */
export type LivenessProbeResult =
  | { kind: 'exit'; code: number }
  | { kind: 'error'; code?: string | number; message?: string }

/** Injectable runner for process liveness probes (tests inject fakes). */
export type LivenessProbeRunner = (
  pid: number,
  timeoutMs: number
) => Promise<LivenessProbeResult>

type DeadlineProbeResult<T> =
  | { completed: true; value: T }
  | { completed: false }

/**
 * Resolve a read-only probe only while the shared deadline remains. The
 * underlying native operation is deliberately consumed after a timeout so a
 * late rejection cannot become an unhandled promise, but its value is never
 * allowed back into the mutation boundary.
 */
export async function resolveBeforeDeadline<T>(
  operation: () => Promise<T>,
  deadlineAt: number
): Promise<DeadlineProbeResult<T>> {
  const remaining = Math.max(0, deadlineAt - Date.now())

  if (remaining <= 0) {return { completed: false }}

  const pending = Promise.resolve().then(operation)
  let timer: NodeJS.Timeout | undefined

  const timeout = new Promise<DeadlineProbeResult<T>>(resolve => {
    timer = setTimeout(() => resolve({ completed: false }), Math.max(1, remaining - 1))
  })

  try {
    const outcome = await Promise.race([
      pending.then(value => ({ completed: true as const, value })),
      timeout
    ])

    if (!outcome.completed || Date.now() > deadlineAt) {return { completed: false }}

    return outcome
  } catch {
    return { completed: false }
  } finally {
    if (timer) {clearTimeout(timer)}
    void pending.catch(() => undefined)
  }
}

/**
 * Pure classification of a liveness probe outcome.
 * live only for kind=exit/code=0; absent only for kind=exit/code=3.
 * Every kind=error is unknown regardless of embedded code metadata.
 */
export function classifyLivenessProbeResult(result: LivenessProbeResult): ProcessLiveness {
  if (result.kind === 'exit') {
    if (result.code === 0) {return 'live'}

    if (result.code === 3) {return 'absent'}

    return 'unknown'
  }

  // kind=error: timeout/access/spawn/malformed — never prove absence from error metadata.
  return 'unknown'
}

function execFileFailureToLivenessResult(error: any): LivenessProbeResult {
  // Timeout/killed probes are errors even if a numeric code is present.
  if (error?.killed === true || error?.signal) {
    return {
      kind: 'error',
      code: typeof error?.code === 'string' || typeof error?.code === 'number' ? error.code : undefined,
      message: String(error?.message ?? error)
    }
  }

  // Node execFile puts authenticated child exit status on error.code as a number.
  if (typeof error?.code === 'number') {
    return { kind: 'exit', code: error.code }
  }

  // Some paths expose exit status on .status while .code is a string errno.
  if (typeof error?.status === 'number') {
    return { kind: 'exit', code: error.status }
  }

  return {
    kind: 'error',
    code: typeof error?.code === 'string' || typeof error?.code === 'number' ? error.code : undefined,
    message: String(error?.message ?? error)
  }
}

async function defaultLivenessProbeRunner(
  pid: number,
  timeoutMs: number
): Promise<LivenessProbeResult> {
  if (process.platform !== 'win32') {
    try {
      process.kill(pid, 0)

      return { kind: 'exit', code: 0 }
    } catch (error: any) {
      // ESRCH is an authenticated "no such process" from the kill(2) probe.
      if (error?.code === 'ESRCH') {return { kind: 'exit', code: 3 }}

      return {
        kind: 'error',
        code: error?.code,
        message: String(error?.message ?? error)
      }
    }
  }

  try {
    await execFileAsync(
      windowsPowerShellExecutable(),
      [
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        // A terminated process object can remain enumerable while another
        // process still owns a handle. Its zero-time wait is nevertheless
        // authoritative terminal-state evidence. Exit 0 = runnable, 3 =
        // absent/terminated, anything else = unknown.
        `$p = Get-Process -Id ${Math.trunc(pid)} -ErrorAction SilentlyContinue; ` +
          'if ($null -eq $p) { exit 3 }; ' +
          'try { if ($p.HasExited -or $p.WaitForExit(0)) { exit 3 }; exit 0 } catch { exit 4 }'
      ],
      { windowsHide: true, timeout: timeoutMs }
    )

    return { kind: 'exit', code: 0 }
  } catch (error: any) {
    return execFileFailureToLivenessResult(error)
  }
}

export async function probeProcessLiveness(
  pid: number,
  timeoutMs = 2_000,
  runner: LivenessProbeRunner = defaultLivenessProbeRunner
): Promise<ProcessLiveness> {
  if (!Number.isInteger(pid) || pid <= 0) {return 'absent'}
  const budget = Math.trunc(timeoutMs)

  if (budget <= 0) {return 'unknown'}
  const result = await runner(pid, budget)

  return classifyLivenessProbeResult(result)
}

export async function identitiesStillPresent(
  identities: readonly ProcessIdentity[],
  {
    deadlineAt,
    livenessRunner,
    readCreatedAt
  }: {
    deadlineAt?: number
    livenessRunner?: LivenessProbeRunner
    readCreatedAt?: (pid: number, timeoutMs: number) => Promise<number | null>
  } = {}
): Promise<ProcessIdentity[]> {
  const survivors: ProcessIdentity[] = []

  const absoluteDeadline =
    typeof deadlineAt === 'number' && Number.isFinite(deadlineAt) ? deadlineAt : Date.now() + 2_000

  const remaining = () => Math.max(0, absoluteDeadline - Date.now())
  const runner = livenessRunner ?? defaultLivenessProbeRunner
  const createdAtReader = readCreatedAt ?? readProcessCreatedAt

  for (const identity of identities) {
    const left = remaining()

    if (left <= 0) {
      // No time for an absence probe: keep known/unknown identities as survivors.
      survivors.push(identity)

      continue
    }

    const slice = Math.max(1, Math.floor(left / Math.max(1, identities.length - survivors.length)))

    const createdAtResult = await resolveBeforeDeadline(
      () => createdAtReader(identity.pid, slice),
      absoluteDeadline
    )

    if (!createdAtResult.completed) {
      survivors.push(identity)

      continue
    }

    const createdAt = createdAtResult.value

    if (createdAt == null) {
      // Do not infer absence from a null create-time read. Probe liveness with
      // remaining budget; only explicit not-found proves absence.
      const liveLeft = remaining()

      if (liveLeft <= 0) {
        survivors.push(identity)

        continue
      }

      const livenessResult = await resolveBeforeDeadline(
        () => probeProcessLiveness(identity.pid, liveLeft, runner),
        absoluteDeadline
      )

      if (!livenessResult.completed || livenessResult.value !== 'absent') {
        // live, unknown, or an expired probe => survivor
        survivors.push(identity)
      }

      continue
    }

    // Unknown generation: any completed create-time read is still only
    // identity evidence; a matching live generation remains a survivor.
    if (identity.createdAt == null || !Number.isFinite(identity.createdAt)) {
      survivors.push({ pid: identity.pid, createdAt })

      continue
    }

    if (Math.abs(createdAt - identity.createdAt) <= 1.5) {
      survivors.push(identity)
    }
  }

  return survivors
}

/**
 * Snapshot a Windows process tree (root + descendants) as PID + create-time
 * identities so post-kill verification can detect PID reuse.
 */
export async function snapshotProcessTreeIdentities(
  rootPid: number,
  {
    timeoutMs = 1_500,
    deadlineAt
  }: { timeoutMs?: number; deadlineAt?: number } = {}
): Promise<ProcessIdentity[]> {
  if (!Number.isInteger(rootPid) || rootPid <= 0) {return []}
  const requestedBudget = Math.max(0, Math.trunc(timeoutMs))

  const remainingBudget =
    typeof deadlineAt === 'number' && Number.isFinite(deadlineAt)
      ? Math.max(0, Math.trunc(deadlineAt - Date.now()))
      : requestedBudget

  const budget = Math.min(requestedBudget, remainingBudget)

  // A zero/negative caller budget is a hard no-probe condition on every
  // platform. In particular, do not fall through to a fresh default-timeout
  // probe after the shared deadline is exhausted.
  if (budget <= 0) {
    return [{ pid: rootPid }]
  }

  if (process.platform !== 'win32') {
    const createdAt = await readProcessCreatedAt(rootPid, budget)

    return createdAt == null ? [] : [{ pid: rootPid, createdAt }]
  }

  const startedAt = Date.now()

  try {
    const { stdout } = await execFileAsync(
      windowsPowerShellExecutable(),
      [
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-Command',
buildProcessTreeSnapshotScript(rootPid)
      ],
      { encoding: 'utf8', windowsHide: true, timeout: budget }
    )

    const identities: ProcessIdentity[] = []

    for (const part of String(stdout || '')
      .trim()
      .split(';')
      .filter(Boolean)) {
      const [pidText, createdText] = part.split('|')
      const pid = Number(pidText)
      const createdAt = Number(createdText)

      if (Number.isInteger(pid) && pid > 0 && Number.isFinite(createdAt) && createdAt > 0) {
        identities.push({ pid, createdAt })
      }
    }

    return identities
  } catch {
    // If the tree query failed, spend only the caller's remaining slice on an
    // exact root-generation read; never fall back to readProcessCreatedAt's
    // independent two-second default and overrun the absolute deadline.
    const elapsed = Date.now() - startedAt
    const fallbackBudget = Math.max(0, Math.min(budget, elapsed >= budget ? 0 : budget - elapsed))

    if (fallbackBudget <= 0) {return [{ pid: rootPid }]}
    const createdAt = await readProcessCreatedAt(rootPid, fallbackBudget)

    return createdAt == null ? [{ pid: rootPid }] : [{ pid: rootPid, createdAt }]
  }
}

