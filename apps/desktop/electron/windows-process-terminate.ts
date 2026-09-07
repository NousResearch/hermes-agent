/**
 * SuperF4-style exact process termination for Windows update force-release.
 *
 * Behavioral reference (not copied): stefansundin/superf4 commit
 * 6b677d422553e6b908b9eeaff4333b8b457e7bef, superf4.c lines 236-289.
 * SuperF4 is GPL-3.0 — do not copy its implementation. This module follows
 * Microsoft Win32 documentation for:
 *   OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE)
 *   TerminateProcess
 *   WaitForSingleObject
 *   CloseHandle
 * and optionally AdjustTokenPrivileges(SeDebugPrivilege) when present.
 *
 * The PowerShell child that performs TerminateProcess is itself bounded by a
 * hard wall-clock budget. On expiry the child tree is killed and the call
 * returns only after the child PID is confirmed gone (or a Windows Job Object
 * KILL_ON_JOB_CLOSE terminal is applied). No late mutation after return.
 *
 * This module is the executor: it spawns, supervises and cancels. Its two
 * halves live next door — the embedded PowerShell/C# program text in
 * windows-process-terminate-scripts.ts, and the read-only liveness/identity
 * probes in windows-process-liveness.ts.
 */

import { type ChildProcess, execFile, spawn } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { installSharedRuntimeRoot } from './install-mutation-set'
import { windowsPowerShellExecutable, windowsSystem32Executable } from './windows-powershell-path'
import {
  identitiesStillPresent,
  type ProcessIdentity,
  readProcessCreatedAt,
  resolveBeforeDeadline,
  snapshotProcessTreeIdentities
} from './windows-process-liveness'
import {
  buildExactTerminateScript,
  type ExactTerminateScriptOptions,
  TERMINATE_JOB_WATCHER_BOOTSTRAP,
  TERMINATE_JOB_WATCHER_BRIDGE,
  TERMINATE_JOB_WATCHER_COMMAND,
  TERMINATE_JOB_WRAPPER_COMMAND,
  TERMINATE_NAMED_JOB_COMMAND
} from './windows-process-terminate-scripts'
import type { ForceReleaseHolder, ForceReleaseTerminateResult, WindowsUpdateForceReleaseDeps } from './windows-update-force-release'

const execFileAsync = promisify(execFile)

/**
 * The script text and the liveness probes moved into sibling modules. These
 * re-exports keep the module's published surface stable for callers and for
 * the live Windows suite that drives the boundary through this entry point.
 */
export {
  classifyLivenessProbeResult,
  identitiesStillPresent,
  type LivenessProbeResult,
  type LivenessProbeRunner,
  probeProcessLiveness,
  type ProcessLiveness,
  snapshotProcessTreeIdentities
} from './windows-process-liveness'
export {
  buildExactTerminateScript,
  TERMINATE_ACCESS_DENIED_CLASSIFIER,
  TERMINATE_JOB_WATCHER_BRIDGE,
  TERMINATE_JOB_WATCHER_COMMAND,
  TERMINATE_JOB_WRAPPER_COMMAND,
  TERMINATE_NAMED_JOB_COMMAND
} from './windows-process-terminate-scripts'
export type { ExactTerminateScriptOptions, ProcessIdentity }

export type StartTargetJobWatcher = (
  ownerPid: number,
  ownerCreatedAt: number,
  targetJobName: string,
  watcherReadyPath: string,
  watcherReadyNonce: string,
  deadlineAt: number
) => ChildProcess | undefined

export type HardBoundaryDependencies = {
  startWatcher?: StartTargetJobWatcher
}

const WRAPPER_MARKER_POLL_MS = 10

const WRAPPER_PHASE_NAMES = [
  'marker-published',
  'target-job-created',
  'target-boundary-armed',
  'inner-started',
  'inner-exited',
  'target-terminated',
  'finally-close',
  'handles-closed'
] as const

type TargetJobWatcherDiagnostics = {
  spawnedAt: number
  pid?: number
  errorAt?: number
  errorCode?: string | number
  errorMessage?: string
  exitAt?: number
  exitCode?: number | null
  signalCode?: NodeJS.Signals | null
}

export type RunPowerShell = (
  script: string,
  timeoutMs?: number,
  signal?: AbortSignal,
  deadlineAt?: number,
  dependencies?: HardBoundaryDependencies
) => Promise<{ stdout: string; stderr: string; code: number; pid?: number }>

/** Shared kill/identity-confirm reserve used by the terminate boundary. */
export const TERMINATE_KILL_CONFIRM_MS = 1_500
export const TERMINATE_KILL_CONFIRM_MIN_MS = 400
export const TERMINATE_KILL_CONFIRM_RATIO = 0.3

export function terminateKillReserveMs(budgetMs: number): number {
  const budget = Math.max(0, Math.trunc(budgetMs))

  if (budget <= 0) {return 0}

  return Math.min(
    budget,
    TERMINATE_KILL_CONFIRM_MS,
    Math.max(Math.min(TERMINATE_KILL_CONFIRM_MIN_MS, budget), Math.floor(budget * TERMINATE_KILL_CONFIRM_RATIO))
  )
}

export function parseWrapperProcessMarker(value: string, expectedNonce: string): ProcessIdentity | null {
  const match = /^PID:(\d+);CREATED_AT_MS:(\d+);NONCE:([A-Za-z0-9_-]+)$/.exec(String(value ?? '').trim())

  if (!match || match[3] !== expectedNonce) {return null}
  const pid = Number(match[1])
  const createdAtMs = Number(match[2])

  if (!Number.isSafeInteger(pid) || pid <= 0 || !Number.isSafeInteger(createdAtMs) || createdAtMs <= 0) {return null}

  return { pid, createdAt: createdAtMs / 1_000 }
}

async function waitForWrapperProcessMarker(
  markerPath: string,
  markerNonce: string,
  deadlineAt: number
): Promise<ProcessIdentity | null> {
  while (Date.now() < deadlineAt) {
    try {
      if (fs.existsSync(markerPath)) {
        const identity = parseWrapperProcessMarker(fs.readFileSync(markerPath, 'utf8'), markerNonce)

        if (identity && identity.createdAt != null) {return identity}
      }
    } catch {
      // A partial or stale marker is not an authenticated wrapper identity.
    }

    const remaining = Math.max(0, deadlineAt - Date.now())

    if (remaining <= 0) {break}
    await new Promise(resolve => setTimeout(resolve, Math.min(WRAPPER_MARKER_POLL_MS, remaining)))
  }

  return null
}

function readWrapperPhaseDiagnostics(phasePath: string, phaseNonce: string): string {
  const summary = WRAPPER_PHASE_NAMES.map(phase => {
    const markerPath = `${phasePath}.${phase}.marker`
    const expectedPrefix = `PHASE:${phaseNonce};NAME:${phase};TICKS:`

    try {
      if (!fs.existsSync(markerPath)) {return `${phase}=missing`}
      const value = fs.readFileSync(markerPath, 'utf8').trim().replace(/[\r\n]+/g, ' ')

      return value.startsWith(expectedPrefix) ? `${phase}=${value.slice(0, 512)}` : `${phase}=invalid`
    } catch {
      return `${phase}=unreadable`
    }
  })

  return `wrapper-phases ${summary.join(' ')}`.slice(0, 4_096)
}

function hasSuccessfulWrapperReceipt(phasePath: string, phaseNonce: string): boolean {
  const readPhase = (phase: (typeof WRAPPER_PHASE_NAMES)[number]): string | undefined => {
    try {
      const value = fs.readFileSync(`${phasePath}.${phase}.marker`, 'utf8').trim()
      const expectedPrefix = `PHASE:${phaseNonce};NAME:${phase};TICKS:`

      return value.startsWith(expectedPrefix) ? value : undefined
    } catch {
      return undefined
    }
  }

  const innerExited = readPhase('inner-exited')

  return Boolean(
    innerExited?.endsWith(';DETAIL:code=0') &&
      readPhase('target-terminated') &&
      readPhase('handles-closed')
  )
}

function hasTargetBoundaryReceipt(phasePath: string, phaseNonce: string): boolean {
  const hasPhase = (phase: 'target-terminated' | 'handles-closed'): boolean => {
    try {
      const value = fs.readFileSync(`${phasePath}.${phase}.marker`, 'utf8').trim()

      return value.startsWith(`PHASE:${phaseNonce};NAME:${phase};TICKS:`)
    } catch {
      return false
    }
  }

  return hasPhase('target-terminated') && hasPhase('handles-closed')
}

function readExactWatcherReadyValue(watcherReadyPath: string): string | undefined {
  try {
    if (!fs.existsSync(watcherReadyPath)) {return undefined}

    return fs.readFileSync(watcherReadyPath, 'utf8').trim().replace(/[\r\n]+/g, ' ').slice(0, 512)
  } catch {
    return undefined
  }
}

async function runTaskkillWithinDeadline(
  pid: number,
  deadlineAt: number
): Promise<{ completed: boolean; succeeded: boolean }> {
  const remaining = Math.max(0, deadlineAt - Date.now())

  if (remaining <= 0) {return { completed: false, succeeded: false }}

  return await new Promise(resolve => {
    let settled = false
    let timer: NodeJS.Timeout | undefined

    const finish = (result: { completed: boolean; succeeded: boolean }) => {
      if (settled) {return}
      settled = true

      if (timer) {clearTimeout(timer)}
      resolve(result)
    }

    const child = execFile(
      // Absolute path. PATH is attacker-influenced, and during an update the
      // venv's own Scripts directory is prepended to it and rewritten in flight.
      windowsSystem32Executable('taskkill.exe'),
      ['/PID', String(pid), '/T', '/F'],
      {
        windowsHide: true,
        timeout: Math.max(1, remaining - 1),
        killSignal: 'SIGKILL'
      },
      (error: any) => finish({ completed: true, succeeded: !error })
    )

    child.once('error', () => finish({ completed: true, succeeded: false }))
    timer = setTimeout(() => {
      // This command only supervises the already-contained mutation child.
      // If taskkill itself cannot settle in time, stop waiting and fail closed;
      // never report a confirmed tree after an unbounded native command.
      try {
        child.kill('SIGKILL')
      } catch {
        void 0
      }

      finish({ completed: false, succeeded: false })
    }, Math.max(1, remaining - 1))
  })
}

/**
 * Kill a Windows process tree and wait until every pre-captured identity is gone
 * or reused. Uses one absolute deadline shared by taskkill + identity polling.
 * Returns confirmed=false when any identity remains — callers must treat that as
 * a hard terminal-boundary failure, not a settled success.
 */
export async function killProcessTreeAndAwaitGone(
  pid: number,
  {
    confirmMs = TERMINATE_KILL_CONFIRM_MS,
    pollMs = 50,
    preSnapshot,
    deadlineAt,
    snapshotProcessTree,
    readCreatedAt
  }: {
    confirmMs?: number
    pollMs?: number
    preSnapshot?: readonly ProcessIdentity[]
    /** Absolute Date.now() deadline; overrides confirmMs window when provided. */
    deadlineAt?: number
    /** Injectable only to force snapshot failure in the real boundary canary. */
    snapshotProcessTree?: typeof snapshotProcessTreeIdentities
    /** Injectable create-time reader for deadline regressions. */
    readCreatedAt?: (pid: number, timeoutMs: number) => Promise<number | null>
  } = {}
): Promise<{ confirmed: boolean; identities: ProcessIdentity[]; survivors: ProcessIdentity[] }> {
  if (!Number.isInteger(pid) || pid <= 0) {
    return { confirmed: true, identities: [], survivors: [] }
  }

  const absoluteDeadline =
    typeof deadlineAt === 'number' && Number.isFinite(deadlineAt)
      ? deadlineAt
      : Date.now() + Math.max(0, Math.trunc(confirmMs))

  const remaining = () => Math.max(0, absoluteDeadline - Date.now())

  if (remaining() <= 0) {
    // No time for probes: fail closed with known/unknown identities as survivors.
    const identities =
      preSnapshot && preSnapshot.length > 0 ? [...preSnapshot] : [{ pid }]

    return { confirmed: false, identities, survivors: identities }
  }

  const snapshot = snapshotProcessTree ?? snapshotProcessTreeIdentities
  let captured: ProcessIdentity[] = []
  let snapshotTimedOut = false

  if (preSnapshot && preSnapshot.length > 0) {
    captured = [...preSnapshot]
  } else {
    const snapshotBudget = remaining()

    if (snapshotBudget <= 0) {
      return { confirmed: false, identities: [{ pid }], survivors: [{ pid }] }
    }

    const snapshotPromise = Promise.resolve().then(() =>
      snapshot(pid, {
        timeoutMs: Math.min(500, snapshotBudget),
        deadlineAt: absoluteDeadline
      })
    )

    let snapshotTimer: NodeJS.Timeout | undefined

    try {
      // Native execFile has its own timeout, but keep the boundary safe even
      // if an injected/native adapter ignores that option. Snapshotting is
      // read-only, so abandoning its late result cannot leave mutation work.
      captured = await Promise.race([
        snapshotPromise,
        new Promise<ProcessIdentity[]>(resolve => {
          snapshotTimer = setTimeout(() => {
            snapshotTimedOut = true
            resolve([])
          }, Math.max(1, snapshotBudget - 1))
        })
      ])
    } catch {
      // Snapshot failure is intentionally represented by the unknown root.
      // The caller's Job Object boundary, when present, is the hard fallback.
      captured = []
    } finally {
      if (snapshotTimer) {clearTimeout(snapshotTimer)}
    }

    // A timed-out adapter may settle later; consume its rejection without
    // extending this boundary or creating an unhandled-rejection side effect.
    void snapshotPromise.catch(() => undefined)

    if (snapshotTimedOut) {
      // The shared deadline was consumed by snapshotting. Do not launch a
      // second native probe for a fabricated/fresh root generation.
      captured = []
    }
  }

  const identities = captured
  const createdAtReader = readCreatedAt ?? readProcessCreatedAt

  // Always include the root identity. Prefer a real create-time; if unavailable,
  // keep an unknown-generation identity (createdAt omitted) so liveness alone
  // keeps confirmation false while the PID lives.
  if (!identities.some(entry => entry.pid === pid)) {
    const left = remaining()
    let createdAt: number | null = null

    if (!snapshotTimedOut && left > 0) {
      const rootResult = await resolveBeforeDeadline(
        () => createdAtReader(pid, left),
        absoluteDeadline
      )

      if (rootResult.completed) {createdAt = rootResult.value}
    }

    identities.push(createdAt != null ? { pid, createdAt } : { pid })
  }

  const taskkillBudget = Math.min(remaining(), Math.max(0, Math.trunc(confirmMs)))

  if (taskkillBudget > 0) {
    const taskkillResult = await runTaskkillWithinDeadline(pid, absoluteDeadline)

    if (!taskkillResult.completed) {
      return { confirmed: false, identities, survivors: identities }
    }

    if (!taskkillResult.succeeded) {
      try {
        process.kill(pid, 'SIGKILL')
      } catch {
        void 0
      }
    }
  }

  // Also hard-kill every known identity in case /T missed a detached Start-Process child.
  for (const identity of identities) {
    const left = remaining()

    if (left <= 0) {break}

    if (identity.pid === pid) {continue}
    const taskkillResult = await runTaskkillWithinDeadline(identity.pid, absoluteDeadline)

    if (!taskkillResult.completed) {
      return { confirmed: false, identities, survivors: identities }
    }

    if (!taskkillResult.succeeded) {
      try {
        process.kill(identity.pid, 'SIGKILL')
      } catch {
        void 0
      }
    }
  }

  if (remaining() <= 0) {
    return { confirmed: false, identities, survivors: identities }
  }

  let survivors = await identitiesStillPresent(identities, { deadlineAt: absoluteDeadline })

  while (survivors.length > 0 && remaining() > 0) {
    await new Promise(resolve => setTimeout(resolve, Math.max(1, Math.min(pollMs, remaining()))))

    if (remaining() <= 0) {break}
    survivors = await identitiesStillPresent(identities, { deadlineAt: absoluteDeadline })
  }

  if (remaining() <= 0 && survivors.length > 0) {
    return { confirmed: false, identities, survivors }
  }

  return {
    confirmed: survivors.length === 0,
    identities,
    survivors
  }
}

function waitForChildExit(child: ChildProcess, timeoutMs: number): Promise<boolean> {
  if (child.exitCode != null || child.signalCode != null) {
    return Promise.resolve(true)
  }

  return new Promise(resolve => {
    let settled = false

    const done = (ok: boolean) => {
      if (settled) {return}
      settled = true
      resolve(ok)
    }

    const timer = setTimeout(() => done(false), Math.max(1, timeoutMs))
    child.once('exit', () => {
      clearTimeout(timer)
      done(true)
    })
    child.once('error', () => {
      clearTimeout(timer)
      done(false)
    })
  })
}

async function terminateNamedTargetJobWithinDeadline(
  targetJobName: string,
  deadlineAt: number,
  maxBudgetMs = 1_000
): Promise<boolean> {
  if (process.platform !== 'win32' || !targetJobName) {return false}
  const remaining = Math.max(0, deadlineAt - Date.now())
  const budget = Math.min(Math.max(0, Math.trunc(maxBudgetMs)), remaining)

  if (budget <= 0) {return false}

  try {
    await execFileAsync(
      windowsPowerShellExecutable(),
      [
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-Command',
        TERMINATE_NAMED_JOB_COMMAND
      ],
      {
        windowsHide: true,
        timeout: Math.max(1, budget - 1),
        env: {
          ...process.env,
          HERMES_TERMINATE_TARGET_JOB_NAME: targetJobName,
          HERMES_TERMINATE_TARGET_WAIT_MS: String(Math.max(0, budget - 100))
        }
      }
    )

    return true
  } catch {
    return false
  }
}

function startTargetJobWatcher(
  ownerPid: number,
  ownerCreatedAt: number,
  targetJobName: string,
  watcherReadyPath: string,
  watcherReadyNonce: string,
  deadlineAt: number
): ReturnType<StartTargetJobWatcher> {
  if (process.platform !== 'win32' || !Number.isInteger(ownerPid) || ownerPid <= 0) {return undefined}
  const encodedWatcherCommand = Buffer.from(TERMINATE_JOB_WATCHER_BOOTSTRAP, 'utf16le').toString('base64')

  try {
    const watcher = spawn(
      process.execPath,
      ['-e', TERMINATE_JOB_WATCHER_BRIDGE],
      {
        windowsHide: true,
        detached: true,
        stdio: 'ignore',
        env: {
          ...process.env,
          ELECTRON_RUN_AS_NODE: '1',
          HERMES_TERMINATE_WATCHER_POWERSHELL: windowsPowerShellExecutable(),
          HERMES_TERMINATE_WATCHER_ENCODED_COMMAND: encodedWatcherCommand,
          HERMES_TERMINATE_OWNER_PID: String(ownerPid),
          HERMES_TERMINATE_OWNER_CREATED_AT: String(ownerCreatedAt),
          HERMES_TERMINATE_TARGET_JOB_NAME: targetJobName,
          HERMES_TERMINATE_WATCHER_READY_PATH: watcherReadyPath,
          HERMES_TERMINATE_WATCHER_READY_NONCE: watcherReadyNonce,
          // The watcher body was originally authored for direct Windows argv
          // transport, where escaped C# quotes are consumed before PowerShell
          // parses the here-string. ScriptBlock/environment transport preserves
          // those backslashes, so normalize only escaped double quotes here.
          HERMES_TERMINATE_WATCHER_SCRIPT: TERMINATE_JOB_WATCHER_COMMAND.replace(/\\"/g, '"'),
          HERMES_TERMINATE_WATCHER_DEADLINE_AT: String(Math.trunc(deadlineAt))
        }
      }
    )

    return watcher
  } catch {
    return undefined
  }
}

function boundedWatcherErrorMessage(message: unknown): string {
  return String(message ?? '')
    .replace(/[\r\n]+/g, ' ')
    .slice(0, 256)
}

function observeTargetJobWatcher(child: ChildProcess, spawnedAt: number): TargetJobWatcherDiagnostics {
  const diagnostics: TargetJobWatcherDiagnostics = {
    spawnedAt,
    pid: child.pid
  }

  child.on('error', error => {
    if (diagnostics.errorAt == null) {diagnostics.errorAt = Date.now()}

    if (diagnostics.errorCode == null) {
      const errorCode = (error as NodeJS.ErrnoException).code

      if (typeof errorCode === 'string' || typeof errorCode === 'number') {diagnostics.errorCode = errorCode}
    }

    if (!diagnostics.errorMessage) {diagnostics.errorMessage = boundedWatcherErrorMessage(error?.message)}
  })
  child.on('exit', (code, signal) => {
    if (diagnostics.exitAt == null) {diagnostics.exitAt = Date.now()}
    diagnostics.exitCode = code
    diagnostics.signalCode = signal
  })

  if (child.exitCode != null || child.signalCode != null) {
    diagnostics.exitAt = Date.now()
    diagnostics.exitCode = child.exitCode
    diagnostics.signalCode = child.signalCode
  }

  return diagnostics
}

function formatTargetJobWatcherDiagnostics(diagnostics?: TargetJobWatcherDiagnostics): string {
  if (!diagnostics) {return 'watcher-diagnostics=none'}
  const now = Date.now()
  const errorElapsedMs = diagnostics.errorAt == null ? 'none' : Math.max(0, diagnostics.errorAt - diagnostics.spawnedAt)
  const exitElapsedMs = diagnostics.exitAt == null ? 'none' : Math.max(0, diagnostics.exitAt - diagnostics.spawnedAt)

  const elapsedMs = diagnostics.exitAt != null
    ? exitElapsedMs
    : diagnostics.errorAt != null
      ? errorElapsedMs
      : Math.max(0, now - diagnostics.spawnedAt)

  return [
    `watcher-spawned-at=${diagnostics.spawnedAt}`,
    `watcher-pid=${diagnostics.pid ?? 'none'}`,
    `watcher-error-code=${diagnostics.errorCode ?? 'none'}`,
    `watcher-error-message=${JSON.stringify(diagnostics.errorMessage ?? '')}`,
    `watcher-exit-code=${diagnostics.exitCode ?? 'none'}`,
    `watcher-signal=${diagnostics.signalCode ?? 'none'}`,
    `watcher-error-elapsed-ms=${errorElapsedMs}`,
    `watcher-exit-elapsed-ms=${exitElapsedMs}`,
    `watcher-elapsed-ms=${elapsedMs}`
  ].join(' ')
}

function sanitizeBoundaryDiagnostics(value: unknown): string {
  const safeLines: string[] = []

  for (const rawLine of String(value ?? '').split(/\r?\n/)) {
    const line = rawLine.trim()

    if (!line || /Command failed:/i.test(line)) {continue}

    if (/watcher-(?:spawned|exited|stalled|error|exit|signal|elapsed)/i.test(line)) {
      safeLines.push(line.slice(0, 1_200))

      continue
    }

    if (/^(?:TARGET_[A-Z0-9_ -]+|unconfirmed-tree-survivors:[0-9,]+|exit -?\d+|timeout|killed|aborted|deadline-exhausted)\b/i.test(line)) {
      safeLines.push(line.slice(0, 512))
    }
  }

  return [...new Set(safeLines)].slice(0, 8).join('\n')
}

function deadlineFailureDetail(stderr: unknown): string {
  const diagnostics = sanitizeBoundaryDiagnostics(stderr)

  return diagnostics ? `deadline-exhausted ${diagnostics}` : 'deadline-exhausted'
}

function sanitizeRunnerStderr(value: unknown): string {
  const stderr = String(value ?? '').replace(/\0/g, '')

  if (/Command failed:/i.test(stderr)) {return sanitizeBoundaryDiagnostics(stderr)}

  return stderr.slice(0, 4_096)
}

const TARGET_JOB_WATCHER_GRACE_MS = 400

type TargetJobWatcherResult = {
  terminal: boolean
  healthy: boolean
  targetBoundaryConfirmed: boolean
  detail: string
}

/**
 * Wait for the detached target-job watcher without allowing it to become an
 * unobserved mutator. A watcher that exits unsuccessfully or misses its short
 * settle window is killed and the named target job is terminated within the
 * same absolute deadline. The caller must not remove watcher artifacts until
 * this function reports terminal=true.
 */
async function waitForTargetJobWatcher(
  watcher: ChildProcess | undefined,
  targetJobName: string,
  deadlineAt: number,
  startupFailure?: string,
  diagnostics?: TargetJobWatcherDiagnostics,
  skipGrace = false
): Promise<TargetJobWatcherResult> {
  if (!watcher || !Number.isInteger(watcher.pid) || (watcher.pid as number) <= 0) {
    const remaining = Math.max(0, deadlineAt - Date.now())

    const targetJobAttempt =
      remaining > 0
        ? await terminateNamedTargetJobWithinDeadline(targetJobName, deadlineAt, Math.min(1_000, remaining))
        : false

    return {
      terminal: true,
      healthy: false,
      targetBoundaryConfirmed: targetJobAttempt,
      detail: `${startupFailure ?? 'watcher-missing'} ${formatTargetJobWatcherDiagnostics(diagnostics)} target-job-termination=${targetJobAttempt ? 'confirmed' : 'not-confirmed'}`
    }
  }

  const watcherPid = watcher.pid
  const settleBudget = skipGrace ? 0 : Math.min(TARGET_JOB_WATCHER_GRACE_MS, Math.max(0, deadlineAt - Date.now()))

  if (settleBudget > 0 && (await waitForChildExit(watcher, settleBudget))) {
    if (watcher.exitCode === 0 && watcher.signalCode == null) {
      return {
        terminal: true,
        healthy: true,
        targetBoundaryConfirmed: true,
        detail: `watcher-exited-cleanly ${formatTargetJobWatcherDiagnostics(diagnostics)}`
      }
    }

    const remaining = Math.max(0, deadlineAt - Date.now())

    const targetJobAttempt =
      remaining > 0
        ? await terminateNamedTargetJobWithinDeadline(targetJobName, deadlineAt, Math.min(1_000, remaining))
        : false

    return {
      terminal: true,
      healthy: false,
      targetBoundaryConfirmed: targetJobAttempt,
      detail: `watcher-exited-with-failure code=${String(watcher.exitCode)} signal=${String(watcher.signalCode)} ${formatTargetJobWatcherDiagnostics(diagnostics)} target-job-termination=${targetJobAttempt ? 'confirmed' : 'not-confirmed'}`
    }
  }

  // The watcher is stalled. Kill its exact process tree and close the named
  // target job in parallel, then observe the watcher exit before returning.
  const remaining = Math.max(0, deadlineAt - Date.now())

  const watcherTreeSnapshot =
    remaining > 0
      ? await snapshotProcessTreeIdentities(watcherPid, {
          timeoutMs: Math.min(300, remaining),
          deadlineAt
        })
      : []

  const targetJobTermination =
    remaining > 0
      ? terminateNamedTargetJobWithinDeadline(targetJobName, deadlineAt, Math.min(1_000, remaining))
      : Promise.resolve(false)

  try {
    watcher.kill('SIGKILL')
  } catch {
    void 0
  }

  const watcherTreeTermination = killProcessTreeAndAwaitGone(watcherPid, {
    confirmMs: Math.min(500, remaining),
    preSnapshot: watcherTreeSnapshot.length > 0 ? watcherTreeSnapshot : [{ pid: watcherPid }],
    deadlineAt
  })

  const [targetJobAttempt, watcherTreeResult] = await Promise.all([targetJobTermination, watcherTreeTermination])
  const remainingAfterKill = Math.max(0, deadlineAt - Date.now())

  const watcherExited =
    remainingAfterKill > 0
      ? await waitForChildExit(watcher, remainingAfterKill)
      : watcher.exitCode != null || watcher.signalCode != null

  return {
    // Native exact-generation absence is terminal proof even when Node has not
    // delivered the child `exit` event yet (for example while inherited stdio
    // handles finish closing).
    terminal: watcherExited || watcherTreeResult.confirmed,
    healthy: false,
    targetBoundaryConfirmed: targetJobAttempt,
    detail: `watcher-stalled tree-confirmed=${watcherTreeResult.confirmed} ${formatTargetJobWatcherDiagnostics(diagnostics)} target-job-termination=${targetJobAttempt ? 'confirmed' : 'not-confirmed'}`
  }
}

/**
 * Run PowerShell under a hard budget. On abort/timeout the child tree is killed
 * and the promise resolves only after every pre-captured tree identity is
 * confirmed gone. Confirmation failure is a hard boundary failure.
 */
export async function runPowerShellWithHardBoundary(
  script: string,
  timeoutMs = 4_000,
  signal?: AbortSignal,
  deadlineAt?: number,
  dependencies?: HardBoundaryDependencies
): Promise<{ stdout: string; stderr: string; code: number; pid?: number }> {
  const requestedBudget = Math.max(0, Math.trunc(timeoutMs))
  const startedAt = Date.now()
  // Leave a small scheduler/IPC margin so the public return stays inside the
  // caller's hard wall-clock budget even when native callbacks settle late.
  const deadlineSafetyMarginMs = Math.min(100, Math.floor(requestedBudget / 10))

  const requestedDeadline =
    typeof deadlineAt === 'number' && Number.isFinite(deadlineAt)
      ? Math.min(Math.trunc(deadlineAt), startedAt + requestedBudget)
      : startedAt + requestedBudget

  const absoluteDeadline = Math.max(startedAt, requestedDeadline - deadlineSafetyMarginMs)
  const budget = Math.max(0, absoluteDeadline - startedAt)

  if (budget <= 0) {
    return { stdout: '', stderr: 'aborted', code: 1 }
  }

  if (signal?.aborted) {
    return { stdout: '', stderr: 'aborted', code: 1 }
  }

  // Reserve part of the budget for kill + confirmed absence of the whole tree.
  const killReserveMs = terminateKillReserveMs(budget)
  const wrapperDeadline = absoluteDeadline - killReserveMs
  // Keep a bounded outer tail for one final PID-not-found proof when Node's exit
  // event lags behind a successfully delivered watcher kill.
  const watcherSettleReserveMs = Math.min(250, Math.floor(killReserveMs / 4))
  const watcherDeadline = absoluteDeadline - watcherSettleReserveMs
  const runMs = Math.max(0, wrapperDeadline - startedAt)

  if (runMs <= 0) {
    return { stdout: '', stderr: 'deadline-exhausted', code: 1 }
  }

  const jobName = `HermesTerminateHelper-${randomBytes(16).toString('hex')}`
  const targetJobName = `HermesTerminateTarget-${randomBytes(16).toString('hex')}`
  const targetWaitMs = Math.max(0, Math.min(1_500, runMs - 100))
  // Production relies on the wrapper-owned kill-on-close target/helper Jobs.
  // A watcher can be injected only by boundary tests; it is never an ambient
  // or production lifecycle dependency.
  const watcherStarter = dependencies?.startWatcher
  const watcherEnabled = typeof watcherStarter === 'function'
  const watcherReadyDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-terminate-watcher-'))
  const watcherReadyPath = path.join(watcherReadyDirectory, 'ready')
  const watcherReadyNonce = randomBytes(16).toString('hex')
  const wrapperPidMarkerPath = path.join(watcherReadyDirectory, 'wrapper.pid')
  const wrapperPidMarkerNonce = randomBytes(16).toString('hex')
  const wrapperPhasePath = path.join(watcherReadyDirectory, 'wrapper.phase')
  const wrapperPhaseNonce = randomBytes(16).toString('hex')
  const helperScriptPath = path.join(watcherReadyDirectory, 'helper.ps1')
  const helperGatePath = path.join(watcherReadyDirectory, 'helper.go')

  return await new Promise(resolve => {
    let settled = false
    let childPid: number | undefined
    let treeSnapshot: ProcessIdentity[] = []
    let killing = false
    let terminalizing: Promise<void> | null = null
    let childProcess: ChildProcess | undefined
    let wrapperProcessIdentity: ProcessIdentity | undefined
    let wrapperMarkerSetup: Promise<void> | undefined
    let watcherProcess: ChildProcess | undefined
    let watcherDiagnostics: TargetJobWatcherDiagnostics | undefined
    let watcherStartupFailure: string | undefined

    const cleanupWatcherArtifacts = () => {
      // Every writer path is parent-known and nonce-directory-bound. Never
      // recursively remove the directory, because unrelated files must not be swept.
      const exactPaths = [
        watcherReadyPath,
        `${watcherReadyPath}.tmp`,
        wrapperPidMarkerPath,
        `${wrapperPidMarkerPath}.tmp`,
        helperScriptPath,
        helperGatePath,
        ...WRAPPER_PHASE_NAMES.flatMap(phase => [
          `${wrapperPhasePath}.${phase}.marker`,
          `${wrapperPhasePath}.${phase}.marker.tmp`
        ])
      ].filter(
        (entry): entry is string => typeof entry === 'string'
      )

      for (const exactPath of exactPaths) {
        try {
          fs.rmSync(exactPath, { force: true })
        } catch {
          void 0
        }
      }

      try {
        fs.rmdirSync(watcherReadyDirectory)
      } catch {
        // A non-empty directory is intentionally retained rather than removed
        // recursively. The caller still receives the terminal failure.
        void 0
      }
    }

    const finish = async (result: { stdout: string; stderr: string; code: number; pid?: number }) => {
      if (settled || terminalizing) {return}
      terminalizing = (async () => {
        let finalResult = { ...result, pid: childPid }
        const mustKill = killing || signal?.aborted === true

        if (wrapperMarkerSetup) {await wrapperMarkerSetup}

        const watcherReadyAtFinish = watcherEnabled ? readExactWatcherReadyValue(watcherReadyPath) : undefined

        if (
          watcherEnabled &&
          watcherReadyAtFinish !== `ARMED:${watcherReadyNonce}` &&
          watcherProcess?.exitCode == null &&
          watcherProcess?.signalCode == null
        ) {
          try {
            watcherProcess?.kill('SIGKILL')
          } catch {
            void 0
          }
        }

        const watcherResult: TargetJobWatcherResult = watcherEnabled
          ? await waitForTargetJobWatcher(
              watcherProcess,
              targetJobName,
              absoluteDeadline,
              watcherStartupFailure,
              watcherDiagnostics,
              watcherReadyAtFinish !== `ARMED:${watcherReadyNonce}`
            )
          : {
              terminal: true,
              healthy: true,
              targetBoundaryConfirmed: hasTargetBoundaryReceipt(wrapperPhasePath, wrapperPhaseNonce),
              detail: 'wrapper-owned-job-boundary'
            }

        let targetBoundaryConfirmed = watcherResult.targetBoundaryConfirmed

        // On cancellation, drain the named target Job while the wrapper still
        // owns its persistent handle. Only then kill and confirm the wrapper /
        // helper process tree. This removes the close-before-open race.
        if (!watcherEnabled && mustKill && !targetBoundaryConfirmed && Date.now() < absoluteDeadline) {
          targetBoundaryConfirmed = await terminateNamedTargetJobWithinDeadline(
            targetJobName,
            absoluteDeadline,
            Math.min(1_000, Math.max(0, absoluteDeadline - Date.now()))
          )
        }

        let killResult =
          mustKill && typeof childPid === 'number'
            ? await killProcessTreeAndAwaitGone(childPid, {
                confirmMs: killReserveMs,
                preSnapshot: treeSnapshot,
                deadlineAt: absoluteDeadline
              })
            : undefined

        let watcherTerminal = watcherResult.terminal

        if (
          watcherEnabled &&
          !watcherTerminal &&
          Number.isInteger(watcherProcess?.pid) &&
          (watcherProcess?.pid as number) > 0 &&
          Date.now() < absoluteDeadline
        ) {
          const watcherSurvivors = await identitiesStillPresent([{ pid: watcherProcess?.pid as number }], {
            deadlineAt: absoluteDeadline
          })

          watcherTerminal = watcherSurvivors.length === 0
        }

        if (watcherEnabled && !watcherTerminal && watcherProcess) {
          watcherTerminal = watcherProcess.exitCode != null || watcherProcess.signalCode != null
        }

        let wrapperBoundaryRequired = mustKill

        // Only the injected legacy watcher can authenticate wrapper absence.
        // Production requires the normal wrapper-tree liveness proof.
        if (watcherEnabled && watcherResult.healthy && killResult && wrapperProcessIdentity) {
          const survivors = killResult.survivors.filter(identity => {
            if (identity.pid !== wrapperProcessIdentity?.pid) {return true}

            if (identity.createdAt == null || wrapperProcessIdentity.createdAt == null) {return false}

            return Math.abs(identity.createdAt - wrapperProcessIdentity.createdAt) > 1.5
          })

          killResult = { ...killResult, survivors, confirmed: survivors.length === 0 }
        }

        // execFile completion can lag exact process death. Accept only the
        // nonce-bound success receipt written after target drain and handle
        // closure; watcher READY is required only for an explicitly injected
        // watcher test path.
        if (
          finalResult.code !== 0 &&
          targetBoundaryConfirmed &&
          (!watcherEnabled || watcherReadyAtFinish === `ARMED:${watcherReadyNonce}`) &&
          hasSuccessfulWrapperReceipt(wrapperPhasePath, wrapperPhaseNonce)
        ) {
          finalResult = { stdout: 'TERMINATED\n', stderr: '', code: 0, pid: childPid }
        }

        if (!watcherResult.healthy || !targetBoundaryConfirmed) {
          wrapperBoundaryRequired = true
          killing = true
          const remaining = Math.max(0, absoluteDeadline - Date.now())

          if (!targetBoundaryConfirmed && remaining > 0) {
            targetBoundaryConfirmed = await terminateNamedTargetJobWithinDeadline(
              targetJobName,
              absoluteDeadline,
              Math.min(1_000, remaining)
            )
          }

          if (typeof childPid === 'number' && (!killResult || !killResult.confirmed)) {
            killResult = await killProcessTreeAndAwaitGone(childPid, {
              confirmMs: Math.min(killReserveMs, Math.max(0, absoluteDeadline - Date.now())),
              preSnapshot: treeSnapshot,
              deadlineAt: absoluteDeadline
            })
          }
        }

        if (watcherEnabled && !watcherResult.healthy) {
          finalResult = {
            ...finalResult,
            stderr: [watcherStartupFailure, watcherResult.detail, finalResult.stderr]
              .filter(Boolean)
              .join('\n'),
            code: 1,
            pid: childPid
          }
        }

        if (watcherEnabled && watcherDiagnostics && finalResult.code !== 0) {
          finalResult = {
            ...finalResult,
            stderr: [formatTargetJobWatcherDiagnostics(watcherDiagnostics), finalResult.stderr]
              .filter(Boolean)
              .join('\n'),
            pid: childPid
          }
        }

        if (!targetBoundaryConfirmed) {
          finalResult = {
            ...finalResult,
            stderr: [finalResult.stderr, 'target-boundary-unconfirmed'].filter(Boolean).join('\n'),
            code: 1,
            pid: childPid
          }
        }

        // Node's exit status is evidence for the exact child it spawned, even
        // when a fresh PowerShell liveness probe exhausts the remaining budget.
        // The target receipt separately proves the external holder job drained.
        if (
          killResult && !killResult.confirmed && targetBoundaryConfirmed &&
          childProcess && childProcess.pid === wrapperProcessIdentity?.pid &&
          (childProcess.exitCode != null || childProcess.signalCode != null) &&
          hasSuccessfulWrapperReceipt(wrapperPhasePath, wrapperPhaseNonce)
        ) {
          const survivors = killResult.survivors.filter(identity => identity.pid !== childProcess.pid)
          killResult = { ...killResult, survivors, confirmed: survivors.length === 0 }
        }

        if (killResult && !killResult.confirmed) {
          finalResult = {
            ...finalResult,
            stderr: [
              finalResult.stderr,
              `unconfirmed-tree-survivors:${killResult.survivors.map(s => s.pid).join(',')}`
            ]
              .filter(Boolean)
              .join('\n'),
            code: 1,
            pid: childPid
          }
        }

        if (!watcherTerminal) {
          finalResult = {
            ...finalResult,
            stderr: [finalResult.stderr, 'watcher-terminal-state-unconfirmed'].filter(Boolean).join('\n'),
            code: 1,
            pid: childPid
          }
        }

        // Native confirmation can observe the process object gone before Node
        // delivers its exit event. Drain that event under the same deadline.
        if (wrapperBoundaryRequired && childProcess && absoluteDeadline > Date.now()) {
          await waitForChildExit(childProcess, absoluteDeadline - Date.now())
        }

        // Capture the exact authenticated READY value and the ordered,
        // nonce-validated wrapper phases before any artifact cleanup. These
        // bounded diagnostics identify which side of the boundary stalled
        // without making cleanup depend on a wildcard directory scan.
        const wrapperPhaseDiagnostics = readWrapperPhaseDiagnostics(wrapperPhasePath, wrapperPhaseNonce)

        if (finalResult.code !== 0) {
          finalResult = {
            ...finalResult,
            stderr: [
              watcherEnabled
                ? watcherReadyAtFinish
                  ? 'watcher-ready=armed'
                  : 'watcher-ready=missing'
                : 'boundary=wrapper-owned-job',
              wrapperPhaseDiagnostics,
              finalResult.stderr
            ]
              .filter(Boolean)
              .join('\n'),
            pid: childPid
          }
        }

        // Do not remove READY or its PID-qualified temp file while the watcher
        // can still publish FAILED or terminate the target job.
        if (watcherTerminal) {cleanupWatcherArtifacts()}
        settled = true
        signal?.removeEventListener('abort', onAbort)
        resolve(finalResult)
      })().catch(error => {
        // Preserve the exact artifacts if watcher terminalization itself failed;
        // recursive cleanup here could permit a late watcher mutation.
        settled = true
        signal?.removeEventListener('abort', onAbort)
        resolve({
          stdout: result.stdout,
          stderr: `termination-boundary-error:${String(error?.message ?? error)}`,
          code: 1,
          pid: childPid
        })
      })
    }

    childProcess = execFile(
      windowsPowerShellExecutable(),
      [
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-Command',
        TERMINATE_JOB_WRAPPER_COMMAND
      ],
      {
        encoding: 'utf8',
        timeout: runMs,
        windowsHide: true,
        maxBuffer: 1024 * 1024,
        killSignal: 'SIGTERM',
        env: {
          ...process.env,
          HERMES_TERMINATE_SCRIPT: script,
          HERMES_TERMINATE_JOB_NAME: jobName,
          HERMES_TERMINATE_TARGET_JOB_NAME: targetJobName,
          HERMES_TERMINATE_TARGET_WAIT_MS: String(targetWaitMs),
          HERMES_TERMINATE_DEADLINE_AT: String(Math.trunc(wrapperDeadline)),
          HERMES_TERMINATE_WATCHER_READY_PATH: watcherReadyPath,
          HERMES_TERMINATE_WATCHER_READY_NONCE: watcherReadyNonce,
          HERMES_TERMINATE_WRAPPER_PID_MARKER_PATH: wrapperPidMarkerPath,
          HERMES_TERMINATE_WRAPPER_PID_MARKER_NONCE: wrapperPidMarkerNonce,
          HERMES_TERMINATE_WRAPPER_PHASE_PATH: wrapperPhasePath,
          HERMES_TERMINATE_WRAPPER_PHASE_NONCE: wrapperPhaseNonce,
          HERMES_TERMINATE_HELPER_SCRIPT_PATH: helperScriptPath,
          HERMES_TERMINATE_HELPER_GATE_PATH: helperGatePath
        }
      },
      (error: any, stdout, stderr) => {
        void (async () => {
          const errorStdout = String(stdout ?? error?.stdout ?? '')
          const capturedStderr = sanitizeRunnerStderr(stderr ?? error?.stderr ?? '')
          const errorMessage = String(error?.message ?? '')

          const lifecycleFact =
            error?.killed === true || /ETIMEDOUT|timeout|killed/i.test(errorMessage) ? 'timeout' : ''

          const errorStderr = [
            capturedStderr,
            typeof error?.code === 'number' ? `exit ${error.code}` : '',
            lifecycleFact
          ]
            .filter(Boolean)
            .join('\n')

          if (!error) {
            await finish({ stdout: String(stdout ?? ''), stderr: String(stderr ?? ''), code: 0 })

            return
          }

          if (
            typeof childPid === 'number' &&
            (error?.killed || /ETIMEDOUT|timeout/i.test(String(error?.message ?? '')))
          ) {
            killing = true
            await finish({
              stdout: errorStdout,
              stderr: errorStderr || 'timeout',
              code: typeof error?.code === 'number' ? error.code : 1
            })

            return
          }

          await finish({
            stdout: errorStdout,
            stderr: errorStderr,
            code: typeof error?.code === 'number' ? error.code : 1
          })
        })()
      }
    )

    if (typeof childProcess.pid === 'number' && childProcess.pid > 0) {
      childPid = childProcess.pid
      const launchedPid = childPid
      // The execFile PID is only a launch hint. The wrapper writes an exact,
      // nonce-bound self marker before opening the target job; authenticate that
      // generation before starting the detached watcher.
      wrapperMarkerSetup = (async () => {
        wrapperProcessIdentity = await waitForWrapperProcessMarker(
          wrapperPidMarkerPath,
          wrapperPidMarkerNonce,
          wrapperDeadline
        )

        if (!wrapperProcessIdentity) {
          watcherStartupFailure = `wrapper-marker-unavailable launch-pid=${launchedPid}`
          killing = true
          treeSnapshot = [{ pid: launchedPid }]

          try {
            childProcess?.kill('SIGKILL')
          } catch {
            void 0
          }

          return
        }

        if (watcherStarter) {
          const watcherSpawnedAt = Date.now()
          watcherProcess = watcherStarter(
            wrapperProcessIdentity.pid,
            wrapperProcessIdentity.createdAt ?? 0,
            targetJobName,
            watcherReadyPath,
            watcherReadyNonce,
            watcherDeadline
          )

          if (watcherProcess) {watcherDiagnostics = observeTargetJobWatcher(watcherProcess, watcherSpawnedAt)}

          if (!watcherProcess) {
            watcherStartupFailure = `watcher-spawn-failed wrapper-pid=${wrapperProcessIdentity.pid} launch-pid=${launchedPid}`
            killing = true

            return
          }
        }

        // Capture identities promptly after spawn with a bounded slice; abort must
        // not wait for a fresh snapshot before killing. Keep the authenticated
        // wrapper generation in the same confirmation set even when the launch
        // PID is a short-lived proxy.
        const snapshot = await snapshotProcessTreeIdentities(launchedPid, {
          timeoutMs: Math.min(400, killReserveMs)
        })

        // Never let the launch hint or an unknown-generation snapshot entry
        // replace the exact nonce-authenticated wrapper generation.
        treeSnapshot = [
          ...snapshot.filter(identity => identity.pid !== wrapperProcessIdentity?.pid),
          { pid: wrapperProcessIdentity.pid, createdAt: wrapperProcessIdentity.createdAt }
        ]
      })()
      // A marker failure kills the launch child; make sure the public promise
      // still enters terminalization even if execFile reports no callback.
      void wrapperMarkerSetup.then(() => {
        if (watcherStartupFailure && !settled && !terminalizing) {
          void finish({ stdout: '', stderr: watcherStartupFailure, code: 1, pid: childPid })
        }
      })
    }

    const onAbort = () => {
      if (settled || terminalizing) {return}
      killing = true

      if (typeof childPid === 'number') {
        // Kill immediately with already-captured identities (+ root). Never
        // delay kill for a fresh unbounded snapshot, and never synthesize
        // create-time. Unknown generation keeps confirmation fail-safe.
        if (treeSnapshot.length === 0) {
          treeSnapshot = [{ pid: childPid }]
        }
      }

      // finish() first drains the wrapper-owned named target Job, then kills and
      // confirms the authenticated wrapper/helper tree. Do not pre-kill the
      // wrapper here or its only persistent target-Job handle disappears before
      // the bounded drain can open it.
      void finish({ stdout: '', stderr: 'aborted', code: 1, pid: childPid })
    }

    if (signal) {
      if (signal.aborted) {
        onAbort()
      } else {
        signal.addEventListener('abort', onAbort, { once: true })
      }
    }
  })
}

export function parseTerminateScriptOutput(stdout: string, code: number): ForceReleaseTerminateResult {
  const text = String(stdout || '').trim()

  if (/PROTECTED/i.test(text)) {
    const win32 = text.match(/win32=(\d+)/i)

    return { kind: 'protected', win32Error: win32 ? Number(win32[1]) : 5 }
  }

  if (/ALREADY_GONE/i.test(text) || (code === 0 && /TERMINATED/i.test(text))) {
    if (/TERMINATED/i.test(text)) {return { kind: 'terminated' }}

    if (/ALREADY_GONE/i.test(text)) {return { kind: 'already-gone' }}
  }

  if (/CREATE_TIME_MISMATCH/i.test(text) || code === 3) {
    return { kind: 'create-time-mismatch' }
  }

  if (/ACCESS_DENIED/i.test(text)) {
    const marker = text.match(/ACCESS_DENIED(?:\s+([\s\S]*))?/i)
    const detail = marker?.[1]?.trim()

    return detail ? { kind: 'access-denied', win32Error: 5, detail } : { kind: 'access-denied', win32Error: 5 }
  }

  const win32 = text.match(/win32=(\d+)/i)

  if (win32) {
    const err = Number(win32[1])

    // Do not treat 6/87 as already-gone: the process was observed live above.
    return { kind: 'failed', detail: text || `win32=${err}`, win32Error: err }
  }

  if (code === 0 && /TERMINATED/i.test(text)) {return { kind: 'terminated' }}

  return { kind: 'failed', detail: text || `exit ${code}` }
}

export async function terminateWindowsHolderExact(
  target: ForceReleaseHolder,
  {
    platform = process.platform,
    run = runPowerShellWithHardBoundary,
    waitMs = 1_500,
    timeoutMs,
    signal,
    deadlineAt,
    installRoot,
    sharedRuntimeRoot,
    buildScript = buildExactTerminateScript
  }: {
    platform?: NodeJS.Platform
    run?: RunPowerShell
    waitMs?: number
    /** Hard wall-clock budget for the PowerShell child including kill/confirm. */
    timeoutMs?: number
    /** When aborted, kill the child tree, await confirmed absence, then return. */
    signal?: AbortSignal
    /** Absolute deadline shared with the caller's orchestration budget. */
    deadlineAt?: number
    /** Canonical update root for final resource-authorization validation. */
    installRoot?: string
    /** Tree inside the install that the mutation set excludes; see ExactTerminateScriptOptions. */
    sharedRuntimeRoot?: string
    /** Explicit test seam; production uses buildExactTerminateScript. */
    buildScript?: (
      pid: number,
      createdAt: number,
      waitMs: number,
      options?: ExactTerminateScriptOptions
    ) => string
  } = {}
): Promise<ForceReleaseTerminateResult> {
  if (platform !== 'win32') {
    return { kind: 'failed', detail: 'windows-only' }
  }

  if (signal?.aborted) {
    return { kind: 'failed', detail: 'deadline-exhausted' }
  }

  if (!Number.isInteger(target.pid) || target.pid <= 0) {
    return { kind: 'failed', detail: 'invalid pid' }
  }

  if (!Number.isFinite(target.createdAt) || target.createdAt <= 0) {
    return { kind: 'failed', detail: 'invalid createdAt' }
  }

  const requestedBudget = Math.max(0, Math.trunc(timeoutMs ?? Math.max(2_000, waitMs + 1_000)))

  const remainingBudget =
    typeof deadlineAt === 'number' && Number.isFinite(deadlineAt)
      ? Math.max(0, Math.trunc(deadlineAt - Date.now()))
      : requestedBudget

  const budget = Math.min(requestedBudget, remainingBudget)
  // Keep TerminateProcess wait short enough that kill-reserve still fits.
  const killReserveMs = terminateKillReserveMs(budget)
  const runBudget = Math.max(1, budget - killReserveMs)
  const effectiveWait = Math.max(0, Math.min(Math.trunc(waitMs), Math.max(0, runBudget - 250)))

  if (budget <= 50) {
    return { kind: 'failed', detail: 'deadline-exhausted' }
  }

  const script = buildScript(target.pid, target.createdAt, effectiveWait, {
    installRoot,
    resource: target.resource,
    sharedRuntimeRoot: sharedRuntimeRoot ?? (installRoot ? installSharedRuntimeRoot(installRoot) : undefined)
  })

  const result = await run(script, budget, signal, deadlineAt)

  if (signal?.aborted) {
    // Child tree must already be confirmed gone by run(); never claim mutation.
    return { kind: 'failed', detail: deadlineFailureDetail(result.stderr) }
  }

  if (/unconfirmed-tree-survivors/i.test(result.stderr || '')) {
    return { kind: 'failed', detail: sanitizeBoundaryDiagnostics(result.stderr) || 'unconfirmed-tree-survivors' }
  }

  // Timed-out/killed child: do not parse a partial TerminateProcess success.
  if (/aborted|ETIMEDOUT|timeout/i.test(result.stderr || '') && !/TERMINATED|ACCESS_DENIED|PROTECTED|CREATE_TIME/i.test(result.stdout || '')) {
    return { kind: 'failed', detail: deadlineFailureDetail(result.stderr) }
  }

  return parseTerminateScriptOutput(result.stdout + '\n' + result.stderr, result.code)
}

/**
 * Execute exact termination using the caller's remaining absolute deadline.
 * This is the production adapter used by the updater path; keeping the
 * deadline calculation here prevents a stale per-holder timeout from
 * extending the overall force-release contract.
 */
export async function terminateWindowsHolderWithinDeadline(
  target: ForceReleaseHolder,
  {
    platform = process.platform,
    run = runPowerShellWithHardBoundary,
    budgetMs,
    deadlineAt,
    installRoot,
    sharedRuntimeRoot,
    signal
  }: {
    platform?: NodeJS.Platform
    run?: RunPowerShell
    budgetMs: number
    deadlineAt: number
    installRoot?: string
    sharedRuntimeRoot?: string
    signal?: AbortSignal
  }
): Promise<ForceReleaseTerminateResult> {
  const requestedBudget = Math.max(0, Math.trunc(budgetMs))
  const absoluteDeadline = Number.isFinite(deadlineAt) ? Math.trunc(deadlineAt) : Date.now() + requestedBudget
  const remainingBudget = Math.max(0, absoluteDeadline - Date.now())
  const budget = Math.min(requestedBudget, remainingBudget)

  if (budget <= 50 || signal?.aborted) {
    return { kind: 'failed', detail: 'deadline-exhausted' }
  }

  return terminateWindowsHolderExact(target, {
    platform,
    run,
    timeoutMs: budget,
    waitMs: Math.max(0, Math.min(1_500, budget - 250)),
    signal,
    deadlineAt: absoluteDeadline,
    installRoot,
    ...(sharedRuntimeRoot ? { sharedRuntimeRoot } : {})
  })
}

/** Route plugin units through their owner and authorize every other holder by install root. */
export function createWindowsHolderTerminator(
  installRoot: string,
  stopPluginService: (service: NonNullable<ForceReleaseHolder['service']>) => Promise<boolean>,
  terminate = terminateWindowsHolderWithinDeadline
): WindowsUpdateForceReleaseDeps['terminateHolder'] {
  return async (holder, budgetMs, signal, deadlineAt) => {
    if (holder.terminateVia === 'desktop-plugin-service' && holder.service) {
      const stopped = await stopPluginService(holder.service)

      return stopped ? { kind: 'terminated' } : { kind: 'failed', detail: 'plugin service unit not stopped' }
    }

    return terminate(holder, {
      budgetMs,
      deadlineAt: deadlineAt ?? Date.now() + Math.max(0, budgetMs),
      signal,
      installRoot
    })
  }
}
