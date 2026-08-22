/**
 * Fail-closed lifecycle control for Desktop-owned backend execution scopes.
 *
 * A numeric PID is observation, never destructive authority. Electron signals
 * only the retained ChildProcess object. Platform authority installed before
 * `hermes_cli.main` expands that retained-root signal to the complete owned
 * scope: a Windows Job Object or a POSIX session supervisor.
 */

export const STOP_REQUESTED = 'StopRequested' as const
export const STOP_EXITED = 'Exited' as const
export const STOP_ALREADY_EXITED = 'AlreadyExited' as const
export const STOP_NO_AUTHORITY = 'NoAuthority' as const
export const STOP_PERMISSION_DENIED = 'PermissionDenied' as const
export const STOP_TIMED_OUT = 'TimedOut' as const

export type BackendStopKind =
  | typeof STOP_REQUESTED
  | typeof STOP_EXITED
  | typeof STOP_ALREADY_EXITED
  | typeof STOP_NO_AUTHORITY
  | typeof STOP_PERMISSION_DENIED
  | typeof STOP_TIMED_OUT

export interface BackendStopResult {
  readonly kind: BackendStopKind
  readonly pid?: number | null
  readonly exitCode?: number | null
  readonly signalCode?: string | null
  readonly detail?: string
}

export class BackendStopError extends Error {
  readonly result: BackendStopResult

  constructor(operation: string, result: BackendStopResult) {
    super(`${operation} failed with ${result.kind}${result.detail ? `: ${result.detail}` : ''}`)
    this.name = 'BackendStopError'
    this.result = result
  }
}

export interface BackendProcessRoot {
  pid?: number | null
  exitCode?: null | number
  signalCode?: null | string
}

export interface KillableChild extends BackendProcessRoot {
  killed?: boolean
  kill: (signal?: NodeJS.Signals | number | null) => unknown
  once?: (event: 'exit', listener: (code: number | null, signal: string | null) => void) => unknown
  removeListener?: (event: 'exit', listener: (...args: any[]) => void) => unknown
}

export interface StopBackendTreesForUpdateDeps {
  /** Stops pooled backends through their retained ChildProcess owners. */
  stopAllPoolBackends: () => Promise<void> | void
}

/**
 * A forced stop is not terminal proof. Keep the exact retained owners whose
 * force request has been submitted but whose exit has not yet been observed.
 * New backend generations must not start across this barrier: otherwise a
 * caller that times out waiting for teardown can race a replacement backend
 * against a still-live prior generation.
 */
const pendingForcedBackendStops = new Set<KillableChild>()

function snapshot(child: KillableChild | null | undefined): Omit<BackendStopResult, 'kind'> {
  return child
    ? {
        pid: child.pid,
        exitCode: child.exitCode,
        signalCode: child.signalCode
      }
    : {}
}

/** Missing lifecycle fields mean no authority, never a legacy fallback. */
export function isLiveProcessRoot(root: BackendProcessRoot | null | undefined): boolean {
  return Boolean(
    root &&
      Number.isInteger(root.pid) &&
      (root.pid as number) > 0 &&
      root.exitCode === null &&
      root.signalCode === null
  )
}

function armForcedStopBarrier(child: KillableChild | null | undefined): void {
  if (!child || typeof child.once !== 'function' || !isLiveProcessRoot(child)) {
    return
  }

  pendingForcedBackendStops.add(child)
  child.once('exit', () => {
    pendingForcedBackendStops.delete(child)
  })
}

/**
 * Fail closed at the generation boundary until every forced retained owner has
 * supplied terminal proof. The lifecycle caller may have exhausted its local
 * wait budget, but that never authorizes a replacement generation to start
 * while the prior owner remains live.
 */
export function assertNoPendingForcedBackendStops(): void {
  for (const child of pendingForcedBackendStops) {
    if (!isLiveProcessRoot(child)) {
      pendingForcedBackendStops.delete(child)
    }
  }

  const pending = pendingForcedBackendStops.values().next().value as KillableChild | undefined

  if (!pending) {
    return
  }

  throw new BackendStopError('backend generation barrier', {
    kind: STOP_TIMED_OUT,
    detail: 'prior forced backend stop has not produced terminal exit proof',
    ...snapshot(pending)
  })
}

function signalRetainedChild(
  child: KillableChild | null | undefined,
  signal: NodeJS.Signals
): BackendStopResult {
  if (!child) {
    return { kind: STOP_ALREADY_EXITED }
  }
  if (typeof child.kill !== 'function') {
    return { kind: STOP_NO_AUTHORITY, ...snapshot(child) }
  }
  if (child.exitCode != null || child.signalCode != null) {
    return { kind: STOP_ALREADY_EXITED, ...snapshot(child) }
  }
  if (!isLiveProcessRoot(child)) {
    return { kind: STOP_NO_AUTHORITY, ...snapshot(child) }
  }

  try {
    if (child.kill(signal) === false) {
      return {
        kind: STOP_PERMISSION_DENIED,
        detail: `retained ChildProcess refused ${signal}`,
        ...snapshot(child)
      }
    }
    // Signal submission is not exit observation. The retained owner remains
    // live until its exit event or populated exit fields prove otherwise.
    return {
      kind: STOP_REQUESTED,
      detail: `submitted ${signal} to retained owner`,
      ...snapshot(child)
    }
  } catch (error) {
    return {
      kind: STOP_PERMISSION_DENIED,
      detail: error instanceof Error ? error.message : String(error),
      ...snapshot(child)
    }
  }
}

function requireStopSubmission(operation: string, result: BackendStopResult): BackendStopResult {
  if (result.kind === STOP_NO_AUTHORITY || result.kind === STOP_PERMISSION_DENIED) {
    // Existing lifecycle call sites may ignore a compatibility return value,
    // but they can no longer silently discard a hard authority failure.
    throw new BackendStopError(operation, result)
  }
  return result
}

/** Graceful stop through the retained owner only. */
export function requestBackendGracefulStop(
  child: KillableChild | null | undefined
): BackendStopResult {
  return signalRetainedChild(child, 'SIGTERM')
}

/**
 * Forced stop through the same retained owner. POSIX uses SIGUSR2 as the
 * supervisor's non-PID force command; Windows uses SIGKILL, whose root exit
 * closes the generation-bound Job and reaps descendants.
 */
export function requestBackendForceStop(
  child: KillableChild | null | undefined,
  platform = process.platform
): BackendStopResult {
  const signal: NodeJS.Signals = platform === 'win32' ? 'SIGKILL' : 'SIGUSR2'
  return signalRetainedChild(child, signal)
}

/** Compatibility entry point that preserves typed outcomes and fails loudly. */
export function stopBackendChild(
  child: KillableChild | null | undefined
): BackendStopResult {
  return requireStopSubmission('graceful backend stop', requestBackendGracefulStop(child))
}

/** Compatibility entry point that preserves typed outcomes and fails loudly. */
export function forceStopBackendChild(
  child: KillableChild | null | undefined,
  platform = process.platform
): BackendStopResult {
  const result = requireStopSubmission('forced backend stop', requestBackendForceStop(child, platform))

  if (result.kind === STOP_REQUESTED) {
    armForcedStopBarrier(child)
  }

  return result
}

function waitForExit(
  child: KillableChild | null | undefined,
  timeoutMs: number
): Promise<BackendStopResult> {
  if (!child) {
    return Promise.resolve({ kind: STOP_ALREADY_EXITED })
  }
  if (typeof child.once !== 'function') {
    return Promise.resolve({ kind: STOP_NO_AUTHORITY, ...snapshot(child) })
  }
  if (!isLiveProcessRoot(child)) {
    return Promise.resolve({ kind: STOP_ALREADY_EXITED, ...snapshot(child) })
  }

  return new Promise(resolve => {
    const onExit = (code: number | null, signal: string | null) => {
      clearTimeout(timer)
      child.exitCode = code
      child.signalCode = signal
      resolve({ kind: STOP_EXITED, ...snapshot(child) })
    }
    const timer = setTimeout(() => {
      child.removeListener?.('exit', onExit)
      resolve({ kind: STOP_TIMED_OUT, ...snapshot(child) })
    }, Math.max(0, Math.min(Math.trunc(timeoutMs), 120_000)))
    child.once?.('exit', onExit)
  })
}

/** Graceful -> bounded wait -> force -> terminal confirmation. */
export async function stopBackendChildAndWait(
  child: KillableChild | null | undefined,
  options: { gracefulTimeoutMs?: number; forceTimeoutMs?: number; platform?: NodeJS.Platform } = {}
): Promise<BackendStopResult> {
  const graceful = requestBackendGracefulStop(child)
  if (graceful.kind !== STOP_REQUESTED) {
    return graceful
  }

  const gracefulExit = await waitForExit(child, options.gracefulTimeoutMs ?? 5_000)
  if (gracefulExit.kind !== STOP_TIMED_OUT) {
    return gracefulExit
  }

  const forced = requestBackendForceStop(child, options.platform ?? process.platform)
  if (forced.kind !== STOP_REQUESTED) {
    return forced
  }

  armForcedStopBarrier(child)
  return waitForExit(child, options.forceTimeoutMs ?? 2_000)
}

export async function stopBackendTreesForUpdate(
  primary: KillableChild | null | undefined,
  deps: StopBackendTreesForUpdateDeps
): Promise<BackendStopResult> {
  const primaryResult = stopBackendChild(primary)
  await deps.stopAllPoolBackends()
  return primaryResult
}
