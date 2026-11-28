/**
 * Foreground default-plugins preflight before spawning a local Hermes backend.
 *
 * Runs `<backend python/runtime> -m hermes_cli.default_plugins ensure --json --timeout <secs>`
 * (or `<launcher> --run-module hermes_cli.default_plugins ensure --json --timeout <secs>` for
 * PM-managed CLI launchers) using the exact same runtime and environment as `runGatewayEnsure`.
 *
 * Bounded (default 120 s on primary boot; 5 s on pooled local dials so they stay well inside
 * renderer dial budgets), awaited before the local backend starts, and strictly fail-open: every
 * non-zero exit (2 = owed, 3 = timeout continuing in background, 4 = skipped by policy), timeout
 * kill, or interpreter crash is logged as one `[default-plugins]` line to desktop.log and startup
 * continues. Never runs for remote backends.
 */

import { spawn } from 'node:child_process'

import { mountFreeEnv } from './relaunch-target'
import { hiddenWindowsChildOptions, windowsShellCommand } from './windows-child-options'

export const DEFAULT_PLUGINS_PREFLIGHT_TIMEOUT_SECONDS = 120
// Pooled local dials (`dialPoolBackend`) have no pausable boot timer and are bounded by
// SOURCE_SWITCH_DIAL_TIMEOUT_MS (135s) / RECONNECT_ATTEMPT_TIMEOUT_MS (20s): 5s + 3s grace + 1s
// force-kill (9s ceiling) fits cleanly inside both alongside `runGatewayEnsure` (75s), while a
// still-installing detached child finishes in the background.
export const DEFAULT_PLUGINS_POOL_PREFLIGHT_TIMEOUT_SECONDS = 5
export const DEFAULT_PLUGINS_TIMEOUT_SECONDS = DEFAULT_PLUGINS_PREFLIGHT_TIMEOUT_SECONDS
export const DEFAULT_PLUGINS_LOG_PREFIX = '[default-plugins]'
export const DEFAULT_PLUGINS_BOOT_PHASE = 'backend.default-plugins'
export const DEFAULT_PLUGINS_BOOT_MESSAGE = 'Installing default plugins...'

const DEFAULT_PLUGINS_MODULE = 'hermes_cli.default_plugins'
const DEFAULT_PLUGINS_MAX_STDOUT_BYTES = 65_536
const DEFAULT_PLUGINS_MAX_STDERR_BYTES = 4_096
const DEFAULT_PLUGINS_MAX_ERROR_CHARS = 512
const DEFAULT_PLUGINS_FORCE_KILL_MS = 1_000

export interface PreflightBackend {
  kind: string
  command: string | null
  env?: NodeJS.ProcessEnv
  shell?: boolean
  /** Bundled payload interpreter (the CLI shim in `command` is not a Python interpreter). */
  ticketCommand?: string
  /** Bundled payload PYTHONPATH / PYTHONPYCACHEPREFIX paired with `ticketCommand`. */
  ticketEnv?: NodeJS.ProcessEnv
  /** Optional connection mode; any non-'local' value skips preflight. */
  mode?: string
}

export interface DefaultPluginsPreflightInvocation {
  command: string
  args: string[]
  env: NodeJS.ProcessEnv
  shell: boolean
  timeoutSeconds: number
}

export interface DefaultPluginsPreflightResult {
  skipped: boolean
  exitCode: number | null
  timedOut: boolean
  aborted: boolean
  elapsedMs: number
  payload: Record<string, unknown> | null
  error: string | null
  logLine: string | null
}

export interface RunDefaultPluginsPreflightOptions {
  cwd: string
  home: string
  parentEnv?: NodeJS.ProcessEnv
  execPath?: string
  log?: (line: string) => void
  signal?: AbortSignal
  /** Explicit timeout in seconds; defaults to 120. */
  timeoutSeconds?: number
  /** Extra grace beyond `--timeout` before Desktop sends SIGTERM so Python can emit exit-3 JSON. */
  killGraceMs?: number
  /** Wait after SIGTERM before escalating to SIGKILL and settling. */
  forceKillMs?: number
  /** Injectable spawn for unit tests. */
  spawnImpl?: typeof spawn
  /** Injectable wall clock for unit tests. */
  now?: () => number
  /** Injectable platform flag for unit tests. */
  isWindows?: boolean
}

export function resolveDefaultPluginsTimeoutSeconds(explicitSeconds?: number): number {
  if (typeof explicitSeconds === 'number' && Number.isFinite(explicitSeconds) && explicitSeconds > 0) {
    return explicitSeconds
  }

  return DEFAULT_PLUGINS_PREFLIGHT_TIMEOUT_SECONDS
}

export function isLocalPreflightBackend(
  backend: PreflightBackend | null | undefined
): backend is PreflightBackend & { command: string } {
  if (!backend) {
    return false
  }

  if (backend.mode && backend.mode !== 'local') {
    return false
  }

  if (backend.kind === 'bootstrap-needed') {
    return false
  }

  return Boolean(backend.ticketCommand || backend.command)
}

/**
 * Resolve the exact command, argv, shell flag and environment used to run the
 * default-plugins ensure preflight for a local backend descriptor.
 *
 * Mirrors `runGatewayEnsure` env construction (`mountFreeEnv`, `HERMES_HOME`,
 * stripping `HERMES_DESKTOP_DEV_SERVER`, `windowsShellCommand` quoting), and
 * picks the Python interpreter for bundled payloads (`ticketCommand` + `ticketEnv`)
 * and source checkouts (`kind === 'python'`), or `--run-module` for PM-managed
 * installation launchers (`kind === 'command'`).
 */
export function buildDefaultPluginsPreflightInvocation(
  backend: PreflightBackend | null | undefined,
  options: {
    home: string
    parentEnv?: NodeJS.ProcessEnv
    execPath?: string
    timeoutSeconds?: number
    isWindows?: boolean
  }
): DefaultPluginsPreflightInvocation | null {
  if (!isLocalPreflightBackend(backend)) {
    return null
  }

  const parentEnv = options.parentEnv ?? process.env
  const execPath = options.execPath ?? process.execPath
  const isWindows = options.isWindows ?? process.platform === 'win32'

  const useBundledPython = Boolean(backend.ticketCommand)
  const useDirectPython = !useBundledPython && backend.kind === 'python' && !backend.shell

  const rawCommand = useBundledPython ? backend.ticketCommand! : backend.command!
  const shell = !useBundledPython && !useDirectPython && Boolean(backend.shell)
  const command = shell ? windowsShellCommand(rawCommand, true, isWindows) : rawCommand

  const merged: NodeJS.ProcessEnv = {
    ...parentEnv,
    HERMES_HOME: options.home,
    ...backend.env,
    ...(useBundledPython ? backend.ticketEnv : undefined)
  }

  const { HERMES_DESKTOP_DEV_SERVER: _devServer, ...env } = mountFreeEnv(merged, rawCommand, execPath)
  const timeoutSeconds = resolveDefaultPluginsTimeoutSeconds(options.timeoutSeconds)
  const ensureArgs = ['ensure', '--json', '--timeout', String(timeoutSeconds)]

  const args =
    useBundledPython || useDirectPython
      ? ['-P', '-m', DEFAULT_PLUGINS_MODULE, ...ensureArgs]
      : ['--run-module', DEFAULT_PLUGINS_MODULE, ...ensureArgs]

  return { command, args, env, shell, timeoutSeconds }
}

/** Extract the last valid JSON object line from stdout, normalized to a single line. */
export function parseDefaultPluginsJsonLine(stdout: string): { raw: string; payload: Record<string, unknown> } | null {
  const lines = String(stdout || '')
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)

  for (let i = lines.length - 1; i >= 0; i--) {
    try {
      const parsed: unknown = JSON.parse(lines[i])

      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return { raw: JSON.stringify(parsed), payload: parsed as Record<string, unknown> }
      }
    } catch {
      // Skip non-JSON diagnostic noise and keep scanning earlier lines.
    }
  }

  return null
}

function sanitizeOneLine(text: string): string {
  const collapsed = String(text || '')
    .replace(/\s+/g, ' ')
    .trim()

  if (collapsed.length <= DEFAULT_PLUGINS_MAX_ERROR_CHARS) {
    return collapsed
  }

  return `${collapsed.slice(0, DEFAULT_PLUGINS_MAX_ERROR_CHARS - 1)}…`
}

function lastNonEmptyLine(text: string): string {
  const lines = String(text || '')
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)

  return lines.length > 0 ? lines[lines.length - 1] : ''
}

export function formatDefaultPluginsLogLine(outcome: {
  exitLabel: string
  jsonRaw: string | null
  errorDetail: string | null
  elapsedMs: number
}): string {
  const detail = outcome.jsonRaw ?? sanitizeOneLine(outcome.errorDetail || 'no output')

  return `${DEFAULT_PLUGINS_LOG_PREFIX} exit=${outcome.exitLabel} ${detail} elapsed_ms=${outcome.elapsedMs}`
}

function safeKill(child: ReturnType<typeof spawn>, signal?: NodeJS.Signals): void {
  try {
    if (signal) {
      child.kill(signal)
    } else {
      child.kill()
    }
  } catch {
    // Child may already be gone.
  }
}

/**
 * Run the bounded foreground default-plugins ensure command before starting a
 * local backend. Always resolves (never throws into startup).
 */
export async function runDefaultPluginsPreflight(
  backend: PreflightBackend | null | undefined,
  options: RunDefaultPluginsPreflightOptions
): Promise<DefaultPluginsPreflightResult> {
  const now = options.now ?? Date.now
  const startedAt = now()

  const safeLog = (line: string): void => {
    try {
      options.log?.(line)
    } catch {
      // Never let a logging failure interfere with startup.
    }
  }

  if (options.signal?.aborted) {
    return {
      skipped: true,
      exitCode: null,
      timedOut: false,
      aborted: true,
      elapsedMs: Math.max(0, now() - startedAt),
      payload: null,
      error: 'aborted',
      logLine: null
    }
  }

  let invocation: DefaultPluginsPreflightInvocation | null

  try {
    invocation = buildDefaultPluginsPreflightInvocation(backend, options)
  } catch (err) {
    const elapsedMs = Math.max(0, now() - startedAt)
    const error = err instanceof Error ? err.message : String(err)
    const logLine = formatDefaultPluginsLogLine({ exitLabel: 'error', jsonRaw: null, errorDetail: error, elapsedMs })

    safeLog(logLine)

    return {
      skipped: false,
      exitCode: null,
      timedOut: false,
      aborted: false,
      elapsedMs,
      payload: null,
      error,
      logLine
    }
  }

  if (!invocation) {
    return {
      skipped: true,
      exitCode: null,
      timedOut: false,
      aborted: false,
      elapsedMs: Math.max(0, now() - startedAt),
      payload: null,
      error: null,
      logLine: null
    }
  }

  const spawnImpl = options.spawnImpl ?? spawn
  const isWindows = options.isWindows ?? process.platform === 'win32'
  const timeoutMs = Math.max(1, Math.round(invocation.timeoutSeconds * 1000))
  // Grace past Python's own --timeout so it can spawn its detached continuation and print the exit-3
  // JSON: at least 3 s, since a cold bundled interpreter takes ~0.3 s just to start (a 10% grace on a
  // 2 s timeout killed it mid-report).
  const killGraceMs = Math.max(0, options.killGraceMs ?? Math.min(10_000, Math.max(3_000, Math.round(timeoutMs * 0.1))))
  const hardTimeoutMs = timeoutMs + killGraceMs
  const forceKillMs = Math.max(1, options.forceKillMs ?? DEFAULT_PLUGINS_FORCE_KILL_MS)

  return new Promise<DefaultPluginsPreflightResult>(resolve => {
    let settled = false
    let timedOut = false
    let aborted = false
    let stdout = ''
    let stderr = ''
    let timeoutTimer: ReturnType<typeof setTimeout> | undefined
    let forceKillTimer: ReturnType<typeof setTimeout> | undefined
    let child: ReturnType<typeof spawn> | undefined

    const cleanup = (): void => {
      clearTimeout(timeoutTimer)
      clearTimeout(forceKillTimer)
      options.signal?.removeEventListener('abort', onAbort)
    }

    const finish = (outcome: {
      exitCode: number | null
      exitLabel: string
      jsonRaw: string | null
      payload: Record<string, unknown> | null
      error: string | null
    }): void => {
      if (settled) {
        return
      }

      settled = true
      cleanup()
      const elapsedMs = Math.max(0, now() - startedAt)

      const logLine = formatDefaultPluginsLogLine({
        exitLabel: outcome.exitLabel,
        jsonRaw: outcome.jsonRaw,
        errorDetail: outcome.error,
        elapsedMs
      })

      safeLog(logLine)

      resolve({
        skipped: false,
        exitCode: outcome.exitCode,
        timedOut,
        aborted,
        elapsedMs,
        payload: outcome.payload,
        error: outcome.error,
        logLine
      })
    }

    const terminateChild = (reason: 'timeout' | 'aborted'): void => {
      if (settled) {
        return
      }

      if (reason === 'timeout') {
        timedOut = true
      } else {
        aborted = true
      }

      if (child) {
        safeKill(child, 'SIGTERM')
      }

      if (!forceKillTimer) {
        forceKillTimer = setTimeout(() => {
          if (child) {
            safeKill(child, 'SIGKILL')
          }

          const message =
            reason === 'timeout'
              ? `timed out after ${invocation.timeoutSeconds}s (killed)`
              : 'aborted while starting'

          finish({
            exitCode: null,
            exitLabel: reason,
            jsonRaw: null,
            payload: null,
            error: message
          })
        }, forceKillMs)
      }
    }

    const onAbort = (): void => terminateChild('aborted')

    try {
      child = spawnImpl(
        invocation.command,
        invocation.args,
        hiddenWindowsChildOptions(
          {
            cwd: options.cwd,
            env: invocation.env,
            shell: invocation.shell,
            stdio: ['ignore', 'pipe', 'pipe']
          },
          isWindows
        )
      )
    } catch (err) {
      const error = err instanceof Error ? err.message : String(err)

      finish({
        exitCode: null,
        exitLabel: 'error',
        jsonRaw: null,
        payload: null,
        error
      })

      return
    }

    options.signal?.addEventListener('abort', onAbort, { once: true })
    timeoutTimer = setTimeout(() => terminateChild('timeout'), hardTimeoutMs)

    child.stdout?.on('data', (chunk: Buffer | string) => {
      if (stdout.length < DEFAULT_PLUGINS_MAX_STDOUT_BYTES) {
        stdout = (stdout + chunk.toString()).slice(0, DEFAULT_PLUGINS_MAX_STDOUT_BYTES)
      }
    })

    child.stderr?.on('data', (chunk: Buffer | string) => {
      stderr = (stderr + chunk.toString()).slice(-DEFAULT_PLUGINS_MAX_STDERR_BYTES)
    })

    child.on('error', (err: Error) => {
      finish({
        exitCode: null,
        exitLabel: 'error',
        jsonRaw: null,
        payload: null,
        error: err.message || String(err)
      })
    })

    child.on('close', (code: number | null, signalCode: NodeJS.Signals | null) => {
      const parsed = parseDefaultPluginsJsonLine(stdout)

      if (timedOut) {
        const stderrTail = lastNonEmptyLine(stderr)

        const error = stderrTail
          ? `timed out after ${invocation.timeoutSeconds}s (killed): ${stderrTail}`
          : `timed out after ${invocation.timeoutSeconds}s (killed)`

        finish({
          exitCode: code,
          exitLabel: 'timeout',
          jsonRaw: parsed?.raw ?? null,
          payload: parsed?.payload ?? null,
          error
        })

        return
      }

      if (aborted) {
        finish({
          exitCode: code,
          exitLabel: 'aborted',
          jsonRaw: parsed?.raw ?? null,
          payload: parsed?.payload ?? null,
          error: 'aborted while starting'
        })

        return
      }

      const exitLabel =
        typeof code === 'number' ? String(code) : signalCode ? `signal:${signalCode}` : 'error'

      if (parsed) {
        finish({
          exitCode: code,
          exitLabel,
          jsonRaw: parsed.raw,
          payload: parsed.payload,
          error: null
        })

        return
      }

      const fallbackDetail =
        lastNonEmptyLine(stderr) ||
        lastNonEmptyLine(stdout) ||
        (signalCode ? `terminated by ${signalCode}` : 'exited without JSON output')

      finish({
        exitCode: code,
        exitLabel,
        jsonRaw: null,
        payload: null,
        error: fallbackDetail
      })
    })
  })
}

export interface PreflightGatewayRunnerDeps<TResult> {
  hermesHome: string
  resolveCwd: () => string
  buildSpawnEnv: (env: NodeJS.ProcessEnv) => NodeJS.ProcessEnv
  buildParentEnv: (profile: string) => NodeJS.ProcessEnv
  runEnsure: (
    backend: PreflightBackend,
    cwd: string,
    home: string,
    parentEnv: NodeJS.ProcessEnv,
    options?: { timeoutMs?: number; label?: string; execPath?: string }
  ) => Promise<TResult>
  log: (line: string) => void
  getSignal: () => AbortSignal
  runPreflight?: typeof runDefaultPluginsPreflight
}

export interface PreflightGatewayCallOptions {
  once?: { ran: boolean }
  onBefore?: () => Promise<void> | void
  onAfter?: () => Promise<void> | void
  timeoutSeconds?: number
  ensureOptions?: { timeoutMs?: number; label?: string; execPath?: string }
}

function throwIfPreflightAborted(signal: AbortSignal | undefined, aborted = false): void {
  if (signal?.aborted) {
    signal.throwIfAborted()
  }

  if (aborted) {
    throw new Error('Default plugins preflight aborted')
  }
}

/**
 * Bind Desktop's shared local-gateway spawn environment, preflight runner, and
 * `runGatewayEnsure` so every local start/pool/restart call site shares one
 * preflight-before-ensure sequence without duplicating env assembly.
 */
export function createPreflightGatewayRunner<TResult>(deps: PreflightGatewayRunnerDeps<TResult>) {
  const preflight = deps.runPreflight ?? runDefaultPluginsPreflight

  async function prepare<TBackend extends PreflightBackend>(
    backend: TBackend,
    profile: string,
    options: PreflightGatewayCallOptions = {}
  ): Promise<() => Promise<TResult>> {
    const spawnBackend = { ...backend, env: deps.buildSpawnEnv(backend.env || {}) }
    const cwd = deps.resolveCwd()
    const parentEnv = deps.buildParentEnv(profile)
    const signal = deps.getSignal()

    throwIfPreflightAborted(signal)

    if (!options.once || !options.once.ran) {
      if (options.once) {
        options.once.ran = true
      }

      await options.onBefore?.()

      const result = await preflight(spawnBackend, {
        cwd,
        home: deps.hermesHome,
        parentEnv,
        log: deps.log,
        signal,
        ...(options.timeoutSeconds !== undefined ? { timeoutSeconds: options.timeoutSeconds } : {})
      })

      throwIfPreflightAborted(signal, result.aborted)
      await options.onAfter?.()
    }

    return () => {
      throwIfPreflightAborted(deps.getSignal())

      return deps.runEnsure(spawnBackend, cwd, deps.hermesHome, parentEnv, options.ensureOptions)
    }
  }

  async function run<TBackend extends PreflightBackend>(
    backend: TBackend,
    profile: string,
    options: PreflightGatewayCallOptions = {}
  ): Promise<TResult> {
    const invokeEnsure = await prepare(backend, profile, options)

    return invokeEnsure()
  }

  return { prepare, run }
}
