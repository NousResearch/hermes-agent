/** Bounded backend probes. A file on disk is not proof of a usable runtime. */

import { spawn } from 'node:child_process'
import { statSync } from 'node:fs'

import { buildDesktopBackendEnv } from './backend-env'
import { windowsShellCommand } from './windows-child-options'

/** Default probe budget. 5s false-negativeed healthy Windows cold starts (#61764). */
const DEFAULT_PROBE_TIMEOUT_MS = 15_000

/**
 * Resolve the backend probe timeout (ms).
 * Honours HERMES_PROBE_TIMEOUT_MS when it parses as a positive integer.
 */
function resolveProbeTimeoutMs(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.HERMES_PROBE_TIMEOUT_MS

  if (raw == null || raw === '') {
    return DEFAULT_PROBE_TIMEOUT_MS
  }

  const n = Number.parseInt(String(raw), 10)

  if (!Number.isFinite(n) || n <= 0) {
    return DEFAULT_PROBE_TIMEOUT_MS
  }

  // Clamp absurd values (ms) so a typo can't hang startup forever.
  return Math.min(n, 120_000)
}

const PROBE_TIMEOUT_MS = resolveProbeTimeoutMs()

// Probe-verdict memo. Backend resolution re-runs on every profile dial, and
// each `--version`/import probe is a fresh child spawn (15s budget + one
// retry on timeout). A healthy install therefore pays seconds of spawns per
// dial for an answer that changes only when the binary does: cache the
// boolean verdict per command (+ its mtime, so a replaced binary re-probes)
// with a short TTL, and share the in-flight promise across concurrent dials.
const PROBE_VERDICT_TTL_MS = 30_000
const PROBE_VERDICT_MAX_ENTRIES = 32

interface ProbeVerdict {
  at: number
  promise: Promise<boolean> | null
  result: boolean | null
}

const probeVerdicts = new Map<string, ProbeVerdict>()

function mtimeOf(command: string): string {
  // Shell probes quote the binary ("C:\...cmd"); stat the inner path. A
  // bare PATH name has no statable file — the TTL alone bounds its staleness.
  try {
    return String(statSync(command.replace(/^"(.*)"$/, '$1')).mtimeMs)
  } catch {
    return 'nomtime'
  }
}

async function memoizedProbeVerdict(key: string, probe: () => Promise<boolean>): Promise<boolean> {
  const cached = probeVerdicts.get(key)
  const now = Date.now()

  if (cached) {
    if (cached.result !== null && now - cached.at < PROBE_VERDICT_TTL_MS) {
      return cached.result
    }

    if (cached.promise) {
      return cached.promise
    }
  }

  if (probeVerdicts.size >= PROBE_VERDICT_MAX_ENTRIES && !probeVerdicts.has(key)) {
    const oldest = probeVerdicts.keys().next()

    if (!oldest.done) {
      probeVerdicts.delete(oldest.value)
    }
  }

  const entry: ProbeVerdict = { at: now, promise: null, result: cached?.result ?? null }
  probeVerdicts.set(key, entry)
  entry.promise = probe().then(
    result => {
      entry.promise = null
      entry.result = result
      entry.at = Date.now()

      return result
    },
    error => {
      // A failed probe carries no verdict: drop the entry so the next
      // caller re-probes instead of sharing the throw.
      if (probeVerdicts.get(key) === entry) {
        probeVerdicts.delete(key)
      }

      throw error
    }
  )

  return entry.promise
}

/**
 * Forget every cached probe verdict — after a bootstrap install lands (a
 * verdict minted seconds ago must not hide the new runtime for the TTL), or
 * in tests that script success-then-failure on one path.
 */
function invalidateBackendProbeVerdicts(): void {
  probeVerdicts.clear()
}

function isTimeoutError(err: unknown): boolean {
  if (!err || typeof err !== 'object') {
    return false
  }

  const e = err as { code?: string; killed?: boolean; signal?: string }

  if (e.killed === true) {
    return true
  }

  if (e.code === 'ETIMEDOUT') {
    return true
  }

  // Node marks timed-out execFileSync with SIGTERM on some platforms.
  if (e.signal === 'SIGTERM') {
    return true
  }

  return false
}

/**
 * Run without blocking the event loop; on timeout only, retry once before failing.
 * Non-timeout failures (ENOENT, non-zero exit) fail immediately.
 */
async function execProbe(
  command: string,
  args: string[],
  options: {
    cwd?: string
    env?: NodeJS.ProcessEnv
    stdio: 'ignore'
    timeout: number
    shell?: boolean
    windowsHide?: boolean
  }
): Promise<void> {
  const run = () =>
    new Promise<void>((resolve, reject) => {
      const child = spawn(command, args, options)
      child.once('error', reject)
      child.once('close', (code, signal) => {
        // A timed-out probe may handle SIGTERM and exit zero; it is still a timeout.
        if (code === 0 && !child.killed) {
          resolve()
        } else {
          reject(
            Object.assign(new Error(`Runtime probe failed: ${command} (${signal || code})`), {
              code,
              signal,
              killed: child.killed
            })
          )
        }
      })
    })

  try {
    await run()
  } catch (err) {
    if (!isTimeoutError(err)) {
      throw err
    }

    // One cold-cache / AV miss should not force hermes-setup --update (#61764).
    await run()
  }
}

/** Probe the checkout at cwd with the same dependency activation as launch. */
async function canImportHermesCli(
  pythonPath: string,
  opts: { env?: NodeJS.ProcessEnv; cwd?: string } = {}
): Promise<boolean> {
  if (!pythonPath) {
    return false
  }

  const key = `hermes-import:${pythonPath}:${opts.cwd ?? ''}:${mtimeOf(pythonPath)}`

  return memoizedProbeVerdict(key, async () => {
    try {
      const env: NodeJS.ProcessEnv = { ...process.env, ...opts.env }

      // Bootstrap selects the committed generation before any dependency import.
      await execProbe(
        pythonPath,
        ['-c', 'import hermes_bootstrap; import hermes_yaml; import dotenv; import hermes_cli.config'],
        {
          cwd: opts.cwd,
          env: { ...env, ...buildDesktopBackendEnv({ currentEnv: env }) },
          stdio: 'ignore',
          timeout: PROBE_TIMEOUT_MS,
          windowsHide: true
        }
      )

      return true
    } catch {
      return false
    }
  })
}

/**
 * Return true iff `<hermesCommand> --version` exits 0.
 *
 * Used to gate the "existing `hermes` on PATH" rung. Without this, a
 * stale hermes.cmd shim left behind by an uninstalled pip install (or
 * a half-built venv whose `hermes` entry-point points at a deleted
 * Python) survives findOnPath() and gets selected as the backend.
 *
 * We intentionally avoid invoking the command with the dashboard args
 * here -- `--version` is the cheapest "is this binary alive" smoke
 * test that every hermes_cli entry-point has supported since 0.1.
 *
 * @param {string} hermesCommand - Resolved absolute path to a hermes
 *   executable (or an interpreter+script wrapper).
 * @param {boolean} [opts.shell] - Whether to run through a shell. For
 *   .cmd/.bat shims on Windows spawn needs shell:true to find
 *   the cmd interpreter; mirrors the same flag isCommandScript() drives
 *   in resolveHermesBackend.
 * @returns {boolean}
 */
/**
 * An explicit desktop backend command is a deployment contract, not a PATH
 * discovery candidate. In particular, the Nix desktop wrapper points this at
 * its immutable, matching Hermes package; it must never fall through to the
 * mutable install-script bootstrap path if a best-effort probe is slow.
 */
function shouldTrustHermesOverride(hermesOverride?: string) {
  return typeof hermesOverride === 'string' && hermesOverride.trim().length > 0
}

async function verifyHermesCli(hermesCommand: string, opts?: { shell?: boolean }) {
  if (!hermesCommand) {
    return false
  }

  const shell = Boolean(opts?.shell)
  const key = `hermes-version:${shell}:${hermesCommand}:${mtimeOf(windowsShellCommand(hermesCommand, shell))}`

  return memoizedProbeVerdict(key, async () => {
    try {
      await execProbe(windowsShellCommand(hermesCommand, shell), ['--version'], {
        stdio: 'ignore',
        timeout: PROBE_TIMEOUT_MS,
        shell,
        windowsHide: true
      })

      return true
    } catch {
      return false
    }
  })
}

export {
  canImportHermesCli,
  DEFAULT_PROBE_TIMEOUT_MS,
  execProbe,
  invalidateBackendProbeVerdicts,
  isTimeoutError,
  memoizedProbeVerdict,
  mtimeOf,
  PROBE_TIMEOUT_MS,
  resolveProbeTimeoutMs,
  shouldTrustHermesOverride,
  verifyHermesCli
}
