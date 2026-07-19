/**
 * backend-stale-reap.ts
 *
 * Pre-spawn reaper for stale `hermes serve --profile <name>` processes
 * (Issue #67026).
 *
 * The desktop keeps its spawned backends in an in-memory `backendPool`; the
 * OS-level subprocesses outlive the pool entry because `app.on('before-quit')`
 * is fire-and-forget. Every restart of the desktop therefore leaves the
 * previous round of pool+primary backends running. After a handful of
 * `hermes update` + relaunch cycles users observed ~47 orphan `hermes serve`
 * processes (~300 MB RSS waste) where each profile was supposed to have
 * exactly one.
 *
 * This module gives the two spawn sites (`startHermes()` and
 * `spawnPoolBackend()` in `main.ts`) a way to enumerate and terminate stale
 * `hermes serve --profile <name>` processes immediately before they spawn a
 * fresh one. The platform-specific process-enumeration layer is injectable
 * so tests assert contract without depending on `wmic`/`ps` being present
 * in the test environment.
 *
 * Reuses primitives already in tree:
 *  - `forceKillProcessTree` (Windows-only `taskkill /T /F`) from `main.ts`.
 *  - `isPidAlive` from `update-marker.ts` (signal-0 probe, injectable).
 *
 * Not the same as update-time venv-holder cleanup (`_detect_venv_python_processes`
 * in `hermes_cli/main.py`) — that runs ONCE on `hermes update` and refuses the
 * whole update if any venv holder is detected. This module runs PER SPAWN and
 * reaps so the new instance can come up cleanly. Different lifecycle, different
 * contract.
 */

import { execFileSync } from 'node:child_process'

import { isPidAlive } from './update-marker'

const IS_WINDOWS_DEFAULT = process.platform === 'win32'

/**
 * Injectable dependencies so the platform-specific bits (`wmic`/`ps` argv
 * parsing, the actual kill mechanism) can be substituted in tests without
 * spawning processes or touching the host.
 */
export interface ReapDeps {
  isWindows?: boolean
  /**
   * Return the list of PIDs currently running whose command line matches the
   * `hermes serve --profile <profile>` shape. The implementation must be
   * platform-conditional by the caller — we don't want this module to shell
   * out to `wmic` AND `ps` on every host.
   */
  listCandidatePids: (profile: string) => number[]
  /** Windows-only: pre-existing main.ts forceKillProcessTree via taskkill /T /F. */
  forceKillProcessTree: (pid: number) => void
  /** POSIX-only: process.kill(pid, signal). */
  terminatePid: (pid: number) => void
  /** Sleep helper so the bounded wait can be tested. */
  sleep: (ms: number) => Promise<void>
  /** Bound for the SIGTERM→escalate window (ms). Default 2500. */
  terminationWaitMs?: number
  /** Liveness probe — defaults to `isPidAlive`. Injectable for tests. */
  pidAlive?: (pid: number) => boolean
}

export const DEFAULT_REAP_DEPS = {
  isWindows: IS_WINDOWS_DEFAULT,
  listCandidatePids: (profile: string): number[] =>
    IS_WINDOWS_DEFAULT
      ? listWindowsBackendPids(profile)
      : listPosixBackendPids(profile),
  forceKillProcessTree: (pid: number) => {
    if (!IS_WINDOWS_DEFAULT || !Number.isInteger(pid) || pid <= 0) {
      return
    }
    try {
      execFileSync(
        'taskkill',
        ['/PID', String(pid), '/T', '/F'],
        { stdio: 'ignore', windowsHide: true },
      )
    } catch {
      // Already gone or no permission — best effort.
    }
  },
  terminatePid: (pid: number) => {
    try {
      process.kill(pid, 'SIGTERM')
    } catch {
      // ESRCH etc. — process already gone.
    }
  },
  sleep: (ms: number) =>
    new Promise<void>((resolve) => {
      setTimeout(resolve, ms)
    }),
  terminationWaitMs: 2500,
  pidAlive: isPidAlive,
}

/**
 * Find candidate PIDs that look like stale `hermes serve --profile <profile>`
 * processes. Always excludes `process.pid` (the desktop itself).
 *
 * The shape is intentionally narrow: a backend whose cmdline carries BOTH a
 * `serve` token AND a `--profile <name>` flag. The default implementations
 * gate on that exact two-token pattern; tests can substitute any enumerator
 * that respects the contract.
 */
export function findStaleBackendPids(
  profile: string,
  deps: Pick<ReapDeps, 'listCandidatePids'> = DEFAULT_REAP_DEPS,
): number[] {
  if (!profile) {
    return []
  }
  const candidates = deps.listCandidatePids(profile)
  const exclude = process.pid
  const seen = new Set<number>()
  const out: number[] = []
  for (const pid of candidates) {
    if (!Number.isInteger(pid) || pid <= 0 || pid === exclude || seen.has(pid)) {
      continue
    }
    seen.add(pid)
    out.push(pid)
  }
  return out
}

/**
 * Terminate a list of stale backend PIDs using the platform-appropriate
 * strategy. Returns the set of PIDs that were STILL alive after the bounded
 * wait — those are considered respawning / supervised and should NOT cause
 * the spawn site to abort; the new spawn simply proceeds in parallel and the
 * conflict surfaces through the normal port-already-bound/timeout error path.
 *
 * Idempotent: an empty input is a no-op. Never raises: any kill failure or
 * non-integer pid is silently skipped (mirrors `stopBackendChild` semantics).
 */
export async function reapStaleBackendPids(
  pids: number[],
  deps: ReapDeps = DEFAULT_REAP_DEPS,
): Promise<number[]> {
  if (pids.length === 0) {
    return []
  }
  const isWindows = deps.isWindows ?? IS_WINDOWS_DEFAULT
  const waitMs = deps.terminationWaitMs ?? 2500
  const pidAlive = deps.pidAlive ?? isPidAlive

  for (const pid of pids) {
    if (!Number.isInteger(pid) || pid <= 0) {
      continue
    }
    try {
      if (isWindows) {
        deps.forceKillProcessTree(pid)
      } else {
        deps.terminatePid(pid)
      }
    } catch {
      // Already gone or no permission — best effort.
    }
  }

  // Bounded wait for processes to exit. On POSIX SIGTERM is honored quickly;
  // on Windows the tree-kill is synchronous but we still let stragglers
  // settle so the new spawn doesn't race the old backend's port release.
  await deps.sleep(waitMs)

  const survivors: number[] = []
  for (const pid of pids) {
    if (pidAlive(pid)) {
      survivors.push(pid)
    }
  }
  return survivors
}

/**
 * One-shot helper the spawn sites call. Combines enumeration + kill +
 * bounded wait; returns the survivors so callers can log them.
 */
export async function reapStaleBackendsForProfile(
  profile: string,
  deps: ReapDeps = DEFAULT_REAP_DEPS,
): Promise<number[]> {
  const stale = findStaleBackendPids(profile, deps)
  return await reapStaleBackendPids(stale, deps)
}

// ---------------------------------------------------------------------------
// Platform enumerators (production defaults, NOT covered by unit tests — the
// layer below them is injectable in tests).
// ---------------------------------------------------------------------------

/**
 * Windows: parse `wmic process get ProcessId,CommandLine /FORMAT:CSV`. Output
 * looks like:
 *
 *   Node,CommandLine,ProcessId
 *   ...
 *   desktop,"C:\Program Files\Hermes\hermes.exe" --profile worker serve ...,1234
 *
 * CSV fields are comma-separated with double-quoted strings; embedded quotes
 * are doubled (CSV-RFC-4180). We split on the first comma only for the Node
 * column, then on commas for the remaining two.
 *
 * Win11 24H2+ deprecates `wmic`; on non-zero exit we fall back to
 * `tasklist /v /fo csv` which uses a different column order (`"PID","Image
 * Name","..."`). Either CSV is parsed tolerantly — the only column we need
 * is the integer PID.
 */
function listWindowsBackendPids(profile: string): number[] {
  const matches: number[] = []
  let csv = ''
  try {
    csv = execFileSync(
      'wmic',
      ['process', 'get', 'ProcessId,CommandLine', '/FORMAT:CSV'],
      { stdio: ['ignore', 'pipe', 'ignore'], encoding: 'utf8', windowsHide: true },
    )
  } catch {
    // wmic is deprecated on Win11 24H2+ and frequently missing or
    // permission-denied in restricted environments. We deliberately do NOT
    // fall back to `tasklist /v /fo csv`: tasklist does NOT expose the
    // full command line on recent Windows builds, so pretending we can
    // parse it would just emit false positives (every visible process
    // gets killed). Fail closed — return empty — until a `Get-CimInstance`
    // or `ps`-shim enumerator is wired up.
    return []
  }
  // Profile match must be EXACT, not substring-prefix, so reaping `work`
  // never matches `--profile=worker`. Accepts both `--profile NAME`
  // (followed by space) and `--profile=NAME` (followed by end-of-string,
  // whitespace, or quote). Profile token is lowercased to match `lc`.
  const profileLower = profile.toLowerCase()
  const isBackendLine = (cmdline: string): boolean => {
    const lc = cmdline.toLowerCase()
    if (!lc.includes('serve')) {
      return false
    }
    // `--profile NAME` form: needs a non-name boundary after NAME
    // (whitespace, quote, or end-of-string).
    const spacedIdx = lc.indexOf('--profile ')
    if (spacedIdx !== -1) {
      const after = spacedIdx + '--profile '.length
      const candidate = lc.slice(after, after + profileLower.length)
      if (candidate === profileLower) {
        const trailing = lc.slice(after + profileLower.length)
        if (
          trailing === '' ||
          trailing.startsWith(' ') ||
          trailing.startsWith('\t') ||
          trailing.startsWith('"') ||
          trailing.startsWith("'")
        ) {
          return true
        }
      }
    }
    // `--profile=NAME` form: similar non-name boundary.
    const equalsTag = `--profile=${profileLower}`
    const equalsIdx = lc.indexOf(equalsTag)
    if (equalsIdx !== -1) {
      const after = equalsIdx + equalsTag.length
      const trailing = lc.slice(after)
      if (
        trailing === '' ||
        trailing.startsWith(' ') ||
        trailing.startsWith('\t') ||
        trailing.startsWith('"') ||
        trailing.startsWith("'")
      ) {
        return true
      }
    }
    return false
  }
  for (const line of csv.split(/\r?\n/)) {
    if (!line.trim()) {
      continue
    }
    const parts = parseWindowsCsvLine(line)
    if (parts.length < 3) {
      continue
    }
    const pid = Number.parseInt(parts[parts.length - 1], 10)
    const cmdline = parts.slice(1, parts.length - 1).join(',')
    if (!Number.isInteger(pid) || pid <= 0) {
      continue
    }
    if (isBackendLine(cmdline)) {
      matches.push(pid)
    }
  }
  return matches
}

/**
 * POSIX: `ps -eo pid,args` (Linux) / `ps -axo pid,command` (macOS) — each line
 * is `<pid> <args>` with header. Linux argv is space-separated; macOS
 * `command` joins argv by single spaces but does NOT honor quoting. Both
 * work for the simple presence-test we need.
 */
function listPosixBackendPids(profile: string): number[] {
  const isMac = process.platform === 'darwin'
  // Enumerate ALL processes; findStaleBackendPids filters out `process.pid`
  // (the desktop itself) and any other ancestor PIDs. The previous incarnation
  // passed `-p <pid>` to limit `ps` output to the desktop process — which
  // also dropped every other PID, defeating the macOS enumerator entirely.
  // See PR #67408 review (teknium1 hermes-sweeper, 2026-07-19).
  const psArgs = isMac
    ? ['-axo', 'pid=,command=']
    : ['-eo', 'pid,args', '--no-headers']
  let out = ''
  try {
    out = execFileSync(
      'ps',
      psArgs,
      { stdio: ['ignore', 'pipe', 'ignore'], encoding: 'utf8' },
    )
  } catch {
    return []
  }
  const matches: number[] = []
  const profileLower = profile.toLowerCase()
  for (const line of out.split(/\r?\n/)) {
    const trimmed = line.replace(/^\s+/, '')
    if (!trimmed) {
      continue
    }
    const sp = trimmed.indexOf(' ')
    if (sp <= 0) {
      continue
    }
    const pid = Number.parseInt(trimmed.slice(0, sp), 10)
    const rest = trimmed.slice(sp + 1).toLowerCase()
    if (!Number.isInteger(pid) || pid <= 0) {
      continue
    }
    if (!rest.includes('serve')) {
      continue
    }
    // Exact-match profile detection (--profile NAME serves NAME,
    // --profile=NAME likewise). Substring prefix match would let
    // profile=`work` reap profile=`worker`; see sweeper review.
    const spacedIdx = rest.indexOf('--profile ')
    if (spacedIdx !== -1) {
      const after = spacedIdx + '--profile '.length
      const candidate = rest.slice(after, after + profileLower.length)
      if (candidate === profileLower) {
        const trailing = rest.slice(after + profileLower.length)
        if (
          trailing === '' ||
          trailing.startsWith(' ') ||
          trailing.startsWith('\t') ||
          trailing.startsWith('"') ||
          trailing.startsWith("'")
        ) {
          matches.push(pid)
          continue
        }
      }
    }
    const equalsTag = `--profile=${profileLower}`
    const equalsIdx = rest.indexOf(equalsTag)
    if (equalsIdx !== -1) {
      const after = equalsIdx + equalsTag.length
      const trailing = rest.slice(after)
      if (
        trailing === '' ||
        trailing.startsWith(' ') ||
        trailing.startsWith('\t') ||
        trailing.startsWith('"') ||
        trailing.startsWith("'")
      ) {
        matches.push(pid)
      }
    }
  }
  return matches
}

/**
 * RFC-4180-ish split: the first column never has quotes (it's the literal
 * header token like `Node`); remaining fields may be quoted with embedded
 * doubled quotes. We tolerate both shapes — wmic CSV and tasklist CSV —
 * because either may be the source on a given Windows build.
 */
function parseWindowsCsvLine(line: string): string[] {
  const out: string[] = []
  let cur = ''
  let inQuotes = false
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          cur += '"'
          i++
        } else {
          inQuotes = false
        }
      } else {
        cur += ch
      }
    } else {
      if (ch === ',') {
        out.push(cur)
        cur = ''
      } else if (ch === '"' && cur === '') {
        inQuotes = true
      } else {
        cur += ch
      }
    }
  }
  out.push(cur)
  return out
}
