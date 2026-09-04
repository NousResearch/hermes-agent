/**
 * package-process-reap.ts
 *
 * Reap every process still running out of our MSIX package root at shutdown.
 *
 * Why this exists. A Desktop-Bridge (Centennial) MSIX package activates inside
 * a Desktop AppX container: Windows creates a Job Object for the package family
 * and converts it to a silo. One live process rooted in the family pins that
 * silo. While it is pinned, EVERY later activation of the family fails at the
 * conversion step:
 *
 *   AppModel-Runtime/Admin
 *   [215] 0x80070020: Cannot create the Desktop AppX container for package
 *         <full name> because an error was encountered converting the job.
 *   [208] 0x80070020: Cannot create the process for package <full name>
 *         because an error was encountered while configuring runtime.
 *         [LaunchProcess]
 *
 * 0x80070020 is ERROR_SHARING_VIOLATION, which Windows surfaces to the user as
 * "The process cannot access the file because it is being used by another
 * process" — a file-lock message for what is really a container-layer failure.
 *
 * Observed live on windows-11-arm: the payload's own bundled-git gpg-agent.exe
 * (under tools/git-<version>/usr/bin/) had daemonized, outlived its package by
 * a week, and pinned the family. Killing that ONE pid turned [215]/[208] into
 * [210] Created / [211] Added on the very next activation, with nothing else
 * changed.
 *
 * Why the platform does not save us. Deployment already asks for the shutdown
 * (ForceApplicationShutdownOption + ForceTargetApplicationShutdownOption) and
 * it still missed the daemon; the old package folder then failed to move with
 * `0x80070005` and Windows logged `[503] The file system entries ... could not
 * be cleaned up after reboot. The package is removed from the purge list.`
 * Once that happens not even a reboot clears it. Our daemon defeated the
 * platform's own mechanism, so the reap is ours to do.
 *
 * Why not a process-TREE kill. `taskkill /T` walks parentage, and a daemon that
 * detaches is by construction no longer in our tree — that is what daemonizing
 * means. Tree scoping is the actual bug. Install-root scoping is what matches
 * the silo: the pin is "a process whose image lives under this package", so
 * that is exactly the predicate to kill on.
 *
 * Why not a gpg-specific kill. `gpgconf --kill all` is scoped by GNUPGHOME, not
 * by install root, so it can reach the user's OWN gpg-agent outside our package
 * — real collateral harm for a case the generic path reap already covers. One
 * mechanism, no per-daemon special cases: any future payload tool that
 * daemonizes is caught with no new code.
 *
 * Scoping. Two roots hold binaries this install owns, and either can hold the
 * straggler:
 *
 *   1. The artifact itself — resources/ on a bundled/MSIX install, which
 *      contains agent-payload (repo + venv + its own tool store).
 *   2. The per-install managed tool store — where pm stages node, git, uv,
 *      ripgrep and friends for a MUTABLE install (an install.ps1 / install.sh
 *      checkout). That is HERMES_RUNTIME_DIR when set, else <hermes root>/tools.
 *      A bundled payload sets HERMES_RUNTIME_DIR at its own store, so the two
 *      roots coincide there and the overlap is harmless.
 *
 * A mutable install has exactly the same daemon problem with none of the MSIX
 * framing: node tooling, a language server, or bundled git's gpg-agent under
 * <hermes root>/tools keeps running and holds an open handle on the very files
 * the next `hermes update` wants to replace. On Windows a running image cannot
 * be overwritten, which is the same 0x80070020 the user reads as "used by
 * another process". Reaping both roots covers both install shapes with one
 * predicate.
 *
 * The roots are deliberately the CURRENT install's, not the package family or
 * every Hermes on the box: once this ships, each shutdown reaps its own
 * processes, so no install leaves a daemon behind. Cleaning up a straggler
 * that a PREVIOUS version already leaked needs a repair path outside the
 * install (the app cannot launch to clean up after itself) and is tracked
 * separately.
 *
 * Live-verified on windows-11-arm at Medium Mandatory Level (admin=False, the
 * integrity level the app actually runs at): all 6 package-rooted processes
 * enumerated, all killed, 0 remaining, 0 with an unreadable path — then the
 * next activation logged [210]/[211] and the app came up clean.
 *
 * Pure module: no electron import, no process globals reached directly, every
 * dependency injected — so the predicate and the ordering are asserted against
 * the real function rather than by grepping main.ts.
 */

/** One running process, reduced to what the reap decision needs. */
export interface RunningProcess {
  pid: number
  /**
   * Absolute path of the process image, or null when it cannot be read.
   * A path we cannot read is never a match — see reapPackageRootedProcesses.
   */
  path: string | null
}

export interface ReapPackageRootedProcessesDeps {
  /**
   * Absolute roots to scope on: the artifact resources dir and/or the managed
   * tool store. Empty/nullish entries are ignored, and an empty list disables
   * the reap entirely.
   */
  installRoots: ReadonlyArray<string | null | undefined>
  /** Snapshot of running processes. Real: enumerate with readable image paths. */
  listProcesses: () => RunningProcess[]
  /** Force-kill one pid. Real: process.kill(pid, 'SIGKILL') / taskkill /F. */
  killProcess: (pid: number) => void
  /** Our own pid, never reaped — we are rooted in the package too. */
  selfPid: number
  /**
   * Pids already being torn down by the normal backend path. Skipped so the
   * graceful teardown owns them and this stays a net for what it missed.
   */
  excludePids?: Iterable<number>
  /** Defaults to the real platform check; injectable for tests. */
  isWindows?: boolean
  /** Optional diagnostic sink; failures must never break quit. */
  log?: (message: string) => void
}

export interface ReapOutcome {
  /** Package-rooted processes found (excluding self and excludePids). */
  matched: number
  /** Pids the kill call was made for. */
  killed: number[]
  /** Pids whose kill threw (already gone, or not permitted). */
  failed: number[]
  /** True when the reap did not run (non-Windows, or no install root). */
  skipped: boolean
}

/**
 * Enumerate running processes with their image paths, via PowerShell's
 * Get-Process.
 *
 * Live-verified at Medium Mandatory Level (admin=False): all package-rooted
 * processes were enumerated with readable paths and killed, so this needs no
 * elevation and no WMI. `.Path` throws for processes the caller cannot open,
 * which is why each read is guarded and reported as a null path rather than
 * aborting the sweep.
 *
 * Bounded and best effort: this runs on the quit path, so it takes a hard
 * timeout and returns an empty list rather than delaying shutdown.
 */
export function listWindowsProcesses(
  execFile: (file: string, args: string[], options: { timeout: number; windowsHide: boolean }) => string
): RunningProcess[] {
  const script =
    '$ErrorActionPreference = "SilentlyContinue"; ' +
    'Get-Process | ForEach-Object { ' +
    '$p = $null; try { $p = $_.Path } catch { $p = $null }; ' +
    '"{0}|{1}" -f $_.Id, $p }'

  const stdout = execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script], {
    timeout: 10_000,
    windowsHide: true
  })

  const processes: RunningProcess[] = []

  for (const line of String(stdout).split(/\r?\n/)) {
    const separator = line.indexOf('|')

    if (separator <= 0) {
      continue
    }

    const pid = Number.parseInt(line.slice(0, separator), 10)

    if (!Number.isInteger(pid)) {
      continue
    }

    const imagePath = line.slice(separator + 1).trim()
    processes.push({ pid, path: imagePath === '' ? null : imagePath })
  }

  return processes
}

/**
 * True when `imagePath` lives under any of `roots`.
 *
 * Windows paths are case-insensitive, so the compare is too. The separator
 * check is what keeps the prefix honest: a bare startsWith would match a
 * sibling directory whose name merely begins with a root
 * (`...\HermesBundled_0.21` vs `...\HermesBundled_0.21.20`), and killing
 * another package's processes is a far worse bug than the one being fixed.
 */
export function isUnderInstallRoot(
  imagePath: string | null | undefined,
  roots: ReadonlyArray<string | null | undefined> | string | null | undefined
): boolean {
  if (!imagePath) {
    return false
  }

  const normalize = (value: string): string => value.replace(/[\\/]+$/, '').replace(/\//g, '\\').toLowerCase()
  const normalizedPath = normalize(imagePath)
  const candidates = typeof roots === 'string' || roots == null ? [roots] : roots

  for (const root of candidates) {
    if (typeof root !== 'string' || !root) {
      continue
    }

    const normalizedRoot = normalize(root)

    if (!normalizedRoot) {
      continue
    }

    if (normalizedPath === normalizedRoot || normalizedPath.startsWith(`${normalizedRoot}\\`)) {
      return true
    }
  }

  return false
}

/**
 * Kill every process whose image lives under the install root.
 *
 * Best effort by construction: this runs on the quit path, so a failure to
 * enumerate or to kill is logged and swallowed rather than allowed to hang or
 * crash the shutdown. A process whose path cannot be read is left alone — the
 * cost of missing one pinner is the bug we already have, while killing a
 * process we could not identify is unbounded damage.
 */
export function reapPackageRootedProcesses(deps: ReapPackageRootedProcessesDeps): ReapOutcome {
  const isWindows = deps.isWindows ?? process.platform === 'win32'
  const log = deps.log ?? ((): void => undefined)
  const roots = (deps.installRoots ?? []).filter((root): root is string => Boolean(root))

  // The container silo is a Windows construct; POSIX has nothing to reap.
  if (!isWindows || roots.length === 0) {
    return { matched: 0, killed: [], failed: [], skipped: true }
  }

  let running: RunningProcess[]

  try {
    running = deps.listProcesses()
  } catch (err) {
    log(`[package-reap] process enumeration failed: ${(err as Error).message}`)

    return { matched: 0, killed: [], failed: [], skipped: true }
  }

  const excluded = new Set<number>(deps.excludePids ?? [])
  const killed: number[] = []
  const failed: number[] = []
  let matched = 0

  for (const candidate of running) {
    if (!Number.isInteger(candidate.pid) || candidate.pid === deps.selfPid || excluded.has(candidate.pid)) {
      continue
    }

    if (!isUnderInstallRoot(candidate.path, roots)) {
      continue
    }

    matched += 1

    try {
      deps.killProcess(candidate.pid)
      killed.push(candidate.pid)
    } catch (err) {
      failed.push(candidate.pid)
      log(`[package-reap] kill pid=${candidate.pid} failed: ${(err as Error).message}`)
    }
  }

  if (matched > 0) {
    log(`[package-reap] matched=${matched} killed=${killed.length} failed=${failed.length}`)
  }

  return { matched, killed, failed, skipped: false }
}
