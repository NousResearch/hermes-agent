/** Clean up detached package tools without touching shared runtimes.
 * Package-owned gpg-agent can outlive its parent and pin an MSIX silo,
 * preventing subsequent activation. A root claim never includes the shared
 * tool store. Interpreters, launchers, unidentified processes and their
 * descendants belong to other lifecycle owners and are protected.
 */

/** One running process, reduced to what the reap decision needs. */
export interface RunningProcess {
  pid: number
  /**
   * Parent pid (Win32_Process.ParentProcessId), or null when unavailable.
   * Descendants of a live Hermes runtime process are protected too — see
   * reapPackageRootedProcesses.
   */
  parentPid: number | null
  /**
   * Absolute path of the process image, or null when it cannot be read.
   * A path we cannot read is never a match — see reapPackageRootedProcesses.
   */
  path: string | null
  /**
   * Full command line, or null when it cannot be read. A live Hermes runtime
   * process is identified by its command line and never reaped — see
   * protectedRuntimePids. An UNREADABLE command line is conservatively
   * skipped as well: the cost of missing one pinner is the bug we already
   * have; killing a process we could not identify is unbounded damage.
   */
  commandLine: string | null
}

export interface ReapPackageRootedProcessesDeps {
  /**
   * Absolute roots to scope on: the artifact resources dir (never the shared
   * tool store). Empty/nullish entries are ignored, and an empty list disables
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

/** Shared interpreters and entry points are not owned by a desktop window.
 * Do not parse argv to decide which Python flags or launcher forms qualify.
 * The updater owns their lifecycle; quit only sweeps detached tools.
 */
function isSharedRuntime(imagePath: string): boolean {
  const image = imagePath.replace(/^.*[\\/]/, '').toLowerCase()
  return /^(?:pythonw?(?:[0-9.]+)?|node|hermes(?:-agent|-acp)?)(?:\.exe|\.cmd)?$/.test(image)
}

/**
 * Enumerate running processes with image path, parent pid and command line,
 * via PowerShell's Win32_Process query (Get-Process does not carry the
 * command line, and the ownership decision needs it).
 *
 * Probed READ-ONLY on windows-11-arm at Medium Mandatory Level (admin=False,
 * the integrity level the app actually runs at): all processes enumerated
 * with readable pids/parents and ~45% null ExecutablePath/CommandLine —
 * each null is reported as such rather than aborting the sweep.
 *
 * Bounded and best effort: this runs on the quit path, so it takes a hard
 * timeout and returns an empty list rather than delaying shutdown.
 */
export function listWindowsProcesses(
  execFile: (file: string, args: string[], options: { timeout: number; windowsHide: boolean }) => string
): RunningProcess[] {
  const script =
    '$ErrorActionPreference = "SilentlyContinue"; ' +
    'Get-CimInstance Win32_Process | ForEach-Object { ' +
    '"{0}|{1}|{2}|{3}" -f $_.ProcessId, $_.ParentProcessId, $_.ExecutablePath, $_.CommandLine }'

  const stdout = execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script], {
    timeout: 10_000,
    windowsHide: true
  })

  const processes: RunningProcess[] = []

  for (const line of String(stdout).split(/\r?\n/)) {
    const first = line.indexOf('|')
    const second = first >= 0 ? line.indexOf('|', first + 1) : -1
    const third = second >= 0 ? line.indexOf('|', second + 1) : -1

    if (first <= 0 || second < 0 || third < 0) {
      continue
    }

    const pid = Number.parseInt(line.slice(0, first), 10)
    const parentPid = Number.parseInt(line.slice(first + 1, second), 10)

    if (!Number.isInteger(pid)) {
      continue
    }

    const imagePath = line.slice(second + 1, third).trim()

    processes.push({
      pid,
      parentPid: Number.isInteger(parentPid) ? parentPid : null,
      path: imagePath === '' ? null : imagePath,
      commandLine: line.slice(third + 1)
    })
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
 * The set of pids the reap must never touch: every shared runtime or unidentified
 * process plus ALL of its descendants (Win32_Process parentage). The virtualenv
 * launcher/worker chain — the gateway's python spawning venv shims, workers,
 * and in-flight tools like git — is exactly how a live runtime's children come
 * to be rooted in the payload, so parentage, not just the launcher argv, is
 * the ownership boundary. A stale ParentProcessId that happens to point at a
 * runtime pid errs on the protected side: conservative by design.
 */
export function protectedRuntimePids(running: RunningProcess[]): Set<number> {
  const children = new Map<number, number[]>()

  for (const entry of running) {
    if (entry.parentPid == null) {
      continue
    }

    const siblings = children.get(entry.parentPid)

    if (siblings) {
      siblings.push(entry.pid)
    } else {
      children.set(entry.parentPid, [entry.pid])
    }
  }

  const protectedPids = new Set<number>()

  for (const entry of running) {
    if (!entry.path || !entry.commandLine?.trim() || isSharedRuntime(entry.path)) {
      protectedPids.add(entry.pid)
    }
  }

  const queue = [...protectedPids]

  while (queue.length > 0) {
    const parent = queue.pop() as number

    for (const child of children.get(parent) ?? []) {
      if (!protectedPids.has(child)) {
        protectedPids.add(child)
        queue.push(child)
      }
    }
  }

  return protectedPids
}

/**
 * Reap package-rooted tools outside the protected runtime trees.
 *
 * Best effort by construction: this runs on the quit path, so a failure to
 * enumerate or to kill is logged and swallowed rather than allowed to hang or
 * crash the shutdown. A process whose path OR command line cannot be read is
 * left alone — the cost of missing one pinner is the bug we already have,
 * while killing a process we could not identify is unbounded damage.
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
  const runtimeProtected = protectedRuntimePids(running)
  const killed: number[] = []
  const failed: number[] = []
  let matched = 0

  for (const candidate of running) {
    if (
      !Number.isInteger(candidate.pid) ||
      candidate.pid === deps.selfPid ||
      excluded.has(candidate.pid) ||
      runtimeProtected.has(candidate.pid)
    ) {
      continue
    }

    // Conservative skip: an unreadable command line cannot prove the process
    // is not a live Hermes runtime, so it is never a candidate.
    if (typeof candidate.commandLine !== 'string' || candidate.commandLine.trim() === '') {
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
