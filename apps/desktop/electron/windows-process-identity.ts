import { execFile } from 'node:child_process'
import { readFileSync } from 'node:fs'
import path from 'node:path'

const DEFAULT_PROBE_TIMEOUT_MS = 3_000
// Main polls the update marker once per second. Keep a successful async query
// available across at least one full poll interval; positive updater adoption
// still bypasses this cache and performs a fresh awaited OS query.
const DEFAULT_CACHE_MS = 2_000
const INTEGER_EPOCH_PATTERN = /^[1-9][0-9]{8,11}$/

type RunProbe = (command: string, args: string[], timeoutMs: number) => Promise<string>

interface QueryOptions {
  platform?: NodeJS.Platform
  run?: RunProbe
  timeoutMs?: number
}

interface CacheOptions {
  cacheMs?: number
  now?: () => number
  query?: (pid: number) => Promise<number | null>
}

function powershellExecutable(): string {
  const windowsRoot = process.env.SystemRoot || 'C:\\Windows'

  return path.join(windowsRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
}

function runProbe(command: string, args: string[], timeoutMs: number): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(
      command,
      args,
      { encoding: 'utf8', timeout: timeoutMs, windowsHide: true },
      (error, stdout) => (error ? reject(error) : resolve(String(stdout)))
    )
  })
}

/**
 * Query one Windows process creation time without blocking Electron's event
 * loop. Access denial, timeout, malformed output, and process exit all return
 * null so callers can fail closed.
 */
export async function queryWindowsProcessCreatedAt(
  pid: number,
  { platform = process.platform, run = runProbe, timeoutMs = DEFAULT_PROBE_TIMEOUT_MS }: QueryOptions = {}
): Promise<number | null> {
  if (platform !== 'win32' || !Number.isInteger(pid) || pid <= 0) {return null}

  const script =
    `$p=Get-Process -Id ${pid} -ErrorAction Stop;` +
    '[DateTimeOffset]::new($p.StartTime.ToUniversalTime()).ToUnixTimeSeconds()'

  let raw: string

  try {
    raw = (await run(
      powershellExecutable(),
      ['-NoLogo', '-NoProfile', '-NonInteractive', '-Command', script],
      timeoutMs
    )).trim()
  } catch {
    return null
  }

  if (!INTEGER_EPOCH_PATTERN.test(raw)) {return null}
  const createdAt = Number(raw)

  return Number.isSafeInteger(createdAt) && createdAt > 0 ? createdAt : null
}

// Linux exposes process start time in USER_HZ ticks since boot. USER_HZ is
// fixed at 100 for userspace on every mainstream Linux ABI (the kernel scales
// /proc values to it regardless of CONFIG_HZ), so no sysconf binding is needed.
const LINUX_USER_HZ = 100

/**
 * Linux process creation epoch from /proc alone (#B7).
 *
 * Without this the identity probe returns `unknown` for every live PID on
 * Linux, so a recycled PID keeps a marker alive forever and the MCP bridge
 * never comes back. `/proc/<pid>/stat` field 22 is the start time in USER_HZ
 * ticks since boot; `/proc/stat`'s `btime` is the boot wall clock.
 */
export function readLinuxProcessCreatedAt(
  pid: number,
  readText: (file: string) => string = file => readFileSync(file, 'utf8')
): number | null {
  if (!Number.isInteger(pid) || pid <= 0) {return null}

  try {
    const stat = readText(`/proc/${pid}/stat`)
    // comm (field 2) is parenthesised and may itself contain spaces and ')'.
    const afterComm = stat.slice(stat.lastIndexOf(')') + 1).trim().split(/\s+/)
    // afterComm[0] is field 3 (state), so field 22 is index 19.
    const ticks = Number(afterComm[19])
    const btimeLine = readText('/proc/stat').split('\n').find(line => line.startsWith('btime '))
    const btime = Number(btimeLine?.slice('btime '.length).trim())

    if (!Number.isFinite(ticks) || ticks < 0 || !Number.isFinite(btime) || btime <= 0) {return null}

    const createdAt = btime + ticks / LINUX_USER_HZ

    return Number.isFinite(createdAt) && createdAt > 0 ? createdAt : null
  } catch {
    return null
  }
}

/** macOS process creation epoch from `ps -o lstart=` (no third-party deps). */
export async function queryDarwinProcessCreatedAt(
  pid: number,
  { run = runProbe, timeoutMs = DEFAULT_PROBE_TIMEOUT_MS }: { run?: RunProbe; timeoutMs?: number } = {}
): Promise<number | null> {
  if (!Number.isInteger(pid) || pid <= 0) {return null}

  let raw: string

  try {
    raw = (await run('/bin/ps', ['-o', 'lstart=', '-p', String(pid)], timeoutMs)).trim()
  } catch {
    return null
  }

  if (!raw) {return null}

  const parsed = Date.parse(raw)

  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed / 1000) : null
}

/**
 * Process creation epoch on whichever platform we are running on.
 *
 * Every marker reader treats `null` as "identity unknown", which keeps a claim
 * blocking until the advisory age ceiling instead of forever.
 */
export async function queryProcessCreatedAt(
  pid: number,
  { platform = process.platform, run = runProbe, timeoutMs = DEFAULT_PROBE_TIMEOUT_MS }: QueryOptions = {}
): Promise<number | null> {
  if (platform === 'win32') {
    return queryWindowsProcessCreatedAt(pid, { platform, run, timeoutMs })
  }

  if (platform === 'darwin') {
    return queryDarwinProcessCreatedAt(pid, { run, timeoutMs })
  }

  if (platform === 'linux') {
    return readLinuxProcessCreatedAt(pid)
  }

  return null
}

/**
 * Adapt the async OS query to synchronous marker readers. The first read (and
 * every query failure) returns unknown/null, which keeps the gate closed. A
 * later poll may use the short-lived exact result. Positive updater adoption
 * should call queryWindowsProcessCreatedAt directly instead of this cache.
 */
export function createCachedWindowsProcessCreateTimeProbe({
  cacheMs = DEFAULT_CACHE_MS,
  now = Date.now,
  query = queryProcessCreatedAt
}: CacheOptions = {}): (pid: number) => number | null {
  const entries = new Map<number, { pending: boolean; validUntil: number; value: number | null }>()

  return (pid: number): number | null => {
    if (!Number.isInteger(pid) || pid <= 0) {return null}
    const at = now()
    const existing = entries.get(pid)

    if (existing && !existing.pending && at <= existing.validUntil) {return existing.value}

    if (!existing?.pending) {
      entries.set(pid, { pending: true, validUntil: at, value: null })
      void query(pid).then(
        value => entries.set(pid, { pending: false, validUntil: now() + cacheMs, value }),
        () => entries.set(pid, { pending: false, validUntil: now() + cacheMs, value: null })
      )
    }

    return null
  }
}

export const getCachedWindowsProcessCreatedAt = createCachedWindowsProcessCreateTimeProbe()
