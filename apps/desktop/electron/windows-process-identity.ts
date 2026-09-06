import { execFile } from 'node:child_process'
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

/**
 * Adapt the async OS query to synchronous marker readers. The first read (and
 * every query failure) returns unknown/null, which keeps the gate closed. A
 * later poll may use the short-lived exact result. Positive updater adoption
 * should call queryWindowsProcessCreatedAt directly instead of this cache.
 */
export function createCachedWindowsProcessCreateTimeProbe({
  cacheMs = DEFAULT_CACHE_MS,
  now = Date.now,
  query = queryWindowsProcessCreatedAt
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
