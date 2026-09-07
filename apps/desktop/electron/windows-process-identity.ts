import { execFile } from 'node:child_process'
import { readFileSync } from 'node:fs'

import { windowsPowerShellExecutable } from './windows-powershell-path'

const DEFAULT_PROBE_TIMEOUT_MS = 3_000
// Main polls the update marker once per second. Keep a successful async query
// available across at least one full poll interval; positive updater adoption
// still bypasses this cache and performs a fresh awaited OS query.
const DEFAULT_CACHE_MS = 2_000
// The probe answers about whatever PID a marker, a scan, or a holder list
// names, and main keeps one instance of it for the life of the app. Without a
// ceiling every PID ever asked about stays resident forever.
const DEFAULT_MAX_CACHE_ENTRIES = 256
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
      windowsPowerShellExecutable(),
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

/** One cached OS identity answer, or the in-flight query that will replace it. */
export interface ProcessIdentityCacheEntry {
  pending: boolean
  validUntil: number
  value: number | null
}

/**
 * Keep the identity cache bounded, in place.
 *
 * The cache is keyed by PID and every marker poll, scan, and holder list can
 * introduce new ones, so an unbounded Map grows for as long as the app runs.
 * Two rules, applied on every write:
 *
 *  1. A settled entry whose `validUntil` has passed is dead weight — the next
 *     read would re-query it anyway — so drop it.
 *  2. Whatever is left is trimmed to `maxEntries`, oldest write first. A Map
 *     iterates in insertion order and every write here re-inserts its key, so
 *     the head of the iteration is the least recently written entry.
 *
 * A pending entry is never swept: it is the in-flight-query de-duplication
 * token, and dropping it would let a second query for the same PID start. The
 * capacity trim still evicts it if the cache is full of pending work, so the
 * ceiling is hard either way.
 */
export function pruneProcessIdentityCache(
  entries: Map<number, ProcessIdentityCacheEntry>,
  at: number,
  maxEntries: number = DEFAULT_MAX_CACHE_ENTRIES
): void {
  for (const [pid, entry] of entries) {
    if (!entry.pending && at > entry.validUntil) {entries.delete(pid)}
  }

  for (const pid of entries.keys()) {
    if (entries.size <= maxEntries) {break}
    entries.delete(pid)
  }
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
  const entries = new Map<number, ProcessIdentityCacheEntry>()

  // Re-insert rather than update in place: insertion order is the recency
  // order pruneProcessIdentityCache evicts by.
  const write = (pid: number, entry: ProcessIdentityCacheEntry, at: number) => {
    entries.delete(pid)
    entries.set(pid, entry)
    pruneProcessIdentityCache(entries, at, DEFAULT_MAX_CACHE_ENTRIES)
  }

  const settle = (pid: number, value: number | null) => {
    const at = now()

    write(pid, { pending: false, validUntil: at + cacheMs, value }, at)
  }

  return (pid: number): number | null => {
    if (!Number.isInteger(pid) || pid <= 0) {return null}
    const at = now()
    const existing = entries.get(pid)

    if (existing && !existing.pending && at <= existing.validUntil) {return existing.value}

    if (!existing?.pending) {
      write(pid, { pending: true, validUntil: at, value: null }, at)
      void query(pid).then(
        value => settle(pid, value),
        () => settle(pid, null)
      )
    }

    return null
  }
}

export const getCachedWindowsProcessCreatedAt = createCachedWindowsProcessCreateTimeProbe()
