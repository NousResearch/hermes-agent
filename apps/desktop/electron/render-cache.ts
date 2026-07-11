/**
 * Render-cache writer + reader (Phase 1 of the desktop startup-latency work).
 *
 * Persists the last-painted UI state (session-list page, status snapshot,
 * active-session transcript tail) as three small JSON files under
 * `<userData>/render-cache/`, so a cold launch can paint last-known-good UI
 * immediately and revalidate against the live gateway in the background (SWR).
 * Spec: plans/2026-07-10_desktop-startup-latency-SPEC.md (v0.2, Opus-approved).
 *
 * Invariants owned here:
 *  - I3  fail-open: a corrupt/missing/mismatched cache reads as null (boot
 *    proceeds exactly as today); reads/writes NEVER throw out of this module.
 *  - I4  same-trust-domain at rest: files are chmod 0600 (no broader than
 *    state.db's SQLite default), plaintext by explicit recorded decision.
 *  - I4b bounded + evictable: transcript files are culled on session delete
 *    (renderer→main IPC forward wired in main.ts), capped LRU-by-mtime at
 *    MAX_TRANSCRIPT_FILES, and a boot-time sweep culls orphans whose session
 *    no longer exists (covers a delete lost to crash/SIGKILL).
 *  - D2  debounced (>=5s) atomic writes (tmp + rename), flushed synchronously
 *    on before-quit so a clean exit never systematically loses the tail.
 *  - D5  schema envelope {schema, appVersion, gatewayUrl}: schema or
 *    gatewayUrl mismatch discards (I3 path); appVersion mismatch keeps.
 *  - D7  single-writer: only the main process constructs a writer, and the
 *    app's existing requestSingleInstanceLock guarantees one main process.
 *
 * Pure + unit-testable: dir, clock, and debounce interval are injectable; no
 * Electron imports (main.ts supplies userData path + app version).
 */

import fs from 'node:fs'
import path from 'node:path'

export const RENDER_CACHE_SCHEMA = 1
export const DEFAULT_DEBOUNCE_MS = 5_000
export const MAX_TRANSCRIPT_FILES = 200
export const TRANSCRIPT_ROW_CAP = 200

const TRANSCRIPT_PREFIX = 'transcript-'

export interface RenderCacheEnvelope<T> {
  schema: number
  appVersion: string
  gatewayUrl: string
  savedAt: string
  data: T
}

export interface RenderCacheOptions {
  /** Directory for the cache files (main.ts passes `<userData>/render-cache`). */
  dir: string
  /** App version stamped into the envelope. */
  appVersion: string
  /** Gateway URL this cache belongs to (D5 discard key). */
  gatewayUrl: string
  /** Debounce window for writes; tests shrink it. */
  debounceMs?: number
  /** Injectable clock for tests. */
  now?: () => number
  /** Optional logger (wired to rememberLog in main.ts); never throws. */
  log?: (line: string) => void
}

/** Sanitize a stored-session id into a safe filename fragment. */
function safeSessionFile(storedSessionId: string): string | null {
  const id = String(storedSessionId || '').trim()
  // Session ids are timestamp_hex shaped; refuse anything path-ish.
  if (!id || !/^[\w.-]+$/.test(id) || id.includes('..')) {
    return null
  }
  return `${TRANSCRIPT_PREFIX}${id}.json`
}

/** Extract the stored-session id back out of a transcript filename. */
export function sessionIdFromTranscriptFile(fileName: string): string | null {
  const name = String(fileName || '')
  if (!name.startsWith(TRANSCRIPT_PREFIX) || !name.endsWith('.json')) {
    return null
  }
  const id = name.slice(TRANSCRIPT_PREFIX.length, -'.json'.length)
  return id || null
}

export class RenderCache {
  private readonly dir: string
  private readonly appVersion: string
  private readonly gatewayUrl: string
  private readonly debounceMs: number
  private readonly now: () => number
  private readonly log: (line: string) => void

  /** Pending debounced payloads keyed by file name. */
  private pending = new Map<string, unknown>()
  private timer: ReturnType<typeof setTimeout> | null = null
  private lastFlushAt = 0

  constructor(opts: RenderCacheOptions) {
    this.dir = opts.dir
    this.appVersion = String(opts.appVersion || '')
    this.gatewayUrl = String(opts.gatewayUrl || '')
    this.debounceMs = opts.debounceMs ?? DEFAULT_DEBOUNCE_MS
    this.now = opts.now ?? Date.now
    this.log = opts.log ?? (() => {})
  }

  // ---------------------------------------------------------------- write path

  /** Queue the session-list page for a debounced write. */
  putSessions(data: unknown): void {
    this.queue('sessions.json', data)
  }

  /** Queue the status snapshot for a debounced write. */
  putStatus(data: unknown): void {
    this.queue('status.json', data)
  }

  /** Queue a transcript tail (rows already capped by the caller) for a session. */
  putTranscript(storedSessionId: string, rows: unknown[]): void {
    const file = safeSessionFile(storedSessionId)
    if (!file) {
      return
    }
    const capped = Array.isArray(rows) ? rows.slice(-TRANSCRIPT_ROW_CAP) : []
    this.queue(file, { storedSessionId, rows: capped })
  }

  private queue(file: string, data: unknown): void {
    this.pending.set(file, data)
    if (this.timer) {
      return
    }
    const elapsed = this.now() - this.lastFlushAt
    const wait = Math.max(0, this.debounceMs - Math.max(0, elapsed))
    this.timer = setTimeout(() => {
      this.timer = null
      this.flush()
    }, wait)
    // Never keep the process alive just for a cache write.
    if (typeof (this.timer as any)?.unref === 'function') {
      ;(this.timer as any).unref()
    }
  }

  /**
   * Write all pending payloads NOW, atomically (tmp + rename), mode 0600.
   * Synchronous by design so `before-quit` can call it and a clean exit never
   * loses the tail (D2/AC8). Never throws (I3: full-disk/EACCES → logged).
   */
  flush(): void {
    if (this.timer) {
      clearTimeout(this.timer)
      this.timer = null
    }
    if (this.pending.size === 0) {
      return
    }
    const entries = [...this.pending.entries()]
    this.pending.clear()
    this.lastFlushAt = this.now()
    for (const [file, data] of entries) {
      this.writeOne(file, data)
    }
  }

  private writeOne(file: string, data: unknown): void {
    try {
      fs.mkdirSync(this.dir, { recursive: true, mode: 0o700 })
      const envelope: RenderCacheEnvelope<unknown> = {
        schema: RENDER_CACHE_SCHEMA,
        appVersion: this.appVersion,
        gatewayUrl: this.gatewayUrl,
        savedAt: new Date(this.now()).toISOString(),
        data
      }
      const target = path.join(this.dir, file)
      const tmp = `${target}.tmp-${process.pid}`
      fs.writeFileSync(tmp, JSON.stringify(envelope), { mode: 0o600 })
      fs.renameSync(tmp, target)
      // rename preserves the tmp's 0600; assert it anyway (umask-proof, AC6a).
      fs.chmodSync(target, 0o600)
    } catch (error: any) {
      // Full disk / EACCES / read-only volume: log, never crash (I3).
      this.log(`[render-cache] write failed for ${file}: ${error?.message || error}`)
    }
  }

  // ----------------------------------------------------------------- read path

  /**
   * Read + validate one cache file. Returns the inner data, or null when the
   * file is missing, unparseable, schema-mismatched, or belongs to a different
   * gateway (D5). NEVER throws (I3 fail-open).
   */
  private readOne<T>(file: string): T | null {
    try {
      const raw = fs.readFileSync(path.join(this.dir, file), 'utf8')
      const parsed = JSON.parse(raw) as RenderCacheEnvelope<T>
      if (!parsed || typeof parsed !== 'object') {
        return null
      }
      if (parsed.schema !== RENDER_CACHE_SCHEMA) {
        return null // schema mismatch → discard (D5)
      }
      if (parsed.gatewayUrl !== this.gatewayUrl) {
        return null // different gateway wrote this → discard (D5)
      }
      // appVersion mismatch is deliberately KEPT (schema is the contract, D5).
      return parsed.data ?? null
    } catch {
      return null
    }
  }

  readSessions<T = unknown>(): T | null {
    return this.readOne<T>('sessions.json')
  }

  readStatus<T = unknown>(): T | null {
    return this.readOne<T>('status.json')
  }

  readTranscript<T = unknown>(storedSessionId: string): T | null {
    const file = safeSessionFile(storedSessionId)
    return file ? this.readOne<T>(file) : null
  }

  // ------------------------------------------------------------- eviction (I4b)

  /**
   * Cull a deleted session's transcript file. Called from the renderer→main
   * IPC forward when the renderer observes a session delete. Never throws.
   */
  cullSession(storedSessionId: string): void {
    const file = safeSessionFile(storedSessionId)
    if (!file) {
      return
    }
    this.pending.delete(file)
    try {
      fs.rmSync(path.join(this.dir, file), { force: true })
    } catch (error: any) {
      this.log(`[render-cache] cull failed for ${file}: ${error?.message || error}`)
    }
  }

  /**
   * Enforce the transcript-file COUNT cap: keep the newest `max` by mtime,
   * delete the rest (LRU). Never throws.
   */
  enforceTranscriptCap(max: number = MAX_TRANSCRIPT_FILES): void {
    try {
      const files = this.listTranscriptFiles()
      if (files.length <= max) {
        return
      }
      const withTimes = files
        .map(name => {
          try {
            return { name, mtime: fs.statSync(path.join(this.dir, name)).mtimeMs }
          } catch {
            return { name, mtime: 0 }
          }
        })
        .sort((a, b) => b.mtime - a.mtime)
      for (const { name } of withTimes.slice(max)) {
        try {
          fs.rmSync(path.join(this.dir, name), { force: true })
        } catch {
          // ignore — cap re-runs on the next write cycle
        }
      }
    } catch (error: any) {
      this.log(`[render-cache] cap enforcement failed: ${error?.message || error}`)
    }
  }

  /**
   * Boot-time sweep (I4b belt-and-suspenders): delete any transcript file whose
   * session is NOT in the live session-id set — covers a delete whose IPC
   * forward was lost to a crash/SIGKILL window. Also enforces the count cap.
   * Never throws.
   */
  sweepAgainstLiveSessions(liveSessionIds: Iterable<string>): number {
    let culled = 0
    try {
      const live = new Set<string>()
      for (const id of liveSessionIds) {
        live.add(String(id))
      }
      for (const name of this.listTranscriptFiles()) {
        const id = sessionIdFromTranscriptFile(name)
        if (id && !live.has(id)) {
          try {
            fs.rmSync(path.join(this.dir, name), { force: true })
            culled += 1
          } catch {
            // ignore; retried next boot
          }
        }
      }
      this.enforceTranscriptCap()
    } catch (error: any) {
      this.log(`[render-cache] boot sweep failed: ${error?.message || error}`)
    }
    return culled
  }

  private listTranscriptFiles(): string[] {
    try {
      return fs.readdirSync(this.dir).filter(n => n.startsWith(TRANSCRIPT_PREFIX) && n.endsWith('.json'))
    } catch {
      return []
    }
  }

  /** Test/observability helper: how many transcript files exist right now. */
  transcriptFileCount(): number {
    return this.listTranscriptFiles().length
  }
}
