/**
 * Multi-connection, resumable downloader for the main process.
 *
 * Chromium's `will-download` gives a single sequential connection: no progress
 * the renderer can read, no rate, no resume, and a large file on a flaky link
 * restarts from zero. `pm/downloader.py` already solved this for installs and
 * model pulls (Range requests, worker threads, a durable resume bitmap) — this
 * is the desktop counterpart for user-initiated file downloads.
 *
 * Design notes:
 *
 * - **Sparse writes into one file.** Each worker opens the same destination
 *   once (`r+`, pre-sized with `truncate`) and writes at its own offset, so a
 *   finished download is already the final file — no concatenation step, and a
 *   killed process leaves a file whose valid regions are known.
 * - **Resume state is a sidecar.** `<dest>.hermesdl.json` records every
 *   segment's `[start, end)` and how many bytes of it are durable. Only the
 *   missing tail of each segment is re-requested; a complete sidecar means the
 *   previous run finished.
 * - **`Accept-Ranges` decides the strategy.** No Range support (or a server
 *   that ignores the header and answers 200) degrades to one connection — the
 *   code checks for `206` rather than trusting the advertisement.
 * - **The speed sample is a sliding window, not a cumulative average**, so a
 *   stall shows up as a falling rate instead of looking healthy for minutes.
 * - No `electron` import: the manager is unit-tested under plain Node.
 */

import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import http from 'node:http'
import https from 'node:https'
import path from 'node:path'
import { URL } from 'node:url'

export type DownloadStatus =
  | 'queued'
  | 'probing'
  | 'downloading'
  | 'pausing'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'canceled'

export interface DownloadProgress {
  id: string
  url: string
  filename: string
  savePath: string
  totalBytes: number
  receivedBytes: number
  /** Bytes per second over the sampling window; 0 while stalled or probing. */
  speedBps: number
  /** Milliseconds remaining at the current rate; null when it cannot be estimated. */
  etaMs: number | null
  status: DownloadStatus
  /** Live connections — how many Range workers are running (1 = single-stream). */
  connections: number
  /** True when the server answered 206 and the transfer is segmented. */
  ranged: boolean
  error?: string
}

interface Segment {
  start: number
  end: number
  /** Bytes of this segment already written to the destination. */
  done: number
}

interface Task {
  id: string
  url: string
  filename: string
  savePath: string
  /** Request headers for every probe/segment request (Electron session cookies). */
  headers: Record<string, string>
  totalBytes: number
  segments: Segment[]
  status: DownloadStatus
  ranged: boolean
  error?: string
  /** Aborts the in-flight requests for pause/cancel. */
  controller: AbortController | null
  receivedBytes: number
  /** [timestamp, bytesSoFar] samples for the sliding-window rate. */
  samples: Array<[number, number]>
  speedBps: number
  /** Throttle for chunk-driven progress pushes (see maybeEmit). */
  lastEmit: number
  handle: fsp.FileHandle | null
  /** Set once the caller asked to stop; distinguishes pause from failure. */
  stopReason: 'pause' | 'cancel' | null
  /** Resolves when the run loop has unwound (state persisted, handle closed).
   *  `pause()`/`cancel()` await this so the reported status never runs ahead of
   *  the durable resume point — a kill in that window used to lose the
   *  sidecar write while the UI already said "paused". */
  finished: Promise<void>
  settleFinished: () => void
}

export interface DownloadOptions {
  /** Overrides the filename derived from the URL / Content-Disposition. */
  filename?: string
  /** Directory to save into. Defaults to the OS downloads folder via `defaultDir`. */
  directory?: string
  /** Range worker count cap. Default 4; 1 forces a single stream. */
  connections?: number
  /** Fixed destination path (skips directory/filename resolution). */
  savePath?: string
  /** Extra request headers — the caller forwards the Electron session's
   *  cookies here, because a re-fetched URL does not inherit Chromium's jar. */
  headers?: Record<string, string>
}

const MAX_CONNECTIONS = 8
const MIN_SEGMENT_BYTES = 256 * 1024
const SPEED_WINDOW_MS = 3000
const EMIT_INTERVAL_MS = 200
const SIDECAR_SUFFIX = '.hermesdl.json'
const USER_AGENT = 'HermesAgent-Desktop-Downloader'

function sidecarPath(savePath: string) {
  return `${savePath}${SIDECAR_SUFFIX}`
}

/** Filename from Content-Disposition (RFC 5987 `filename*` first), else the URL's last segment. */
export function filenameFromHeaders(url: string, headers: http.IncomingHttpHeaders): string {
  const disposition = String(headers['content-disposition'] || '')
  const encoded = /filename\*\s*=\s*(?:UTF-8|utf-8)''([^;]+)/i.exec(disposition)

  if (encoded) {
    try {
      return path.basename(decodeURIComponent(encoded[1].trim()))
    } catch {
      // Fall through to the plain form.
    }
  }

  const plain = /filename\s*=\s*"?([^";]+)"?/i.exec(disposition)

  if (plain) {
    return path.basename(plain[1].trim())
  }

  try {
    const parsed = new URL(url)
    const last = decodeURIComponent(parsed.pathname.split('/').filter(Boolean).pop() || '')

    return last || 'download'
  } catch {
    return 'download'
  }
}

function request(target: string, options: { method?: string; headers?: Record<string, string>; signal?: AbortSignal } = {}) {
  const url = new URL(target)
  const transport = url.protocol === 'https:' ? https : http
  const headers = { 'user-agent': USER_AGENT, ...(options.headers || {}) }

  return new Promise<http.IncomingMessage>((resolve, reject) => {
    const req = transport.request(
      url,
      { headers, method: options.method || 'GET', signal: options.signal },
      resolve
    )

    req.on('error', reject)
    req.end()
  })
}

/** Split a byte range into `count` segments, never below MIN_SEGMENT_BYTES. */
export function planSegments(totalBytes: number, requested: number): Segment[] {
  if (totalBytes <= 0) {
    return [{ done: 0, end: -1, start: 0 }]
  }

  const bySize = Math.max(1, Math.floor(totalBytes / MIN_SEGMENT_BYTES))
  const count = Math.max(1, Math.min(requested, MAX_CONNECTIONS, bySize))
  const size = Math.ceil(totalBytes / count)
  const segments: Segment[] = []

  for (let index = 0; index < count; index += 1) {
    const start = index * size
    const end = Math.min(totalBytes - 1, start + size - 1)

    if (start > end) {
      break
    }

    segments.push({ done: 0, end, start })
  }

  return segments
}

export class DownloadManager extends EventEmitter {
  private readonly tasks = new Map<string, Task>()
  private readonly defaultDir: () => string
  private counter = 0

  constructor(options: { defaultDir?: () => string } = {}) {
    super()
    this.defaultDir =
      options.defaultDir ||
      (() => {
        // Electron's app.getPath('downloads') is injected by main.ts; the
        // fallback keeps the manager runnable under plain Node for tests.
        return path.join(process.env.HOME || process.cwd(), 'Downloads')
      })
  }

  list(): DownloadProgress[] {
    return [...this.tasks.values()].map(task => this.snapshot(task))
  }

  get(id: string): DownloadProgress | undefined {
    const task = this.tasks.get(id)

    return task ? this.snapshot(task) : undefined
  }

  async start(url: string, options: DownloadOptions = {}): Promise<DownloadProgress> {
    if (!/^https?:/i.test(url)) {
      throw new Error(`Unsupported download URL: ${url}`)
    }

    this.counter += 1
    const id = `dl-${Date.now().toString(36)}-${this.counter}`
    const directory = options.directory || this.defaultDir()
    const filename = options.filename || filenameFromHeaders(url, {})
    const savePath = options.savePath || path.join(directory, filename)
    let settleFinished = () => undefined as void

    const finished = new Promise<void>(resolve => {
      settleFinished = resolve
    })

    const task: Task = {
      controller: null,
      error: undefined,
      finished,
      settleFinished,
      filename: path.basename(savePath),
      handle: null,
      id,
      receivedBytes: 0,
      lastEmit: 0,
      samples: [],
      savePath,
      headers: { ...(options.headers || {}) },
      segments: [],
      speedBps: 0,
      status: 'queued',
      stopReason: null,
      totalBytes: 0,
      url,
      ranged: false
    }

    this.tasks.set(id, task)
    this.emitProgress(task)

    // Probe + open in the background so the caller gets an id immediately and
    // the UI can render a row in `probing` state.
    void this.run(task, options.connections ?? 4).catch(error => {
      this.fail(task, error instanceof Error ? error.message : String(error))
    })

    return this.snapshot(task)
  }

  async pause(id: string) {
    const task = this.tasks.get(id)

    if (
      !task ||
      task.status === 'completed' ||
      task.status === 'canceled' ||
      task.status === 'paused'
    ) {
      return
    }

    task.stopReason = 'pause'
    // `pausing` is honest about what is happening: the abort is in flight and
    // the resume point has not been written yet.
    task.status = 'pausing'
    this.emitProgress(task)
    task.controller?.abort()
    await task.finished

    // A stop requested during the probe phase unwinds without entering the
    // chunk loop, so the paused state is normalized here for both paths.
    if (task.stopReason === 'pause') {
      if ((task.status as string) !== 'paused') {
        task.status = 'paused'
      }

      this.emitProgress(task)
    }
  }

  async resume(id: string, options: { connections?: number } = {}) {
    const task = this.tasks.get(id)

    if (!task || task.status === 'downloading' || task.status === 'pausing' || task.status === 'completed') {
      return
    }

    task.stopReason = null
    task.error = undefined
    task.status = 'downloading'
    this.emitProgress(task)

    try {
      await this.run(task, options.connections ?? 4, { resumed: true })
    } catch (error) {
      this.fail(task, error instanceof Error ? error.message : String(error))
    }
  }

  async cancel(id: string) {
    const task = this.tasks.get(id)

    if (!task) {
      return
    }

    task.stopReason = 'cancel'
    task.controller?.abort()

    if (task.status === 'downloading' || task.status === 'probing' || task.status === 'pausing') {
      await task.finished
    }

    // Clean up before reporting `canceled`: the row must not disappear from
    // disk listing while the partial file and its sidecar are still there.
    await this.cleanup(task, { removeFile: true })
    task.status = 'canceled'
    this.emitProgress(task)
  }

  /** Drop finished rows from the list (the files themselves stay). */
  dismiss(id: string) {
    const task = this.tasks.get(id)

    if (!task || task.status === 'downloading' || task.status === 'probing') {
      return
    }

    this.tasks.delete(id)
    this.emit('removed', { id })
  }

  // --------------------------------------------------------------------- run

  private async run(task: Task, connections: number, options: { resumed?: boolean } = {}) {
    // Every exit path — completion, failure, pause, or a stop requested during
    // the probe phase — must settle `finished`, or pause()/cancel() hang on it.
    this.resetFinished(task)

    try {
      await this.runInner(task, connections, options)
    } finally {
      task.settleFinished()
    }
  }

  /** Re-arm the settle promise for a fresh (or resumed) transfer. */
  private resetFinished(task: Task) {
    task.finished = new Promise<void>(resolve => {
      task.settleFinished = resolve
    })
  }

  private async runInner(task: Task, connections: number, options: { resumed?: boolean } = {}) {
    if (task.status === 'canceled') {
      return
    }

    task.status = 'probing'
    this.emitProgress(task)

    const probe = await this.probe(task)
    const acceptedRanges = probe.ranged && connections > 1 && probe.total > 0
    task.ranged = acceptedRanges
    task.totalBytes = probe.total

    if (!options.resumed) {
      await fsp.mkdir(path.dirname(task.savePath), { recursive: true })
      task.segments = await this.planFor(task, probe, acceptedRanges, connections)
      task.receivedBytes = task.segments.reduce((sum, segment) => sum + segment.done, 0)
      task.handle = await fsp.open(task.savePath, 'w+')

      if (probe.total > 0) {
        await task.handle.truncate(probe.total)
      }

      await this.persist(task)
    } else if (!task.handle) {
      await fsp.mkdir(path.dirname(task.savePath), { recursive: true })
      task.handle = await fsp.open(task.savePath, 'r+')
    }

    if (task.stopReason) {
      return
    }

    task.status = 'downloading'
    task.samples = [[Date.now(), task.receivedBytes]]
    task.lastEmit = Date.now()
    this.emitProgress(task)

    const ticker = setInterval(() => this.sample(task), 250)
    task.controller = new AbortController()

    try {
      await Promise.all(
        task.segments.map(segment =>
          this.fetchSegment(task, segment, task.controller?.signal, probe.total)
        )
      )
      await task.handle?.sync()
      await task.handle?.close()
      task.handle = null
      await fsp.rm(sidecarPath(task.savePath), { force: true })
      task.status = 'completed'
      task.speedBps = 0
      this.emitProgress(task)
      this.emit('completed', this.snapshot(task))
    } catch (error) {
      if (task.stopReason === 'pause') {
        // Persist BEFORE syncing/closing: the abort throws out of the chunk
        // loop, so `fetchSegment`'s own stop-branch never runs and the sidecar
        // would still hold the pre-transfer `done: 0` — a resume would then
        // re-download every byte it already had.
        await this.persist(task)
        task.status = 'paused'
        await task.handle?.sync().catch(() => undefined)
        await task.handle?.close().catch(() => undefined)
        task.handle = null
        this.emitProgress(task)

        return
      }

      if (task.stopReason === 'cancel') {
        return
      }

      await task.handle?.close().catch(() => undefined)
      task.handle = null
      throw error
    } finally {
      clearInterval(ticker)
      task.controller = null
    }
  }

  private async probe(task: Task): Promise<{ total: number; ranged: boolean }> {
    const baseHeaders = { accept: '*/*', ...task.headers }

    const head = await request(task.url, { headers: baseHeaders, method: 'HEAD' }).catch(
      () => null
    )

    if (head) {
      head.resume()
      const length = Number(head.headers['content-length'] || 0)
      const ranges = String(head.headers['accept-ranges'] || '').toLowerCase().includes('bytes')

      if (head.statusCode && head.statusCode < 400) {
        if (!task.filename || task.filename === 'download') {
          task.filename = filenameFromHeaders(task.url, head.headers)
        }

        if (length > 0) {
          return { ranged: ranges, total: length }
        }
      }
    }

    // No HEAD (or no length): a one-byte ranged GET tells us both.
    const probe = await request(task.url, { headers: { ...baseHeaders, range: 'bytes=0-0' } })

    if (probe.statusCode === 206) {
      const contentRange = String(probe.headers['content-range'] || '')
      const total = Number(/\/(\d+)\s*$/.exec(contentRange)?.[1] || 0)

      probe.resume()

      return { ranged: true, total }
    }

    probe.resume()

    return { ranged: false, total: Number(probe.headers['content-length'] || 0) }
  }

  private async planFor(
    task: Task,
    probe: { total: number; ranged: boolean },
    acceptedRanges: boolean,
    connections: number
  ): Promise<Segment[]> {
    if (!acceptedRanges) {
      return [{ done: 0, end: probe.total > 0 ? probe.total - 1 : -1, start: 0 }]
    }

    const previous = await this.readSidecar(task)

    if (previous && previous.totalBytes === probe.total) {
      return previous.segments
    }

    return planSegments(probe.total, connections)
  }

  private async readSidecar(task: Task): Promise<{ totalBytes: number; segments: Segment[] } | null> {
    try {
      const raw = await fsp.readFile(sidecarPath(task.savePath), 'utf8')
      const parsed = JSON.parse(raw) as { totalBytes: number; segments: Segment[] }

      if (!Array.isArray(parsed.segments) || typeof parsed.totalBytes !== 'number') {
        return null
      }

      return parsed
    } catch {
      return null
    }
  }

  private async persist(task: Task) {
    try {
      await fsp.writeFile(
        sidecarPath(task.savePath),
        JSON.stringify({ totalBytes: task.totalBytes, segments: task.segments })
      )
    } catch {
      // Losing the sidecar only costs resume ability, never the transfer.
    }
  }

  private async fetchSegment(task: Task, segment: Segment, signal: AbortSignal | undefined, total: number) {
    const start = segment.start + segment.done
    const end = segment.end

    if (end >= 0 && start > end) {
      return
    }

    const headers: Record<string, string> = { accept: '*/*', ...task.headers }

    if (task.ranged) {
      headers.range = `bytes=${start}-${end >= 0 ? end : ''}`
    }

    const response = await request(task.url, { headers, signal })

    if (!response.statusCode || response.statusCode >= 400) {
      response.resume()
      throw new Error(`HTTP ${response.statusCode} for ${task.url}`)
    }

    if (task.ranged && response.statusCode !== 206) {
      // The server ignored Range. Everything written under the segmented plan
      // is now suspect, so restart this task as a single stream rather than
      // corrupting the file with overlapping offsets.
      response.resume()
      throw new Error('Server ignored the Range request')
    }

    let position = start

    for await (const chunk of response) {
      if (task.stopReason) {
        break
      }

      const buffer = chunk as Buffer

      if (task.handle) {
        await task.handle.write(buffer, 0, buffer.length, position)
      }

      position += buffer.length
      segment.done = position - segment.start
      task.receivedBytes += buffer.length
      this.maybeEmit(task)
    }

    if (task.stopReason) {
      await this.persist(task)

      // Resolve, not reject: the run() loop treats a stop as a normal unwind.
      return
    }

    if (total > 0 && position <= end && end >= 0) {
      throw new Error(`Connection closed early at byte ${position} of ${end}`)
    }
  }

  /**
   * Push progress off the data path, at most every EMIT_INTERVAL_MS.
   *
   * A fixed 500 ms tick alone means a fast LAN transfer (a few hundred ms end
   * to end) produces a single "downloading" frame and the bar jumps 0 -> 100.
   * Emitting on bytes written keeps the rate and the bar honest at any speed,
   * while the throttle bounds IPC chatter on a 10 GbE link.
   */
  private maybeEmit(task: Task) {
    const now = Date.now()

    if (now - task.lastEmit < EMIT_INTERVAL_MS) {
      return
    }

    task.lastEmit = now
    this.refreshRate(task)
    this.emitProgress(task)
  }

  /** Sliding-window rate; shared by the ticker and the chunk-driven emitter. */
  private refreshRate(task: Task) {
    const now = Date.now()
    task.samples.push([now, task.receivedBytes])

    while (task.samples.length > 1 && now - task.samples[0][0] > SPEED_WINDOW_MS) {
      task.samples.shift()
    }

    const [firstTime, firstBytes] = task.samples[0]
    const elapsed = (now - firstTime) / 1000

    task.speedBps = elapsed > 0.15 ? Math.max(0, (task.receivedBytes - firstBytes) / elapsed) : 0
  }

  private sample(task: Task) {
    this.refreshRate(task)
    this.emitProgress(task)
  }

  private fail(task: Task, message: string) {
    task.status = 'failed'
    task.error = message
    task.speedBps = 0
    void task.handle?.close().catch(() => undefined)
    task.handle = null
    this.emitProgress(task)
    this.emit('failed', this.snapshot(task))
  }

  private async cleanup(task: Task, options: { removeFile: boolean }) {
    await task.handle?.close().catch(() => undefined)
    task.handle = null

    if (options.removeFile) {
      await fsp.rm(task.savePath, { force: true }).catch(() => undefined)
      await fsp.rm(sidecarPath(task.savePath), { force: true }).catch(() => undefined)
    }
  }

  private snapshot(task: Task): DownloadProgress {
    const remaining = task.totalBytes > 0 ? task.totalBytes - task.receivedBytes : null

    return {
      connections:
        task.status === 'downloading' ? Math.max(1, task.segments.length) : 0,
      error: task.error,
      etaMs:
        remaining !== null && task.speedBps > 1024 && task.status === 'downloading'
          ? Math.round((remaining / task.speedBps) * 1000)
          : null,
      filename: task.filename,
      id: task.id,
      ranged: task.ranged,
      receivedBytes: task.receivedBytes,
      savePath: task.savePath,
      speedBps: task.status === 'downloading' ? task.speedBps : 0,
      status: task.status,
      totalBytes: task.totalBytes,
      url: task.url
    }
  }

  private emitProgress(task: Task) {
    this.emit('progress', this.snapshot(task))
  }
}

export function createDownloadManager(options: { defaultDir?: () => string } = {}) {
  return new DownloadManager(options)
}

/** True when a Chromium-initiated download can be re-fetched by this manager. */
export function isRehostableDownload(itemUrl: string): boolean {
  return /^https?:\/\//i.test(itemUrl)
}

export { fs as __fs }
