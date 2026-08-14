// Main-process half of the hitch-capture controller (U3, KTD2/KTD3).
//
// Main is the only process that can reach BOTH the renderers and the gateway,
// so it owns the capture: it mints the capture id and the wall-clock anchor,
// pushes them down to every chat window over IPC, arms the gateway over the
// existing authenticated channel, records its own boundary signals into a
// bounded ring, and — on stop — gathers all three streams into one bundle for
// the exporter (U4).
//
// Two invariants shape the design:
//
//   * Zero cost while disarmed (KTD2). Nothing samples, nothing accumulates.
//     `app.getAppMetrics()` walks every process in the app and is far too
//     expensive to poll around the clock, so the sampler only exists between
//     start and stop, and window responsiveness edges are dropped on the floor
//     unless a capture is running.
//   * Correlate by capture id + per-process monotonic clocks (KTD3). Each
//     process stamps its own monotonic clock and reports the anchor it had at
//     arm time; nothing here rewrites another process's timestamps. The
//     exporter aligns the streams afterwards.
//
// Sanitization: main records durations, counts, pids and error classes only —
// never request bodies, responses, URLs with credentials, or argv. The
// transport recorder deliberately keeps a coarse route prefix instead of the
// full path, because Hermes REST paths carry session ids.
//
// Electron-free (windows, metrics, timers and the gateway client are all
// injected) so it unit-tests under the `electron` vitest project the same way
// stream-throttle.ts and session-windows.ts do.

import type { GatewayAbsentReason, GatewayDiagnosticsClient, GatewayDiagnosticsStream } from './diagnostics-gateway'

/** Ring window, matching the renderer's (~300s around the reported hitch). */
const DEFAULT_MAX_AGE_MS = 300_000
/** Count cap. Main's streams are sparse next to the renderer's long frames. */
const DEFAULT_MAX_EVENTS = 4_000
/** `app.getAppMetrics()` cadence while armed. Low on purpose: this call is not
 * cheap, and memory/CPU drift is the slow-moving signal in the bundle. */
const DEFAULT_METRICS_INTERVAL_MS = 5_000
/** How long a renderer gets to answer a snapshot pull before it is skipped —
 * a renderer wedged by the very hitch under investigation must not hang the
 * export. */
const DEFAULT_COLLECT_TIMEOUT_MS = 3_000

/** main -> renderer: arm/disarm/collect. Mirrored by preload.ts. */
const ARM_CHANNEL = 'diagnostics:arm'
const DISARM_CHANNEL = 'diagnostics:disarm'
const COLLECT_CHANNEL = 'diagnostics:collect'
/** renderer -> main: the answer to one `diagnostics:collect` request. */
const COLLECT_RESULT_CHANNEL = 'diagnostics:collect:result'

export interface MainDiagnosticsEventBase {
  /** Main's monotonic clock, same units as `anchors.mainMonotonicAnchorMs`. */
  t: number
  type: string
}

export interface WindowResponsivenessEvent extends MainDiagnosticsEventBase {
  type: 'window_responsive' | 'window_unresponsive'
  windowId: number
  /** How long the window was unresponsive; only on the 'responsive' edge. */
  blockedMs?: number
}

export interface ProcessMetricSample {
  pid: number
  /** Electron's process type ('Browser', 'Tab', 'GPU', 'Utility', ...). */
  type: string
  percentCpu: number
  workingSetKb: number
}

export interface AppMetricsEvent extends MainDiagnosticsEventBase {
  type: 'app_metrics'
  processes: ProcessMetricSample[]
}

export interface TransportErrorEvent extends MainDiagnosticsEventBase {
  type: 'transport_error'
  /** IPC channel the failure surfaced on, e.g. 'hermes:api'. */
  channel: string
  /** Coarse route prefix ('/api/sessions'), never the full path + query. */
  route: null | string
  durationMs: number
  /** 'timeout' | 'http_<status>' | 'connect' | 'error' — no messages. */
  errorClass: string
}

export type MainDiagnosticsEvent = AppMetricsEvent | TransportErrorEvent | WindowResponsivenessEvent

export interface RendererCaptureSnapshot {
  captureId: string
  wallClockAnchorMs: number
  monotonicAnchorMs: number
  events: unknown[]
  droppedEvents: number
}

export type RendererCaptureStream = RendererCaptureSnapshot & { windowId: number }

export interface DiagnosticsCaptureBundle {
  captureId: string
  anchors: {
    wallClockAnchorMs: number
    mainMonotonicAnchorMs: number
  }
  renderer: RendererCaptureStream[]
  main: MainDiagnosticsEvent[]
  /** Main-ring events evicted by the caps, so the bundle is honest. */
  mainDropped: number
  gateway: GatewayDiagnosticsStream | { absent: GatewayAbsentReason }
}

export interface CaptureWebContentsLike {
  isDestroyed(): boolean
  send(channel: string, payload?: unknown): void
  on?(event: string, listener: () => void): void
  removeListener?(event: string, listener: () => void): void
}

export interface CaptureWindowLike {
  id: number
  isDestroyed(): boolean
  webContents?: CaptureWebContentsLike | null
  on?(event: string, listener: () => void): void
}

/** Raw `app.getAppMetrics()` entry — only the fields main keeps. */
export interface RawAppMetric {
  pid?: number
  type?: string
  cpu?: { percentCPUUsage?: number }
  memory?: { workingSetSize?: number }
}

interface TimersLike {
  clearInterval(handle: unknown): void
  clearTimeout(handle: unknown): void
  setInterval(fn: () => void, ms: number): unknown
  setTimeout(fn: () => void, ms: number): unknown
}

export interface CaptureControllerOptions {
  /** Monotonic clock for main's own stream (`performance.now()` in prod). */
  now: () => number
  /** Wall clock at capture start — the anchor every process aligns onto. */
  wallClock?: () => number
  randomUUID?: () => string
  /** Every window that should be armed: primary + session + instance windows. */
  listWindows: () => CaptureWindowLike[]
  getAppMetrics?: () => RawAppMetric[]
  /** Null when this build has no gateway client at all. */
  gateway?: GatewayDiagnosticsClient | null
  /**
   * True when the active connection is a remote/SSH gateway. The gateway ring
   * is supported only for the locally-spawned backend, so a remote connection
   * skips arming entirely and the stream is marked absent (`remote-gateway`).
   */
  isRemoteGateway?: () => boolean
  timers?: TimersLike
  maxEvents?: number
  maxAgeMs?: number
  metricsIntervalMs?: number
  collectTimeoutMs?: number
}

export interface CaptureController {
  /** Track a window: responsiveness edges, plus live-arming a window that
   * opened mid-capture. Safe to call for every window in the app. */
  attachWindow(win: CaptureWindowLike): void
  isArmed(): boolean
  captureId(): null | string
  /** Arm renderers + gateway + the main sampler. Idempotent-ish: a second
   * call while armed returns the running capture's descriptor. */
  start(): Promise<{ captureId: string; wallClockAnchorMs: number; mainMonotonicAnchorMs: number }>
  /** Disarm everything and gather the bundle. Null when nothing was armed. */
  stop(): Promise<DiagnosticsCaptureBundle | null>
  /** The last completed bundle — U4's export/UI reads this. */
  lastCapture(): DiagnosticsCaptureBundle | null
  /** Wire to `ipcMain.on('diagnostics:collect:result')`. */
  handleCollectResult(payload: unknown): void
  /** Record a renderer->backend transport failure (durations/classes only). */
  recordTransportError(input: { channel: string; path?: null | string; durationMs: number; error: unknown }): void
}

/**
 * Classify a transport failure without keeping its message. `fetchJson` rejects
 * HTTP errors as `Error('<status>: <body>')` and its own timeout as "Timed out
 * connecting to Hermes backend after Nms"; Node socket failures carry a `code`.
 */
export function classifyTransportError(error: unknown): string {
  const code = (error as { code?: unknown })?.code

  if (typeof code === 'string' && code) {
    return code === 'ETIMEDOUT' ? 'timeout' : `connect_${code}`
  }

  const message = error instanceof Error ? error.message : String(error ?? '')

  if (/timed out/i.test(message)) {
    return 'timeout'
  }

  const status = /^\s*(\d{3})\b/.exec(message)?.[1]

  if (status) {
    return `http_${status}`
  }

  return 'error'
}

/**
 * Reduce a REST path to a coarse route prefix: two segments, no query, no
 * fragment. `/api/sessions/abc123/messages?x=y` -> `/api/sessions`. Session ids
 * and tokens live in the parts this drops.
 */
export function routePrefix(rawPath: unknown): null | string {
  const text = typeof rawPath === 'string' ? rawPath : ''

  if (!text) {
    return null
  }

  const withoutQuery = text.split(/[?#]/, 1)[0]
  const segments = withoutQuery.split('/').filter(Boolean).slice(0, 2)

  return segments.length ? `/${segments.join('/')}` : '/'
}

/** Keep only the numeric shape of a metrics sample — no process names or
 * command lines (they routinely carry paths and, for utility processes,
 * extension identities). */
export function sanitizeAppMetrics(metrics: RawAppMetric[] | undefined): ProcessMetricSample[] {
  if (!Array.isArray(metrics)) {
    return []
  }

  return metrics.map(entry => ({
    pid: Number(entry?.pid) || 0,
    type: typeof entry?.type === 'string' ? entry.type : 'unknown',
    percentCpu: Math.round((Number(entry?.cpu?.percentCPUUsage) || 0) * 100) / 100,
    workingSetKb: Math.round(Number(entry?.memory?.workingSetSize) || 0)
  }))
}

/**
 * Bounded, oldest-first event store for the main stream. Count cap + age cap,
 * both enforced on push so the ring can never grow between captures.
 */
export function createMainRing(limits: { maxEvents: number; maxAgeMs: number }) {
  const events: MainDiagnosticsEvent[] = []
  let dropped = 0

  function trim(nowT: number) {
    while (events.length && (events.length > limits.maxEvents || nowT - events[0].t > limits.maxAgeMs)) {
      events.shift()
      dropped += 1
    }
  }

  return {
    push(event: MainDiagnosticsEvent) {
      events.push(event)
      trim(event.t)
    },
    entries: () => events.slice(),
    get dropped() {
      return dropped
    },
    get size() {
      return events.length
    },
    reset() {
      events.length = 0
      dropped = 0
    }
  }
}

const defaultTimers: TimersLike = {
  clearInterval: handle => clearInterval(handle as never),
  clearTimeout: handle => clearTimeout(handle as never),
  setInterval,
  setTimeout
}

function normalizeRendererSnapshot(value: unknown): RendererCaptureSnapshot | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const snapshot = value as Partial<RendererCaptureSnapshot>

  if (typeof snapshot.captureId !== 'string' || !snapshot.captureId) {
    return null
  }

  return {
    captureId: snapshot.captureId,
    wallClockAnchorMs: Number(snapshot.wallClockAnchorMs) || 0,
    monotonicAnchorMs: Number(snapshot.monotonicAnchorMs) || 0,
    events: Array.isArray(snapshot.events) ? snapshot.events : [],
    droppedEvents: Number(snapshot.droppedEvents) || 0
  }
}

export function createCaptureController(options: CaptureControllerOptions): CaptureController {
  const now = options.now
  const wallClock = options.wallClock ?? (() => Date.now())
  const randomUUID = options.randomUUID ?? (() => globalThis.crypto.randomUUID())
  const timers = options.timers ?? defaultTimers
  const metricsIntervalMs = options.metricsIntervalMs ?? DEFAULT_METRICS_INTERVAL_MS
  const collectTimeoutMs = options.collectTimeoutMs ?? DEFAULT_COLLECT_TIMEOUT_MS

  const ring = createMainRing({
    maxEvents: options.maxEvents ?? DEFAULT_MAX_EVENTS,
    maxAgeMs: options.maxAgeMs ?? DEFAULT_MAX_AGE_MS
  })

  const attached = new Set<CaptureWindowLike>()
  // Windows already told about the running capture. Arming is reached from two
  // directions — the start() sweep and a window attaching mid-capture — and a
  // renderer that gets armed twice would reset its ring mid-hitch.
  const armedWindows = new Set<CaptureWindowLike>()
  const pendingCollects = new Map<string, (snapshot: null | RendererCaptureSnapshot) => void>()
  const unresponsiveSince = new Map<number, number>()

  let armed = false
  let activeCaptureId: null | string = null
  let wallClockAnchorMs = 0
  let mainMonotonicAnchorMs = 0
  let metricsHandle: unknown = null
  let gatewayAbsent: GatewayAbsentReason | null = 'disabled'
  let lastBundle: DiagnosticsCaptureBundle | null = null
  let collectSeq = 0

  function liveWebContents(win: CaptureWindowLike): CaptureWebContentsLike | null {
    if (win.isDestroyed()) {
      return null
    }

    const contents = win.webContents

    return contents && !contents.isDestroyed() ? contents : null
  }

  function sendToWindow(win: CaptureWindowLike, channel: string, payload?: unknown) {
    const contents = liveWebContents(win)

    if (!contents) {
      return false
    }

    try {
      contents.send(channel, payload)

      return true
    } catch {
      // A window mid-teardown can throw; it is leaving the capture anyway.
      return false
    }
  }

  function sampleMetrics() {
    if (!armed || !options.getAppMetrics) {
      return
    }

    let raw: RawAppMetric[] = []

    try {
      raw = options.getAppMetrics() ?? []
    } catch {
      return
    }

    ring.push({ type: 'app_metrics', t: now(), processes: sanitizeAppMetrics(raw) })
  }

  function pushArm(win: CaptureWindowLike) {
    if (!armed || !activeCaptureId || armedWindows.has(win)) {
      return
    }

    armedWindows.add(win)
    sendToWindow(win, ARM_CHANNEL, { captureId: activeCaptureId, wallClockAnchorMs })
  }

  function attachWindow(win: CaptureWindowLike) {
    if (!win) {
      return
    }

    if (attached.has(win)) {
      // Already tracked, but a capture may have started since — arm it now.
      pushArm(win)

      return
    }

    attached.add(win)
    win.on?.('closed', () => {
      attached.delete(win)
      armedWindows.delete(win)
      unresponsiveSince.delete(win.id)
    })

    const contents = win.webContents

    contents?.on?.('unresponsive', () => {
      unresponsiveSince.set(win.id, now())

      if (armed) {
        ring.push({ type: 'window_unresponsive', t: now(), windowId: win.id })
      }
    })

    contents?.on?.('responsive', () => {
      const startedAt = unresponsiveSince.get(win.id)
      unresponsiveSince.delete(win.id)

      if (!armed) {
        return
      }

      const t = now()

      ring.push({
        type: 'window_responsive',
        t,
        windowId: win.id,
        ...(startedAt === undefined ? {} : { blockedMs: Math.max(0, Math.round(t - startedAt)) })
      })
    })

    // A window opened mid-capture still owes the bundle its stream.
    pushArm(win)
  }

  async function armGateway(captureId: string): Promise<GatewayAbsentReason | null> {
    if (options.isRemoteGateway?.()) {
      // The gateway ring is supported only for the locally-spawned backend; a
      // remote/SSH host is not asked at all (never mind that its ring would be
      // on the wrong machine's clock).
      return 'remote-gateway'
    }

    if (!options.gateway) {
      return 'disabled'
    }

    const result = await options.gateway.arm({ captureId, wallClockAnchorMs })

    return result.ok ? null : (result.reason ?? 'unavailable')
  }

  async function collectRenderer(win: CaptureWindowLike): Promise<null | RendererCaptureStream> {
    const requestId = `${activeCaptureId}:${(collectSeq += 1)}`

    const snapshot = await new Promise<null | RendererCaptureSnapshot>(resolve => {
      let settled = false

      const finish = (value: null | RendererCaptureSnapshot) => {
        if (settled) {
          return
        }

        settled = true
        pendingCollects.delete(requestId)
        timers.clearTimeout(handle)
        resolve(value)
      }

      const handle = timers.setTimeout(() => finish(null), collectTimeoutMs)

      pendingCollects.set(requestId, finish)

      if (!sendToWindow(win, COLLECT_CHANNEL, { requestId, captureId: activeCaptureId })) {
        finish(null)
      }
    })

    return snapshot ? { windowId: win.id, ...snapshot } : null
  }

  return {
    attachWindow,

    isArmed: () => armed,

    captureId: () => activeCaptureId,

    lastCapture: () => lastBundle,

    handleCollectResult(payload) {
      const body = payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : null
      const requestId = typeof body?.requestId === 'string' ? body.requestId : ''
      const resolve = requestId ? pendingCollects.get(requestId) : undefined

      resolve?.(normalizeRendererSnapshot(body?.snapshot))
    },

    recordTransportError({ channel, path, durationMs, error }) {
      if (!armed) {
        return
      }

      ring.push({
        type: 'transport_error',
        t: now(),
        channel: String(channel || 'unknown'),
        route: routePrefix(path),
        durationMs: Math.max(0, Math.round(durationMs)),
        errorClass: classifyTransportError(error)
      })
    },

    async start() {
      if (armed && activeCaptureId) {
        return { captureId: activeCaptureId, wallClockAnchorMs, mainMonotonicAnchorMs }
      }

      ring.reset()
      unresponsiveSince.clear()
      armedWindows.clear()
      activeCaptureId = randomUUID()
      wallClockAnchorMs = wallClock()
      mainMonotonicAnchorMs = now()
      armed = true

      for (const win of options.listWindows()) {
        attachWindow(win)
      }

      gatewayAbsent = await armGateway(activeCaptureId).catch(() => 'unavailable' as GatewayAbsentReason)

      // Sampling exists only between start and stop (KTD2): armed is the only
      // state in which main pays for diagnostics.
      metricsHandle = timers.setInterval(sampleMetrics, metricsIntervalMs)

      return { captureId: activeCaptureId, wallClockAnchorMs, mainMonotonicAnchorMs }
    },

    async stop() {
      if (!armed || !activeCaptureId) {
        return null
      }

      const captureIdAtStop = activeCaptureId

      if (metricsHandle !== null) {
        timers.clearInterval(metricsHandle)
        metricsHandle = null
      }

      // One last sample so the bundle brackets the hitch on both sides.
      sampleMetrics()

      const windows = options.listWindows()

      for (const win of windows) {
        attachWindow(win)
      }

      const rendererStreams = (await Promise.all(windows.map(win => collectRenderer(win)))).filter(
        (stream): stream is RendererCaptureStream => Boolean(stream)
      )

      let gateway: DiagnosticsCaptureBundle['gateway'] = { absent: gatewayAbsent ?? 'unavailable' }

      if (!gatewayAbsent && options.gateway) {
        const collected = await options.gateway.collect(captureIdAtStop).catch(() => null)

        gateway = collected?.ok && collected.stream ? collected.stream : { absent: collected?.reason ?? 'unavailable' }
      }

      if (!options.isRemoteGateway?.() && options.gateway) {
        await options.gateway.disarm()
      }

      for (const win of windows) {
        sendToWindow(win, DISARM_CHANNEL)
      }

      const bundle: DiagnosticsCaptureBundle = {
        captureId: captureIdAtStop,
        anchors: { wallClockAnchorMs, mainMonotonicAnchorMs },
        renderer: rendererStreams,
        main: ring.entries(),
        mainDropped: ring.dropped,
        gateway
      }

      armed = false
      activeCaptureId = null
      gatewayAbsent = 'disabled'
      unresponsiveSince.clear()
      armedWindows.clear()
      lastBundle = bundle

      return bundle
    }
  }
}

export {
  ARM_CHANNEL,
  COLLECT_CHANNEL,
  COLLECT_RESULT_CHANNEL,
  DEFAULT_COLLECT_TIMEOUT_MS,
  DEFAULT_MAX_AGE_MS,
  DEFAULT_MAX_EVENTS,
  DEFAULT_METRICS_INTERVAL_MS,
  DISARM_CHANNEL
}
