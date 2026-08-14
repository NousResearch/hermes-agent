// Post-processing classification of a capture bundle (U4, R3).
//
// R3 asks the exported bundle to be *sufficient to classify* a hitch. This
// module is the reference reading of that bundle: it walks the three streams
// and answers "which subsystem was busy" with a small set of threshold
// heuristics. It is deliberately dumb — no statistics, no model, no tuning
// knobs in a config file — because its job is to give a reviewer a starting
// point they can immediately check against the raw JSONL beside it, not to be
// authoritative.
//
// Multi-label on purpose: a renderer that spikes *because* the gateway stalled
// and the transcript is huge should say so three times rather than pick one
// winner. `labels` is ordered by descending signal strength, and `primary` is
// simply `labels[0]` (null when nothing crossed a threshold).
//
// Every threshold below is a judgement call, and each one is named so a future
// reader can move it with intent rather than guessing which magic number did
// what. All of them describe *perceptible* trouble: the floor everywhere is
// roughly "a user would have seen this frame drop".
//
// Electron-free and pure (bundle in, summary out) so it unit-tests under the
// `electron` vitest project alongside diagnostics-capture.ts.

import type { DiagnosticsCaptureBundle } from './diagnostics-capture'

/** The five R3 buckets, plus the honest fallback. */
export type HitchLabel =
  | 'gateway-bound'
  | 'history-bound'
  | 'ipc-transport-bound'
  | 'memory-gc-bound'
  | 'renderer-bound'
  | 'unclassified'

/** A frame that took this long is a visible hitch, not jitter. Matches the
 * renderer's own "long frame" intuition (~3 dropped frames at 60Hz). */
const LONG_FRAME_MS = 50
/** A frame this long is unmistakable on its own — one is enough to label. */
const SEVERE_FRAME_MS = 300
/** Below this many long frames the capture is noise (a single 60ms frame at
 * app start is not a renderer-bound hitch). */
const MIN_LONG_FRAMES = 3

/** The gateway ring's own capture floor (`_CAPTURE_FLOOR_S` = 0.25s), so any
 * loop_drift event that reached the ring already cleared it. Restated here so
 * the classifier does not silently inherit a threshold it cannot see. */
const LOOP_DRIFT_S = 0.25
/** A single stall this long blocks every in-flight turn — label on one. */
const SEVERE_DRIFT_S = 1.0
const MIN_DRIFT_EVENTS = 2

/** Transport failures are rare enough that two inside one capture window is
 * already the story; a timeout is severe on its own (the observed 60s
 * `hermes:api` timeouts are exactly this signal). */
const MIN_TRANSPORT_ERRORS = 2

/** Heap growth across the capture that is hard to explain as churn. */
const MEMORY_GROWTH_MB = 150
/** Sitting this close to the heap limit means GC pressure regardless of slope. */
const MEMORY_PRESSURE_RATIO = 0.85
/** Fewer samples than this cannot describe a trend (5s cadence → ~15s). */
const MIN_MEMORY_SAMPLES = 3

/** A commit costing this long is what "the app hitches when I type" looks like. */
const SLOW_COMMIT_MS = 40
/** How much more expensive the long-transcript half must be before transcript
 * size — rather than flush size — is the plausible cause. */
const HISTORY_COST_RATIO = 2
const MIN_DELTA_SAMPLES = 4

export interface HitchSignal {
  label: HitchLabel
  /** Why this label fired, in reviewer-readable terms. */
  reason: string
  /** Ordering key only — comparable *within* a label, not across labels in any
   * physical sense. Higher means "more of the thing". */
  strength: number
  /** The raw counts/durations the reason was computed from. */
  evidence: Record<string, number>
}

export interface HitchClassification {
  /** Labels that crossed a threshold, strongest first. */
  labels: HitchLabel[]
  /** `labels[0]`, or null when the capture is unclassified. */
  primary: HitchLabel | null
  /** One entry per fired label, in the same order. */
  signals: HitchSignal[]
  /** Per-stream tallies, emitted whether or not anything fired — an empty
   * classification with a populated tally reads very differently from an
   * empty classification with nothing in the streams at all. */
  counts: {
    rendererLongFrames: number
    rendererMaxFrameMs: number
    rendererStreamDeltas: number
    rendererMemorySamples: number
    mainTransportErrors: number
    mainUnresponsiveEdges: number
    gatewayLoopDrifts: number
    gatewayMaxDriftS: number
    gatewayWsWriteSlow: number
  }
}

function num(value: unknown): number {
  const parsed = typeof value === 'number' ? value : Number(value)

  return Number.isFinite(parsed) ? parsed : 0
}

function eventType(event: unknown): string {
  const record = event && typeof event === 'object' ? (event as Record<string, unknown>) : null

  // Renderer/main events are tagged `type`; the gateway ring tags `kind`.
  const tag = record?.type ?? record?.kind

  return typeof tag === 'string' ? tag : ''
}

function rendererEvents(bundle: DiagnosticsCaptureBundle): unknown[] {
  return (bundle.renderer ?? []).flatMap(stream => (Array.isArray(stream.events) ? stream.events : []))
}

function gatewayEvents(bundle: DiagnosticsCaptureBundle): unknown[] {
  const gateway = bundle.gateway as { events?: unknown }

  return Array.isArray(gateway?.events) ? gateway.events : []
}

function median(values: number[]): number {
  if (!values.length) {
    return 0
  }

  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)

  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

function mean(values: number[]): number {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0
}

function round(value: number, places = 2): number {
  const factor = 10 ** places

  return Math.round(value * factor) / factor
}

/**
 * Classify one capture bundle. Pure; safe to call on a partially-populated
 * bundle (an absent gateway stream, a window that never answered its collect).
 */
export function classifyCapture(bundle: DiagnosticsCaptureBundle): HitchClassification {
  const renderer = rendererEvents(bundle)
  const main = Array.isArray(bundle.main) ? bundle.main : []
  const gateway = gatewayEvents(bundle)

  const longFrames = renderer.filter(event => eventType(event) === 'long_frame').map(event => num((event as any).ms))

  const memory = renderer
    .filter(event => eventType(event) === 'memory_sample')
    .map(event => ({ usedMb: num((event as any).usedMb), limitMb: num((event as any).limitMb) }))

  const deltas = renderer
    .filter(event => eventType(event) === 'stream_delta_applied')
    .map(event => ({
      commitMs: num((event as any).commitMs),
      writeMs: num((event as any).writeMs),
      historyMessages: num((event as any).historyMessages)
    }))

  const transportErrors = main.filter(event => event?.type === 'transport_error')
  const unresponsive = main.filter(event => event?.type === 'window_unresponsive')

  const drifts = gateway.filter(event => eventType(event) === 'loop_drift').map(event => num((event as any).drift_s))
  const wsWriteSlow = gateway.filter(event => eventType(event) === 'ws_write_slow')

  const maxFrameMs = longFrames.length ? Math.max(...longFrames) : 0
  const maxDriftS = drifts.length ? Math.max(...drifts) : 0

  const signals: HitchSignal[] = []

  // ── renderer-bound ────────────────────────────────────────────────
  // Long animation frames are direct evidence of main-thread work in the
  // renderer. Either enough of them to be a pattern, or one bad enough that
  // the pattern is beside the point.
  const spikes = longFrames.filter(ms => ms >= LONG_FRAME_MS)

  if (spikes.length >= MIN_LONG_FRAMES || maxFrameMs >= SEVERE_FRAME_MS) {
    signals.push({
      label: 'renderer-bound',
      reason: `${spikes.length} long frame(s) >= ${LONG_FRAME_MS}ms, worst ${round(maxFrameMs)}ms`,
      strength: maxFrameMs,
      evidence: { spikes: spikes.length, maxFrameMs: round(maxFrameMs), totalFrames: longFrames.length }
    })
  }

  // ── gateway-bound ─────────────────────────────────────────────────
  // The gateway's heartbeat drift is the only signal in the bundle that names
  // the backend event loop directly. `ws_write_slow` is counted as supporting
  // evidence rather than a trigger: a slow write is as often the *consumer*
  // stalling as the producer.
  const stalls = drifts.filter(driftS => driftS >= LOOP_DRIFT_S)

  if (stalls.length >= MIN_DRIFT_EVENTS || maxDriftS >= SEVERE_DRIFT_S) {
    signals.push({
      label: 'gateway-bound',
      reason: `${stalls.length} loop stall(s) >= ${LOOP_DRIFT_S}s, worst ${round(maxDriftS, 3)}s`,
      strength: maxDriftS * 1000,
      evidence: { stalls: stalls.length, maxDriftMs: round(maxDriftS * 1000), wsWriteSlow: wsWriteSlow.length }
    })
  }

  // ── ipc-transport-bound ───────────────────────────────────────────
  // Main records renderer->backend failures with a duration and an error class
  // only. A timeout means the renderer sat waiting for a reply that never came,
  // which presents to the user as a frozen pane rather than a dropped frame —
  // a different bug from either of the two above.
  const timeouts = transportErrors.filter(event => (event as any).errorClass === 'timeout')

  if (transportErrors.length >= MIN_TRANSPORT_ERRORS || timeouts.length >= 1) {
    const worstMs = transportErrors.length ? Math.max(...transportErrors.map(event => num((event as any).durationMs))) : 0

    signals.push({
      label: 'ipc-transport-bound',
      reason: `${transportErrors.length} transport failure(s) (${timeouts.length} timeout), worst ${round(worstMs)}ms`,
      strength: worstMs,
      evidence: { errors: transportErrors.length, timeouts: timeouts.length, worstMs: round(worstMs) }
    })
  }

  // ── memory-gc-bound ───────────────────────────────────────────────
  // Two independent readings of the heap samples: a growth slope across the
  // capture (a leak the GC keeps chasing), and proximity to the heap limit
  // (pressure regardless of slope). Either one labels.
  if (memory.length >= MIN_MEMORY_SAMPLES) {
    const used = memory.map(sample => sample.usedMb)
    const growthMb = Math.max(...used) - Math.min(...used)
    const limitMb = Math.max(...memory.map(sample => sample.limitMb))
    const ratio = limitMb > 0 ? Math.max(...used) / limitMb : 0

    if (growthMb >= MEMORY_GROWTH_MB || ratio >= MEMORY_PRESSURE_RATIO) {
      signals.push({
        label: 'memory-gc-bound',
        reason: `heap grew ${round(growthMb)}MB, peak ${round(ratio * 100)}% of limit`,
        strength: growthMb,
        evidence: { growthMb: round(growthMb), peakUsedMb: round(Math.max(...used)), limitMb: round(limitMb) }
      })
    }
  }

  // ── history-bound ─────────────────────────────────────────────────
  // The distinguishing question is whether commit cost tracks *transcript
  // size* or *flush size*. Split the flushes at the median transcript length
  // and compare mean commit cost: if the long-transcript half costs materially
  // more, the transcript itself is the cost driver, which is a different fix
  // from making the flush smaller. Needs enough samples to have two halves.
  if (deltas.length >= MIN_DELTA_SAMPLES) {
    const cut = median(deltas.map(delta => delta.historyMessages))
    const longHalf = deltas.filter(delta => delta.historyMessages >= cut)
    const shortHalf = deltas.filter(delta => delta.historyMessages < cut)
    const longCost = mean(longHalf.map(delta => delta.commitMs + delta.writeMs))
    const shortCost = mean(shortHalf.map(delta => delta.commitMs + delta.writeMs))

    if (longHalf.length && shortHalf.length && longCost >= SLOW_COMMIT_MS && longCost >= shortCost * HISTORY_COST_RATIO) {
      signals.push({
        label: 'history-bound',
        reason: `commit cost ${round(longCost)}ms at >=${round(cut)} messages vs ${round(shortCost)}ms below`,
        strength: longCost,
        evidence: {
          samples: deltas.length,
          medianHistoryMessages: round(cut),
          longHalfCostMs: round(longCost),
          shortHalfCostMs: round(shortCost)
        }
      })
    }
  }

  signals.sort((a, b) => b.strength - a.strength)

  const labels = signals.map(signal => signal.label)

  return {
    labels: labels.length ? labels : ['unclassified'],
    primary: labels.length ? labels[0] : null,
    signals,
    counts: {
      rendererLongFrames: longFrames.length,
      rendererMaxFrameMs: round(maxFrameMs),
      rendererStreamDeltas: deltas.length,
      rendererMemorySamples: memory.length,
      mainTransportErrors: transportErrors.length,
      mainUnresponsiveEdges: unresponsive.length,
      gatewayLoopDrifts: drifts.length,
      gatewayMaxDriftS: round(maxDriftS, 3),
      gatewayWsWriteSlow: wsWriteSlow.length
    }
  }
}

export {
  HISTORY_COST_RATIO,
  LONG_FRAME_MS,
  LOOP_DRIFT_S,
  MEMORY_GROWTH_MB,
  MEMORY_PRESSURE_RATIO,
  MIN_DRIFT_EVENTS,
  MIN_LONG_FRAMES,
  MIN_TRANSPORT_ERRORS,
  SEVERE_DRIFT_S,
  SEVERE_FRAME_MS,
  SLOW_COMMIT_MS
}
