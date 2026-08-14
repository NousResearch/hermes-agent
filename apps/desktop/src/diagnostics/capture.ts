// Renderer-side diagnostics capture — the production half of what
// `src/debug/perf-live.ts` does in dev.
//
// The point of this module is that it ships. Hitches happen in the packaged
// app, where the dev profiler and the CDP harness do not exist, so the capture
// has to be present in a normal build and cost nothing until the user arms it.
// Two properties keep that honest:
//
//   1. Disarmed means ZERO observers, ZERO timers, ZERO allocation. Every
//      record entry point returns on a single boolean read; the LoAF observer
//      and the memory sampler are constructed on the arm edge and torn down on
//      the disarm edge, so arming needs no restart.
//   2. Sanitization happens at RECORD time, not export time (R2). Only counts,
//      sizes, durations and IDs are ever written into the ring buffer, so a
//      buffer that leaks (a crash dump, a bug report) cannot leak content.
//
// The capture id and wall-clock anchor come from the capture controller in
// Electron main (KTD3), so this stream can be aligned against the main-process
// and gateway streams at export despite each process using its own monotonic
// clock.

import { createLongFrameObserver, type LongFrameScript } from './long-frames'
import { DiagnosticsRingBuffer } from './ring-buffer'

// ~300s of events (KTD2). The count/byte caps are what actually bound memory;
// the age cap is what keeps a capture left armed all afternoon relevant.
const CAPTURE_WINDOW_MS = 300_000
const MAX_EVENTS = 6_000
const MAX_BYTES = 2_000_000
const MEMORY_SAMPLE_INTERVAL_MS = 5_000

const BYTES_PER_MB = 1024 * 1024

interface DiagnosticsEventBase {
  /** `performance.now()` at record time — monotonic, aligned at export. */
  t: number
}

export interface LongFrameEvent extends DiagnosticsEventBase {
  type: 'long_frame'
  ms: number
  styleMs: number
  blockingMs: number
  scripts: LongFrameScript[]
}

export interface StreamDeltaAppliedEvent extends DiagnosticsEventBase {
  type: 'stream_delta_applied'
  /** Sessions whose queued deltas this flush applied. */
  sessions: number
  /** Characters applied — a size, never the text. */
  queuedChars: number
  /** Transcript length those sessions carry after the apply. */
  historyMessages: number
  /** Synchronous store-write cost of the flush. */
  writeMs: number
  /** In-frame cost of the deferred view sync + React commit, or 0 while the
   *  measurement frame is still pending (or never fires — hidden renderer). */
  commitMs: number
  /** Flush start → the measurement frame's start: how long the commit waited
   *  for a frame. Stays 0 when no frame arrives. */
  rafGapMs: number
  /** Which flush drained the queue: the batched timer flush, or an eager
   *  drain forced by an ordering-sensitive event (tool rows, interim seals,
   *  turn completion). Eager drains skip the commit-measurement frame, so
   *  their `commitMs`/`rafGapMs` stay 0 by construction. */
  path: 'eager' | 'timer'
  /** Sessions with a turn in flight at record time — concurrent threads, not
   *  just the ones with text queued in THIS flush (`sessions` above). This is
   *  the multi-chat attribution signal: a second thread doing tool-call work
   *  contributes render load without ever entering the delta queue. */
  busySessions: number
}

export interface GatewayEventAppliedEvent extends DiagnosticsEventBase {
  type: 'gateway_event_applied'
  /** The gateway event's type tag (`tool.complete`, `subagent.progress`, …) —
   *  an identifier, never payload content. */
  eventType: string
  /** Synchronous main-thread cost of dispatching the event: store writes,
   *  eager flushes, tool-row upserts — everything the handler did in place. */
  durationMs: number
  /** Sessions with a turn in flight at record time (see
   *  StreamDeltaAppliedEvent.busySessions). */
  busySessions: number
}

export interface MemorySampleEvent extends DiagnosticsEventBase {
  type: 'memory_sample'
  usedMb: number
  totalMb: number
  limitMb: number
}

export type DiagnosticsEvent = GatewayEventAppliedEvent | LongFrameEvent | MemorySampleEvent | StreamDeltaAppliedEvent

export interface DiagnosticsCaptureSnapshot {
  captureId: string
  /** Date.now() at arm time, supplied by main — the cross-process anchor. */
  wallClockAnchorMs: number
  /** performance.now() at arm time, so event `t` values can be shifted onto
   *  the wall clock without trusting this renderer's Date.now(). */
  monotonicAnchorMs: number
  events: DiagnosticsEvent[]
  droppedEvents: number
}

interface ArmedCapture {
  captureId: string
  wallClockAnchorMs: number
  monotonicAnchorMs: number
  buffer: DiagnosticsRingBuffer<DiagnosticsEvent>
  longFrames: null | { stop: () => void }
  memoryTimer: null | ReturnType<typeof setInterval>
}

// The single hot-path read. Module-level so `isDiagnosticsArmed()` compiles to
// a null check, not a lookup through a store or a React context.
let armed: ArmedCapture | null = null

/** Cheap guard for instrumentation on hot paths: skip all sampling work when
 *  no capture is running. */
export const isDiagnosticsArmed = () => armed !== null

/** The running capture's id, or null. */
export const activeCaptureId = () => armed?.captureId ?? null

interface MemoryPerformance extends Performance {
  memory?: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number }
}

function sampleMemory(): void {
  const memory = (performance as MemoryPerformance).memory

  if (!armed || !memory) {
    return
  }

  armed.buffer.push({
    limitMb: Math.round(memory.jsHeapSizeLimit / BYTES_PER_MB),
    t: performance.now(),
    totalMb: Math.round(memory.totalJSHeapSize / BYTES_PER_MB),
    type: 'memory_sample',
    usedMb: Math.round(memory.usedJSHeapSize / BYTES_PER_MB)
  })
}

/** Start a capture. Re-arming with the same id is a no-op; re-arming with a new
 *  id restarts cleanly, so main can drive this without a renderer reload. */
export function armDiagnostics(captureId: string, wallClockAnchorMs: number): void {
  if (armed?.captureId === captureId) {
    return
  }

  disarmDiagnostics()

  const capture: ArmedCapture = {
    buffer: new DiagnosticsRingBuffer<DiagnosticsEvent>({
      maxAgeMs: CAPTURE_WINDOW_MS,
      maxBytes: MAX_BYTES,
      maxEvents: MAX_EVENTS
    }),
    captureId,
    longFrames: null,
    memoryTimer: null,
    monotonicAnchorMs: performance.now(),
    wallClockAnchorMs
  }

  armed = capture

  const longFrames = createLongFrameObserver(sample => {
    if (armed !== capture) {
      return
    }

    capture.buffer.push({ ...sample, t: performance.now(), type: 'long_frame' })
  })

  if (longFrames) {
    longFrames.start()
    capture.longFrames = longFrames
  }

  // A periodic heap sample is what separates "memory/GC-bound" from the other
  // classifications (R3); Chromium only exposes it in the renderer.
  if ((performance as MemoryPerformance).memory) {
    capture.memoryTimer = setInterval(sampleMemory, MEMORY_SAMPLE_INTERVAL_MS)
    sampleMemory()
  }
}

/** Stop the capture and release every observer/timer it registered. The buffer
 *  goes with it — read it with `readDiagnosticsCapture()` first. */
export function disarmDiagnostics(): void {
  if (!armed) {
    return
  }

  armed.longFrames?.stop()

  if (armed.memoryTimer !== null) {
    clearInterval(armed.memoryTimer)
  }

  armed.buffer.clear()
  armed = null
}

/** Snapshot the running capture for export (U4). Null when disarmed. */
export function readDiagnosticsCapture(): DiagnosticsCaptureSnapshot | null {
  if (!armed) {
    return null
  }

  return {
    captureId: armed.captureId,
    droppedEvents: armed.buffer.droppedCount,
    events: armed.buffer.entries(),
    monotonicAnchorMs: armed.monotonicAnchorMs,
    wallClockAnchorMs: armed.wallClockAnchorMs
  }
}

export interface StreamDeltaSample {
  sessions: number
  queuedChars: number
  historyMessages: number
  writeMs: number
  path: 'eager' | 'timer'
  busySessions: number
}

/** Record one applied stream-delta flush. Returns the stored event so the
 *  caller's existing commit-cost rAF can fill `commitMs`/`rafGapMs` in place
 *  when the frame lands — one event per flush, and no second measurement pass.
 *  Returns null when disarmed, which is the caller's cue to do nothing. */
export function recordStreamDeltaApplied(sample: StreamDeltaSample): StreamDeltaAppliedEvent | null {
  if (!armed) {
    return null
  }

  return armed.buffer.push({
    busySessions: Math.round(sample.busySessions),
    commitMs: 0,
    historyMessages: Math.round(sample.historyMessages),
    path: sample.path,
    queuedChars: Math.round(sample.queuedChars),
    rafGapMs: 0,
    sessions: Math.round(sample.sessions),
    t: performance.now(),
    type: 'stream_delta_applied',
    writeMs: Math.round(sample.writeMs * 100) / 100
  }) as StreamDeltaAppliedEvent
}

export interface GatewayEventSample {
  eventType: string
  durationMs: number
  busySessions: number
}

/** Record one costly gateway-event dispatch. The caller thresholds on
 *  duration BEFORE calling (a streaming turn emits dozens of sub-millisecond
 *  events per second; recording them all would be ring churn for nothing), so
 *  reaching this function already means "worth a row". */
export function recordGatewayEventApplied(sample: GatewayEventSample): void {
  if (!armed) {
    return
  }

  armed.buffer.push({
    busySessions: Math.round(sample.busySessions),
    durationMs: Math.round(sample.durationMs * 100) / 100,
    eventType: sample.eventType,
    t: performance.now(),
    type: 'gateway_event_applied'
  })
}
