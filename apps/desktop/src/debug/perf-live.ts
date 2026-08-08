// Live interaction profiler — for driving the app by hand and seeing what the
// app actually does under YOUR sessions, not a synthetic scenario's.
//
// The synthetic scenarios seed toy transcripts (short prose, no tool calls, no
// code blocks). A real session is heavier in ways that matter, so a scenario
// can report 57fps on a gesture that visibly lags in the real app. This closes
// that gap by measuring the real thing.
//
// It arms itself on any pointerdown on a resize handle and on composer typing,
// records frames + render attribution for the duration of the interaction, and
// prints a table when the interaction ends. Idle cost is zero: nothing is
// observed until an interaction starts.
//
//   window.__PERF_LIVE__.on()      start watching (also: ?perflive=1)
//   window.__PERF_LIVE__.off()
//   window.__PERF_LIVE__.last()    the most recent report as an object
//
// Dev-only; the whole debug/ graph is aliased out of production builds. The
// LoAF observer + attribution core it uses is NOT dev-only: it lives in
// `@/diagnostics` so the production capture module shares one implementation
// (KTD1).

import { createLongFrameObserver, type LongFrameSample } from '@/diagnostics/long-frames'

interface Sample {
  kind: string
  ms: number
  frames: number
  fps: number
  p95: number
  worst: number
  slow33: number
  commits: number
  top: Array<{ name: string; renders: number; wasted: number; totalMs: number }>
  longFrames: LongFrame[]
}

/** One Long Animation Frame, attributed — see `@/diagnostics/long-frames`.
 *  This is the half the render counter cannot see: a frame can cost 900ms with
 *  almost no React in it, and only LoAF says whether that was layout, a
 *  ResizeObserver callback loop, or some timer. */
type LongFrame = LongFrameSample

const RESIZE_SELECTOR =
  '[role="separator"], [data-slot="pane-resize-handle"], [class*="cursor-col-resize"], [class*="cursor-row-resize"]'

const TYPING_SELECTOR = '[contenteditable="true"], textarea, input[type="text"]'

// A gesture is "over" once this long passes with no further input events.
const IDLE_END_MS = 350
// Don't report trivial blips (a click, a single keypress).
const MIN_FRAMES = 6

let watching = false

let active: null | {
  kind: string
  startedAt: number
  frames: number[]
  last: number
  raf: number
  endTimer: ReturnType<typeof setTimeout> | null
} = null

let lastReport: null | Sample = null

// Long Animation Frames observed while a gesture is active. LoAF entries name
// the scripts inside each long frame and split out style/layout time — the
// half of a frame the render counter cannot see.
let longFrames: LongFrame[] = []

const loafObserver = createLongFrameObserver(sample => {
  if (!active) {
    return
  }

  longFrames.push(sample)
})

const pct = (sorted: number[], p: number) =>
  sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))] : 0

function finish() {
  if (!active) {
    return
  }

  const { kind, startedAt, frames, raf } = active
  active = null
  cancelAnimationFrame(raf)
  loafObserver?.stop()
  const capturedLongFrames = longFrames
  longFrames = []

  const counter = window.__RENDER_COUNTS__
  const commits = counter?.commits() ?? 0

  const top = (counter?.report(12) ?? []).map(r => ({
    name: r.name,
    renders: r.renders,
    wasted: r.wasted,
    totalMs: r.totalMs
  }))

  counter?.stop()

  if (frames.length < MIN_FRAMES) {
    return
  }

  const total = frames.reduce((a, b) => a + b, 0)
  const sorted = [...frames].sort((a, b) => a - b)

  const report: Sample = {
    commits,
    fps: Math.round((frames.length / total) * 1000 * 10) / 10,
    frames: frames.length,
    kind,
    longFrames: capturedLongFrames,
    ms: Math.round(performance.now() - startedAt),
    p95: Math.round(pct(sorted, 0.95) * 10) / 10,
    slow33: frames.filter(f => f > 33).length,
    top,
    worst: Math.round(sorted[sorted.length - 1] * 10) / 10
  }

  lastReport = report

  const headline =
    `%c${kind}%c  ${report.fps}fps  ·  ${report.frames} frames in ${report.ms}ms  ·  ` +
    `p95 ${report.p95}ms  worst ${report.worst}ms  ·  ${report.slow33} slow  ·  ${commits} commits`

  console.log(
    headline,
    `background:${report.fps < 45 ? '#c0392b' : '#27ae60'};color:#fff;padding:1px 6px;border-radius:3px`,
    'color:inherit'
  )

  if (top.length) {
    console.table(top)
  }

  // The frame-engine side: what each long frame actually spent its time on.
  for (const lf of capturedLongFrames.slice(0, 8)) {
    const scripts = lf.scripts.map(s => `${s.invoker}@${s.src} ${s.ms}ms`).join('  |  ') || '(no script ≥5ms)'
    console.log(`  ⏱ longframe ${lf.ms}ms  style+layout ${lf.styleMs}ms  block ${lf.blockingMs}ms  →  ${scripts}`)
  }
}

function begin(kind: string) {
  if (active) {
    // Same gesture continuing — just push the end deadline out.
    if (active.endTimer) {
      clearTimeout(active.endTimer)
    }

    active.endTimer = setTimeout(finish, IDLE_END_MS)

    return
  }

  window.__RENDER_COUNTS__?.start()

  // Buffered observation (a long frame already in flight when the gesture
  // starts is still attributed to it) is handled inside the shared observer;
  // on a runtime without LoAF there is no observer and attribution is empty.
  loafObserver?.start()

  const now = performance.now()

  const state = {
    endTimer: setTimeout(finish, IDLE_END_MS) as ReturnType<typeof setTimeout> | null,
    frames: [] as number[],
    kind,
    last: now,
    raf: 0,
    startedAt: now
  }

  const tick = () => {
    if (active !== state) {
      return
    }

    const t = performance.now()
    state.frames.push(t - state.last)
    state.last = t
    state.raf = requestAnimationFrame(tick)
  }

  active = state
  state.raf = requestAnimationFrame(tick)
}

function onPointerDown(event: PointerEvent) {
  const target = event.target as Element | null

  if (target?.closest?.(RESIZE_SELECTOR)) {
    begin('resize')
  }
}

function onPointerMove() {
  if (active?.kind === 'resize') {
    begin('resize')
  }
}

function onKeyDown(event: KeyboardEvent) {
  const target = event.target as Element | null

  // Ignore pure modifiers and navigation — we want real text entry.
  if (event.key.length !== 1 && event.key !== 'Backspace') {
    return
  }

  if (target?.closest?.(TYPING_SELECTOR)) {
    begin('typing')
  }
}

function on() {
  if (watching) {
    return 'already watching'
  }

  watching = true
  window.addEventListener('pointerdown', onPointerDown, true)
  window.addEventListener('pointermove', onPointerMove, true)
  window.addEventListener('keydown', onKeyDown, true)

  console.log(
    '%cperf-live%c armed — resize a pane or type in the composer; a report prints when you stop.',
    'background:#5a7db0;color:#fff;padding:1px 6px;border-radius:3px',
    'color:inherit'
  )

  return 'watching'
}

function off() {
  watching = false
  window.removeEventListener('pointerdown', onPointerDown, true)
  window.removeEventListener('pointermove', onPointerMove, true)
  window.removeEventListener('keydown', onKeyDown, true)
  finish()

  return 'stopped'
}

declare global {
  interface Window {
    __PERF_LIVE__?: {
      on: () => string
      off: () => string
      last: () => null | Sample
      watching: () => boolean
    }
  }
}

if (typeof window !== 'undefined' && !window.__PERF_LIVE__) {
  window.__PERF_LIVE__ = { last: () => lastReport, off, on, watching: () => watching }

  // Opt in for a whole session with ?perflive=1 so a reload keeps measuring.
  if (new URLSearchParams(window.location.search).get('perflive') === '1') {
    on()
  }
}

export {}
