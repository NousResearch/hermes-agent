import type { TraceSpanNode } from '@/store/trace'

// Time compression. "All" spans a whole session, mostly idle between turns. We
// build a compressed coordinate space (virtual units) where idle gaps collapse
// to a fixed width, and route positioning / zoom / ticks through it — the
// approach Sentry's compressed trace timeline uses. Real busy time maps 1:1;
// long idle gaps shrink and get a marker.

export interface TimeSeg {
  gap: boolean
  r0: number
  r1: number
  v0: number
  v1: number
}

export interface TimeMap {
  gaps: TimeSeg[]
  toReal: (v: number) => number
  toV: (t: number) => number
  totalV: number
}

export function buildTimeMap(nodes: TraceSpanNode[], fullStart: number, fullEnd: number): TimeMap {
  // Busy = actual work (LLM/TOOL). AGENT/CHAIN containers span whole turns
  // including idle, so they'd hide every gap — exclude them from gap detection.
  const busy = nodes
    .filter(n => n.kind === 'LLM' || n.kind === 'TOOL')
    .map(n => [n.start, Math.max(n.end, n.start)] as [number, number])
    .sort((a, b) => a[0] - b[0])

  const merged: [number, number][] = []

  for (const [s, e] of busy) {
    const last = merged.at(-1)

    if (last && s <= last[1]) {
      last[1] = Math.max(last[1], e)
    } else {
      merged.push([s, e])
    }
  }

  if (merged.length === 0) {
    merged.push([fullStart, fullEnd])
  }

  const activeTotal = merged.reduce((sum, [s, e]) => sum + (e - s), 0) || 1
  const compressed = Math.min(Math.max(activeTotal * 0.04, 0.5), 4)

  const segs: TimeSeg[] = []
  let v = 0

  const add = (r0: number, r1: number, gap: boolean) => {
    if (r1 <= r0) {
      return
    }

    const vlen = gap ? Math.min(r1 - r0, compressed) : r1 - r0
    segs.push({ gap, r0, r1, v0: v, v1: v + vlen })
    v += vlen
  }

  let cursor = fullStart

  if (merged[0]![0] > cursor) {
    add(cursor, merged[0]![0], true)
    cursor = merged[0]![0]
  }

  for (let i = 0; i < merged.length; i++) {
    const [s, e] = merged[i]!
    add(Math.max(s, cursor), e, false)
    cursor = Math.max(cursor, e)
    const next = merged[i + 1]

    if (next && next[0] > cursor) {
      add(cursor, next[0], true)
      cursor = next[0]
    }
  }

  if (fullEnd > cursor) {
    add(cursor, fullEnd, true)
  }

  // Degenerate input (e.g. a live trace with only AGENT spans, or a zero-width
  // range at spawn) yields no segments — guarantee at least one so toV/toReal
  // never index into an empty array.
  if (segs.length === 0) {
    const r1 = Math.max(fullEnd, fullStart + 0.001)
    segs.push({ gap: false, r0: fullStart, r1, v0: 0, v1: r1 - fullStart })
    v = r1 - fullStart
  }

  const totalV = v || 1

  const toV = (t: number) => {
    if (t <= segs[0]!.r0) {
      return 0
    }

    for (const s of segs) {
      if (t <= s.r1) {
        return s.v0 + ((t - s.r0) / (s.r1 - s.r0 || 1)) * (s.v1 - s.v0)
      }
    }

    return totalV
  }

  const toReal = (vv: number) => {
    for (const s of segs) {
      if (vv <= s.v1) {
        return s.r0 + ((vv - s.v0) / (s.v1 - s.v0 || 1)) * (s.r1 - s.r0)
      }
    }

    return fullEnd
  }

  const gaps = segs.filter(s => s.gap && s.r1 - s.r0 > s.v1 - s.v0 + 1e-6)

  return { gaps, toReal, toV, totalV }
}

/** "Nice" axis ticks (1/2/5 × 10ⁿ) covering [start, end] in virtual units. */
export function niceTicks(start: number, end: number, target = 6): number[] {
  const span = end - start

  if (span <= 0) {
    return [start]
  }

  const raw = span / target
  const mag = 10 ** Math.floor(Math.log10(raw))
  const norm = raw / mag
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag
  const ticks: number[] = []

  for (let t = Math.ceil(start / step) * step; t <= end; t += step) {
    ticks.push(t)
  }

  return ticks
}
