import type { StarmapNode } from '@/types/hermes'

import { AGE_GRADIENT, FIT_PADDING, RING_INNER, RING_OUTER, TILT, ZOOM_MAX, ZOOM_MIN } from './constants'
import type { Ring, Shape, Viewport } from './types'

export function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

// FNV-1a — stable per-id seed for layout angle / starfield.
export function hash(input: string): number {
  let h = 2166136261

  for (let i = 0; i < input.length; i += 1) {
    h ^= input.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }

  return h >>> 0
}

export function nodeRadius(n: StarmapNode): number {
  if (n.kind === 'memory') {
    return 4.4
  }

  const base = n.state === 'archived' || n.state === 'stale' ? 2.4 : 3

  return base + Math.sqrt(Math.max(0, n.useCount)) * 0.55 + (n.pinned ? 0.8 : 0)
}

// Smoothstep recency → ink alpha along the age gradient.
export function recencyInk(rec: number): number {
  const reach = Math.max(0.01, AGE_GRADIENT.reach)
  const mid = clamp(AGE_GRADIENT.mid, 0.01, 0.99)
  const t = clamp(rec / reach, 0, 1)

  if (t <= mid) {
    const p = t / mid

    return AGE_GRADIENT.oldInk + (AGE_GRADIENT.midInk - AGE_GRADIENT.oldInk) * (p * p * (3 - 2 * p))
  }

  const p = (t - mid) / (1 - mid)

  return AGE_GRADIENT.midInk + (AGE_GRADIENT.newInk - AGE_GRADIENT.midInk) * (p * p * (3 - 2 * p))
}

// Trace a centred geometric shape of radius r into the current path.
export function shapePath(ctx: CanvasRenderingContext2D, shape: Shape, x: number, y: number, r: number): void {
  ctx.beginPath()

  if (shape === 'square') {
    ctx.rect(x - r, y - r, r * 2, r * 2)

    return
  }

  if (shape === 'circle') {
    ctx.arc(x, y, r, 0, Math.PI * 2)

    return
  }

  const pts = shape === 'diamond' ? 4 : shape === 'triangle' ? 3 : 6
  // Diamond/triangle point up; hexagon is flat-topped.
  const rot = shape === 'hexagon' ? Math.PI / 6 : -Math.PI / 2

  for (let i = 0; i < pts; i += 1) {
    const a = rot + (i / pts) * Math.PI * 2
    const px = x + Math.cos(a) * r
    const py = y + Math.sin(a) * r

    if (i === 0) {
      ctx.moveTo(px, py)
    } else {
      ctx.lineTo(px, py)
    }
  }

  ctx.closePath()
}

// Center the tilted disk in the viewport at a fit zoom. `outer` is the radius to
// fit (defaults to the full disk); the scrubber passes the revealed extent so the
// camera tightens at the core and zooms out as the rings grow.
export function fitViewport(w: number, h: number, outer: number = RING_OUTER): Viewport {
  if (w <= 0 || h <= 0) {
    return { k: 1, x: w / 2, y: h / 2 }
  }

  const spanX = (outer + 30) * 2
  const spanY = spanX * TILT
  const k = clamp(Math.min((w - FIT_PADDING * 2) / spanX, (h - FIT_PADDING * 2) / spanY, 2.2), ZOOM_MIN, ZOOM_MAX)

  // Bias the center down a touch — the timeline along the top adds visual weight
  // up there, so true-center reads as sitting high.
  return { k, x: w / 2, y: h / 2 + h * 0.05 }
}

// Target radius for a node at recency `rec` (oldest at the core), scaled to a
// disk of the given outer radius.
export function radiusForRecency(rec: number, outer: number = RING_OUTER): number {
  return RING_INNER + rec * (outer - RING_INNER)
}

// Screen-space scale at the graph's fully-rested fit. Nodes size against THIS,
// not the live (playback) camera — so a spore-zoom moves WHERE they sit, not how
// big they read (billboarded), while a full-map view keeps its honest density.
export const fitScale = (w: number, h: number, rings: Ring[]): number => fitViewport(w, h, rings.at(-1)?.r ?? RING_OUTER).k

// Squared distance from point (px,py) to segment a→b — for cheap link hit-tests.
export function distToSegmentSq(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax
  const dy = by - ay
  const len = dx * dx + dy * dy
  const t = len ? clamp(((px - ax) * dx + (py - ay) * dy) / len, 0, 1) : 0
  const cx = ax + dx * t
  const cy = ay + dy * t

  return (px - cx) ** 2 + (py - cy) ** 2
}
