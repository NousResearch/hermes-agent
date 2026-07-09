import { type Simulation } from 'd3-force'
import { atom, type WritableAtom } from 'nanostores'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { useThemeEpoch } from '@/hooks/use-theme-epoch'
import { createDoubleTapDetector, isSmartZoomWheel } from '@/lib/trackpad-gestures'
import type { StarmapGraph } from '@/types/hermes'

import { computePalette, memoryInkFor, resolveRgb, rgba } from './color'
import { RING_OUTER, TILT, ZOOM_MAX, ZOOM_MIN } from './constants'
import { clamp, distToSegmentSq, fitScale, fitViewport, nodeRadius } from './geometry'
import { NodeContextMenu, type NodeMenuTarget } from './node-context-menu'
import { drawScene, drawScramble } from './render'
import { decodeShareCode, encodeShareCode, ShareCodeError } from './share-code'
import { ShareControls } from './share-controls'
import { buildSimulation } from './simulation'
import { formatDate } from './text'
import { buildTimeAxis, dateAtReveal, type TimeAxis } from './time-axis'
import { Timeline } from './timeline'
import type { FadeBuckets, MemoryCard, Palette, Ring, RingLabelRect, SimLink, SimNode, Viewport } from './types'

// How long a full play-through sweep takes (ms), reveal 0 → 1. Longer = the
// build-up breathes; the eased middle no longer rushes past in a blink.
const SWEEP_MS = 15000

// How far to relax the ease toward a flat linear march. The bare smoothstep
// spikes to 1.5× linear speed mid-sweep, which reads as a "snap" through the
// middle; blending it back toward linear flattens that peak (≈1.3× at GENTLE
// = 0.45) so playback glides instead of lurching, while still keeping a soft
// ease-in / ease-out at the very start and end.
const GENTLE = 0.45

// Cinematic timing: cubic smoothstep (gentle ease-in / ease-out) relaxed toward
// linear by GENTLE, so the middle never rushes. Monotonic on [0,1], so the
// numeric inverse below stays valid.
function cineEase(t: number): number {
  const u = t < 0 ? 0 : t > 1 ? 1 : t
  const smooth = u * u * (3 - 2 * u)

  return GENTLE * u + (1 - GENTLE) * smooth
}

// Numeric inverse (monotonic) so a resume maps the current reveal back to clock
// progress without a closed-form solution.
function invCineEase(y: number): number {
  let lo = 0
  let hi = 1

  for (let i = 0; i < 24; i += 1) {
    const mid = (lo + hi) / 2

    if (cineEase(mid) < y) {
      lo = mid
    } else {
      hi = mid
    }
  }

  return (lo + hi) / 2
}

function revealText(axis: TimeAxis, reveal: number): string {
  const date = dateAtReveal(axis, reveal)

  return date !== null ? formatDate(date) : `${Math.round(reveal * axis.size)} / ${axis.size}`
}

function RevealLabel({ axis, revealStore }: { axis: TimeAxis; revealStore: WritableAtom<number> }) {
  const labelRef = useRef<HTMLSpanElement | null>(null)

  const sync = useCallback(
    (reveal: number) => {
      const el = labelRef.current

      if (el) {
        el.textContent = revealText(axis, reveal)
      }
    },
    [axis]
  )

  useEffect(() => revealStore.subscribe(sync), [revealStore, sync])

  useEffect(() => {
    sync(revealStore.get())
  }, [revealStore, sync])

  return (
    <span className="tabular-nums text-foreground/75" ref={labelRef}>
      {revealText(axis, revealStore.get())}
    </span>
  )
}

// A tilted, top-down star map of what IX Agency has learned. Time is RADIAL: oldest
// at the core, newest on the outer rings. This component owns the refs, effects
// and pointer wiring; layout lives in simulation.ts and painting in render.ts.
export function StarMap({
  graph,
  imported = false,
  onImport,
  onResetMap
}: {
  graph: StarmapGraph
  imported?: boolean
  onImport?: (graph: StarmapGraph) => void
  onResetMap?: () => void
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const wrapRef = useRef<HTMLDivElement | null>(null)

  const simRef = useRef<null | Simulation<SimNode, SimLink>>(null)
  const nodesRef = useRef<SimNode[]>([])
  const linksRef = useRef<SimLink[]>([])
  const byIdRef = useRef(new Map<string, SimNode>())
  const adjacencyRef = useRef(new Map<string, Set<string>>())
  const memByIdRef = useRef(new Map<string, MemoryCard>())
  const ringsRef = useRef<Ring[]>([])
  const ringLabelRectsRef = useRef<RingLabelRect[]>([])

  const fadeRef = useRef<FadeBuckets>({
    appear: new Map(),
    labels: new Map(),
    links: new Map(),
    nodes: new Map(),
    rings: new Map()
  })

  const doubleTapRef = useRef(createDoubleTapDetector())
  const paletteRef = useRef<null | Palette>(null)
  const themeDirtyRef = useRef(true)
  const invalidateRef = useRef<() => void>(() => {})
  const viewportRef = useRef<Viewport>({ k: 1, x: 0, y: 0 })
  const hoverRef = useRef<null | string>(null)
  const hoveredLinkRef = useRef<null | string>(null)
  const hoveredRingRef = useRef<null | number>(null)
  const selectedRingRef = useRef<null | number>(null)
  const selectedIdRef = useRef<null | string>(null)
  const sizeRef = useRef({ h: 0, w: 0 })
  const dprRef = useRef(1)
  const dirtyRef = useRef(true)
  // Scrub = direct manipulation (snap the fades to the pointer); Play = the
  // cinematic birth/fade easing. One frame's worth of state, never re-rendered.
  const snapMotionRef = useRef(false)

  const dragRef = useRef<{
    id: null | string
    mode: 'none' | 'pan'
    moved: boolean
    ring: null | number
    sx: number
    sy: number
    vp: Viewport
  }>({ id: null, mode: 'none', moved: false, ring: null, sx: 0, sy: 0, vp: { k: 1, x: 0, y: 0 } })

  const [selectedId, setSelectedId] = useState<null | string>(null)
  const [menuTarget, setMenuTarget] = useState<NodeMenuTarget | null>(null)
  const [size, setSize] = useState({ h: 0, w: 0 })
  // Increments on every theme repaint (shared hook) so the legend swatch and the
  // canvas palette re-resolve against the freshly-painted CSS custom properties.
  const themeEpoch = useThemeEpoch()
  // Memory's swatch color — the same complementary-of-primary the canvas uses,
  // so the legend matches the rendered diamonds exactly.
  const [memoryColor, setMemoryColor] = useState('var(--theme-secondary)')

  // Time scrubber: reveal 1 = the whole map (idle default); lower values hide
  // not-yet-reached nodes so playing/scrubbing "builds it up". revealRef feeds
  // the canvas loop and revealStore feeds the timeline + legend label — so a
  // play-through / scrub never re-renders StarMap (the perf win). `playing`
  // stays React state since it flips rarely and drives the play effect + button.
  const revealStore = useMemo(() => atom(1), [])
  const [playing, setPlaying] = useState(false)
  // Reveal positions where each dated ring spawns (its inner neighbor's ratio —
  // ringSeen reveals one band ahead), surfaced as markers on the timeline.
  const [ringStops, setRingStops] = useState<number[]>([])
  const revealRef = useRef(1)
  // Spore-style zoom: the camera fits the *leading ring's* radius, a step
  // function of reveal. It holds steady while a band fills, then eases out to the
  // next shell when a new ring is reached — growth in discrete jumps, not a
  // constant creep. This ref is the camera's current (eased) fit radius.
  const camRadiusRef = useRef(RING_OUTER)
  const timeAxis = useMemo(() => buildTimeAxis(graph, 72), [graph])

  // The current map as a WoW-style share code, recomputed only when the graph
  // changes (encode walks every node/edge/card, so don't redo it per render).
  const shareCode = useMemo(() => encodeShareCode(graph), [graph])

  // Decode a pasted code and hand the resulting graph up to the StarmapView,
  // which swaps it in for the live profile scan. Returns an error string for the
  // Timeline to surface inline, or null on success.
  const importCode = useCallback(
    (code: string): null | string => {
      try {
        const next = decodeShareCode(code)
        onImport?.(next)

        return null
      } catch (err) {
        return err instanceof ShareCodeError ? err.message : 'Could not read that map code.'
      }
    },
    [onImport]
  )

  // Mark the canvas dirty and wake the (otherwise-idle) render loop.
  const invalidate = useCallback(() => invalidateRef.current(), [])

  // Single writer for the scrubber position: feeds the canvas (ref), the
  // timeline + legend label (store subscribers), and wakes the paint loop —
  // no React re-render, so playback/scrubbing stays off the render path.
  const setRevealValue = useCallback(
    (value: number) => {
      const next = clamp(value, 0, 1)
      revealRef.current = next
      revealStore.set(next)
      invalidate()
    },
    [invalidate, revealStore]
  )

  // Drop every in-flight ease so the next frame snaps to its targets.
  const resetFades = useCallback(() => {
    for (const bucket of Object.values(fadeRef.current)) {
      bucket.clear()
    }
  }, [])

  const memById = useMemo(() => {
    const m = new Map<string, MemoryCard>()
    graph.memory.forEach((card, i) => m.set(`memory:${card.source}:${i}`, card))

    return m
  }, [graph.memory])

  const adjacency = useMemo(() => {
    const m = new Map<string, Set<string>>()

    for (const n of graph.nodes) {
      m.set(n.id, new Set())
    }

    for (const e of graph.edges) {
      m.get(e.source)?.add(e.target)
      m.get(e.target)?.add(e.source)
    }

    return m
  }, [graph.edges, graph.nodes])

  // Track the wrapper size.
  useEffect(() => {
    const el = wrapRef.current

    if (!el) {
      return
    }

    const sync = () => setSize({ h: el.clientHeight, w: el.clientWidth })
    const ro = new ResizeObserver(sync)
    ro.observe(el)
    sync()

    return () => ro.disconnect()
  }, [])

  // (Re)build the radial simulation whenever the graph or size changes.
  useEffect(() => {
    sizeRef.current = size

    if (size.w === 0 || size.h === 0) {
      return
    }

    const { byId, links, nodes, rings, sim } = buildSimulation(graph, invalidate)
    simRef.current = sim
    nodesRef.current = nodes
    linksRef.current = links
    byIdRef.current = byId
    ringsRef.current = rings
    // Markers fire when a ring spawns: ringSeen(i) flips at rings[i-1].ratio.
    setRingStops(rings.map((rg, i) => (rg.label != null ? (rings[i - 1]?.ratio ?? 0) : -1)).filter(v => v >= 0))
    resetFades()
    // Fit the actual disk (outermost ring), so a 3-ring map frames like a 12-ring
    // one — count changes the disk size, not the framing.
    viewportRef.current = fitViewport(size.w, size.h, rings[rings.length - 1]?.r ?? RING_OUTER)
    invalidate()

    if (selectedIdRef.current && !byId.has(selectedIdRef.current)) {
      selectedIdRef.current = null
      setSelectedId(null)
    }

    return () => {
      sim.stop()

      if (simRef.current === sim) {
        simRef.current = null
      }
    }
  }, [graph, invalidate, resetFades, size])

  useEffect(() => {
    adjacencyRef.current = adjacency
    memByIdRef.current = memById
    invalidate()
  }, [adjacency, invalidate, memById])

  // The empty-core ASCII scramble uses the bundled JetBrains Mono face. Canvas
  // text doesn't reflow when a webfont loads, so repaint once it's ready.
  useEffect(() => {
    document.fonts?.load('1em "JetBrains Mono"').then(invalidate, () => {})
  }, [invalidate])

  useEffect(() => {
    selectedIdRef.current = selectedId
    invalidate()
  }, [invalidate, selectedId])

  // A fresh graph resets the scrubber to "fully built" (the idle default).
  useEffect(() => {
    camRadiusRef.current = RING_OUTER
    snapMotionRef.current = false
    setRevealValue(1)
    setPlaying(false)
  }, [graph, setRevealValue])

  // The stepped fit radius for a reveal: fit ONE ring BEYOND the leading shell
  // (the first not-yet-passed ring) so the ring currently igniting its nodes
  // sits comfortably inside the frame with a band of headroom, instead of jammed
  // against the edge. Steps only when reveal crosses a boundary — the Spore step.
  const targetRadius = useCallback((rev: number): number => {
    const rings = ringsRef.current

    if (!rings.length) {
      return RING_OUTER
    }

    const lead = rings.findIndex(rg => rg.ratio > rev + 1e-3)
    const i = lead === -1 ? rings.length - 1 : lead
    const band = (rings[1]?.r ?? RING_OUTER) - (rings[0]?.r ?? 0)

    // Small headroom (a third of a band) so the igniting ring isn't jammed at the
    // frame edge during playback, without zooming the resting view out.
    return rings[i]!.r + band * 0.35
  }, [])

  const applyFit = useCallback((radius: number) => {
    const { h, w } = sizeRef.current

    if (w > 0 && h > 0) {
      viewportRef.current = fitViewport(w, h, radius)
    }
  }, [])

  // Snap the camera to a reveal's stepped target (scrubbing / reset — no glide).
  const fitForReveal = useCallback(
    (rev: number) => {
      camRadiusRef.current = targetRadius(rev)
      applyFit(camRadiusRef.current)
    },
    [applyFit, targetRadius]
  )

  // Playback: sweep reveal 0 → 1 over SWEEP_MS, then stop (play once).
  useEffect(() => {
    if (!playing) {
      return
    }

    let raf = 0
    let start = 0

    const step = (now: number) => {
      if (!start) {
        // Anchor (in clock-space) so a resume continues from the current reveal.
        start = now - invCineEase(revealRef.current) * SWEEP_MS
      }

      const progress = Math.min(1, (now - start) / SWEEP_MS)
      const next = cineEase(progress)

      // Ease the camera toward the leading ring's radius (a step target): it
      // holds while a band fills, then pushes out when the next shell is reached.
      const target = targetRadius(next)
      camRadiusRef.current += (target - camRadiusRef.current) * 0.1
      applyFit(camRadiusRef.current)
      setRevealValue(next)

      // End once the reveal is complete AND the camera has settled on the final
      // shell, so the last push-out finishes instead of cutting off.
      if (progress >= 1 && Math.abs(target - camRadiusRef.current) < 0.5) {
        camRadiusRef.current = target
        applyFit(target)
        setPlaying(false)

        return
      }

      raf = requestAnimationFrame(step)
    }

    raf = requestAnimationFrame(step)

    return () => cancelAnimationFrame(raf)
  }, [applyFit, playing, setRevealValue, targetRadius])

  const onTogglePlay = useCallback(() => {
    if (playing) {
      setPlaying(false)

      return
    }

    // Leaving scrub: play eases (cinematic) rather than holding the snapped view.
    snapMotionRef.current = false

    // Replay from the start when parked at the end. Snap straight to the empty
    // state (no fade-out) before playing in.
    if (revealRef.current >= 1) {
      resetFades()
      fitForReveal(0)
      setRevealValue(0)
    }

    setPlaying(true)
  }, [fitForReveal, playing, resetFades, setRevealValue])

  const onScrub = useCallback(
    (value: number) => {
      const next = clamp(value, 0, 1)
      setPlaying(false)
      // Scrub is direct manipulation: snap fades + camera to the pointer so a
      // fast drag jumps there instead of replaying the birth-in from a stale spot.
      snapMotionRef.current = true
      fitForReveal(next)
      setRevealValue(next)
    },
    [fitForReveal, setRevealValue]
  )

  // Spacebar toggles playback (unless typing, or the play button itself is
  // focused — that already handles Space natively, so skip to avoid a double).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space' && e.key !== ' ') {
        return
      }

      const el = document.activeElement
      const tag = el?.tagName

      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'BUTTON' || (el as HTMLElement | null)?.isContentEditable) {
        return
      }

      e.preventDefault()
      onTogglePlay()
    }

    window.addEventListener('keydown', onKey)

    return () => window.removeEventListener('keydown', onKey)
  }, [onTogglePlay])

  // Recompute the legend's memory swatch from the live --theme-primary (matches
  // the canvas), re-running on theme change and once the canvas is mounted.
  useEffect(() => {
    const el = canvasRef.current ?? wrapRef.current

    if (!el) {
      return
    }

    const style = getComputedStyle(el)
    const val = style.getPropertyValue('--theme-primary').trim()

    if (val) {
      const bgVal =
        style.getPropertyValue('--background').trim() || style.getPropertyValue('--dt-background').trim() || '#000'

      setMemoryColor(rgba(memoryInkFor(resolveRgb(val), resolveRgb(bgVal)), 0.9))
    }
  }, [size, themeEpoch])

  // Repaint + repalette when the theme/mode repaints (the shared observer fires
  // after applyTheme rewrites the class + inline vars on <html>).
  useEffect(() => {
    themeDirtyRef.current = true
    invalidate()
  }, [invalidate, themeEpoch])

  // Render loop. The core scramble animates continuously, so the loop runs while
  // the window is focused — but each frame is cheap (live scramble + a blit of the
  // cached static layer). The expensive scene only re-renders when invalidate()
  // marks it dirty. Capped to ~30fps; interaction (force) bypasses the cap.
  useEffect(() => {
    let raf = 0
    const ANIM_MS = 1000 / 30
    let lastAnimTs = 0
    let force = true

    // The scramble keeps the loop perpetually "animating", so a fully-built,
    // untouched map still repaints 30×/s for as long as the panel is open. That's
    // wasted CPU/GPU (WindowServer compositing) when the window isn't even the one
    // you're looking at. Freeze the loop while the window is hidden or unfocused;
    // a frozen core next to other work is fine, and it resumes instantly on focus.
    const isPaused = () =>
      (typeof document !== 'undefined' && document.hidden) ||
      (typeof document.hasFocus === 'function' && !document.hasFocus())

    let paused = isPaused()

    const schedule = () => {
      if (!paused && !raf) {
        raf = requestAnimationFrame(frame)
      }
    }

    // The static scene (rings, bands, links, nodes, labels) is cached in an
    // offscreen layer and only re-rendered when something actually changes —
    // dirtyRef flags that. The animated core scramble is the ONLY per-frame work:
    // each frame we just clear, draw the live scramble, and blit the cached layer
    // on top. So an idle map costs a scramble + one drawImage, not a full redraw.
    let staticCanvas: HTMLCanvasElement | null = null

    const paint = () => {
      const canvas = canvasRef.current
      const ctx = canvas?.getContext('2d')

      if (!canvas || !ctx) {
        return
      }

      if (!staticCanvas) {
        staticCanvas = document.createElement('canvas')
      }

      // Keep the offscreen layer matched to the backing store; a resize wipes it,
      // so force a static rebuild.
      if (staticCanvas.width !== canvas.width || staticCanvas.height !== canvas.height) {
        staticCanvas.width = canvas.width
        staticCanvas.height = canvas.height
        dirtyRef.current = true
      }

      const offCtx = staticCanvas.getContext('2d')

      if (!offCtx) {
        return
      }

      if (themeDirtyRef.current || !paletteRef.current) {
        paletteRef.current = computePalette(canvas)
        themeDirtyRef.current = false
        dirtyRef.current = true
      }

      const palette = paletteRef.current

      if (!palette) {
        return
      }

      // Rebuild the cached static layer only when the scene changed; keep
      // rebuilding while fades are mid-ease (drawScene returns `animating`).
      if (dirtyRef.current) {
        const { animating, ringLabelRects } = drawScene({
          adjacency: adjacencyRef.current,
          byId: byIdRef.current,
          ctx: offCtx,
          dpr: dprRef.current,
          fades: fadeRef.current,
          focusId: selectedIdRef.current ?? hoverRef.current,
          hoverId: hoverRef.current,
          hoverLink: hoveredLinkRef.current,
          hoverRing: hoveredRingRef.current,
          links: linksRef.current,
          memById: memByIdRef.current,
          nodes: nodesRef.current,
          palette,
          reveal: revealRef.current,
          rings: ringsRef.current,
          selectedRing: selectedRingRef.current,
          size: sizeRef.current,
          snapMotion: snapMotionRef.current,
          vp: viewportRef.current
        })

        // One-shot: a scrub snaps this frame; hover/focus afterward eases as usual
        // (buckets are already at target, so the next eased frames don't move).
        snapMotionRef.current = false
        ringLabelRectsRef.current = ringLabelRects
        dirtyRef.current = animating
      }

      // Composite order flips on focus/hover. Idle: scene first, sphere on top —
      // its backdrop wash dims the busy centre. Focused/hovered: sphere first,
      // scene on top — so the active node's tooltip + lit lines lift ABOVE the
      // sphere instead of being covered by it.
      const focused = selectedIdRef.current ?? hoverRef.current
      ctx.setTransform(1, 0, 0, 1, 0, 0)
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      if (focused) {
        drawScramble({ ctx, dpr: dprRef.current, palette, rings: ringsRef.current, vp: viewportRef.current })
        ctx.setTransform(1, 0, 0, 1, 0, 0)
        ctx.drawImage(staticCanvas, 0, 0)
      } else {
        ctx.drawImage(staticCanvas, 0, 0)
        drawScramble({ ctx, dpr: dprRef.current, palette, rings: ringsRef.current, vp: viewportRef.current })
      }
    }

    const frame = (ts: number) => {
      raf = 0

      // The scramble animates every frame; throttle to ANIM_MS unless an
      // interaction (force) needs an immediate repaint.
      if (!force && ts - lastAnimTs < ANIM_MS) {
        schedule()

        return
      }

      force = false
      lastAnimTs = ts
      paint()
      schedule()
    }

    invalidateRef.current = () => {
      dirtyRef.current = true
      force = true
      schedule()
    }

    // Suspend the loop when the window drops out of view/focus; wake + force a
    // fresh frame the moment it returns so the resume is seamless.
    const onActivity = () => {
      const next = isPaused()

      if (next === paused) {
        return
      }

      paused = next

      if (paused) {
        if (raf) {
          cancelAnimationFrame(raf)
          raf = 0
        }
      } else {
        dirtyRef.current = true
        force = true
        schedule()
      }
    }

    document.addEventListener('visibilitychange', onActivity)
    window.addEventListener('blur', onActivity)
    window.addEventListener('focus', onActivity)

    schedule()

    return () => {
      cancelAnimationFrame(raf)
      document.removeEventListener('visibilitychange', onActivity)
      window.removeEventListener('blur', onActivity)
      window.removeEventListener('focus', onActivity)

      invalidateRef.current = () => {}
    }
  }, [])

  // Size the backing canvas (DPR-aware).
  useEffect(() => {
    sizeRef.current = size
    dprRef.current = Math.min(2, window.devicePixelRatio || 1)
    const canvas = canvasRef.current

    if (canvas && size.w > 0 && size.h > 0) {
      canvas.width = Math.round(size.w * dprRef.current)
      canvas.height = Math.round(size.h * dprRef.current)
      canvas.style.width = `${size.w}px`
      canvas.style.height = `${size.h}px`
    }

    invalidate()
  }, [invalidate, size])

  // ── Pointer interactions (invert the tilted projection for hit-testing) ─────
  const pickNode = (cssX: number, cssY: number): null | SimNode => {
    const vp = viewportRef.current
    // Hit radius mirrors the billboarded draw: rested fit scale, screen space.
    const nodeK = fitScale(sizeRef.current.w, sizeRef.current.h, ringsRef.current)
    let best: null | SimNode = null
    let bestD = Infinity

    for (const n of nodesRef.current) {
      const r = nodeRadius(n) * nodeK + 6
      const sx = n.x * vp.k + vp.x
      const sy = n.y * vp.k * TILT + vp.y
      const d = (sx - cssX) ** 2 + (sy - cssY) ** 2

      if (d < r * r && d < bestD) {
        bestD = d
        best = n
      }
    }

    return best
  }

  // Nearest link within ~5px of the cursor (screen space), or null.
  const pickLink = (cssX: number, cssY: number): null | string => {
    const vp = viewportRef.current
    let best: null | string = null
    let bestD = 25

    for (const link of linksRef.current) {
      const s = typeof link.source === 'object' ? link.source : byIdRef.current.get(String(link.source))
      const t = typeof link.target === 'object' ? link.target : byIdRef.current.get(String(link.target))

      if (!s || !t) {
        continue
      }

      const d = distToSegmentSq(
        cssX,
        cssY,
        s.x * vp.k + vp.x,
        s.y * vp.k * TILT + vp.y,
        t.x * vp.k + vp.x,
        t.y * vp.k * TILT + vp.y
      )

      if (d < bestD) {
        bestD = d
        best = `${s.id}->${t.id}`
      }
    }

    return best
  }

  const pickRingLabel = (cssX: number, cssY: number): null | number => {
    for (const r of ringLabelRectsRef.current) {
      if (cssX >= r.x && cssX <= r.x + r.w && cssY >= r.y && cssY <= r.y + r.h) {
        return r.i
      }
    }

    return null
  }

  const localXY = (e: React.MouseEvent): { x: number; y: number } => {
    const rect = canvasRef.current?.getBoundingClientRect()

    return { x: e.clientX - (rect?.left ?? 0), y: e.clientY - (rect?.top ?? 0) }
  }

  const resetView = () => {
    setPlaying(false)
    viewportRef.current = fitViewport(
      sizeRef.current.w,
      sizeRef.current.h,
      ringsRef.current[ringsRef.current.length - 1]?.r ?? RING_OUTER
    )
    selectedRingRef.current = null
    invalidate()
    setSelectedId(null)
  }

  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) {
      return
    }

    const { x, y } = localXY(e)
    const ringHit = pickRingLabel(x, y)
    hoveredRingRef.current = null
    // Nodes aren't draggable (static map) — remember which was pressed so a click
    // (press without movement) can select it; any drag just pans.
    const nodeId = ringHit == null ? (pickNode(x, y)?.id ?? null) : null
    dragRef.current = {
      id: nodeId,
      mode: 'pan',
      moved: false,
      ring: ringHit,
      sx: e.clientX,
      sy: e.clientY,
      vp: viewportRef.current
    }
  }

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current

    if (drag.mode === 'none') {
      const { x, y } = localXY(e)
      const ringHit = pickRingLabel(x, y)
      const id = ringHit == null ? (pickNode(x, y)?.id ?? null) : null
      // Links are the last fallback (only when not over a node/date).
      const linkKey = ringHit == null && id == null ? pickLink(x, y) : null

      if (id !== hoverRef.current || ringHit !== hoveredRingRef.current || linkKey !== hoveredLinkRef.current) {
        hoverRef.current = id
        hoveredRingRef.current = ringHit
        hoveredLinkRef.current = linkKey
        invalidate()
      }

      return
    }

    const dx = e.clientX - drag.sx
    const dy = e.clientY - drag.sy

    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
      drag.moved = true
    }

    if (drag.mode === 'pan') {
      // Taking manual control of the camera ends an auto-fit play-through.
      if (drag.moved) {
        setPlaying(false)
      }

      viewportRef.current = { ...drag.vp, x: drag.vp.x + dx, y: drag.vp.y + dy }
      invalidate()
    }
  }

  const endDrag = () => {
    const drag = dragRef.current

    // A click (press without movement) toggles a ring date, a node, or clears.
    if (drag.mode === 'pan' && !drag.moved) {
      // Double tap (trackpad tap-to-click may never emit a dblclick) resets view.
      if (doubleTapRef.current()) {
        resetView()
        dragRef.current = { id: null, mode: 'none', moved: false, ring: null, sx: 0, sy: 0, vp: viewportRef.current }

        return
      }

      // Independent toggles: a date and a node can both be selected.
      if (drag.ring != null) {
        selectedRingRef.current = selectedRingRef.current === drag.ring ? null : drag.ring
      } else if (drag.id) {
        setSelectedId(prev => (prev === drag.id ? null : drag.id))
      } else {
        selectedRingRef.current = null
        setSelectedId(null)
      }

      invalidate()
    }

    dragRef.current = { id: null, mode: 'none', moved: false, ring: null, sx: 0, sy: 0, vp: viewportRef.current }
  }

  const onMouseLeave = () => {
    hoverRef.current = null
    hoveredRingRef.current = null
    hoveredLinkRef.current = null
    invalidate()
    endDrag()
  }

  const onContextMenu = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    const { x, y } = localXY(e)
    const node = pickNode(x, y)

    if (!node) {
      return setMenuTarget(null)
    }

    setSelectedId(node.id)
    setMenuTarget({
      id: node.id,
      kind: node.kind === 'memory' ? 'memory' : 'skill',
      label: node.label,
      x: e.clientX,
      y: e.clientY
    })
  }

  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect()

    if (!rect) {
      return
    }

    // macOS smart zoom (two-finger double-tap) → reset (see lib/trackpad-gestures).
    if (isSmartZoomWheel(e)) {
      resetView()

      return
    }

    // Manual zoom takes over the camera from any auto-fit play-through.
    setPlaying(false)

    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    const vp = viewportRef.current
    const k = clamp(vp.k * (e.deltaY > 0 ? 0.9 : 1.1), ZOOM_MIN, ZOOM_MAX)
    viewportRef.current = { k, x: px - ((px - vp.x) / vp.k) * k, y: py - ((py - vp.y) / vp.k) * k }
    invalidate()
  }

  return (
    <div className="relative min-h-0 flex-1 overflow-hidden" ref={wrapRef}>
      <canvas
        className="block touch-none select-none text-foreground"
        onContextMenu={onContextMenu}
        onDoubleClick={resetView}
        onMouseDown={onMouseDown}
        onMouseLeave={onMouseLeave}
        onMouseMove={onMouseMove}
        onMouseUp={endDrag}
        onWheel={onWheel}
        ref={canvasRef}
      />

      <NodeContextMenu
        onClose={() => setMenuTarget(null)}
        onNodeRemoved={() => {
          setMenuTarget(null)
          setSelectedId(null)
        }}
        target={menuTarget}
      />

      {/* Timeline scrubber — centered along the top, clear of the close button.
          z-20 lifts it above the titlebar's app-region drag layer (z-10) so the
          scrubber receives pointer events instead of dragging the window. */}
      <div className="pointer-events-none absolute inset-x-0 top-6 z-20 flex justify-center px-12">
        <Timeline
          axis={timeAxis}
          memoryColor={memoryColor}
          onScrub={onScrub}
          onTogglePlay={onTogglePlay}
          playing={playing}
          revealStore={revealStore}
          ringStops={ringStops}
        />
      </div>

      {/* Share / import (WoW-talent-style code) — bottom-right, mirroring the legend. */}
      <div className="pointer-events-auto absolute bottom-2 right-2 z-20 [-webkit-app-region:no-drag]">
        <ShareControls imported={imported} onImport={importCode} onResetMap={onResetMap} shareCode={shareCode} />
      </div>

      {/* Legend — bottom-left, one entry per line like a conventional key. */}
      <div className="pointer-events-none absolute bottom-2 left-2 flex flex-col gap-1 text-[0.62rem] text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="inline-block size-2 rounded-full bg-[var(--theme-primary)]/80" /> skill
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block size-2 rotate-45" style={{ backgroundColor: memoryColor }} /> memory
        </span>
        <span className="text-[0.58rem] text-muted-foreground/65">core = oldest · outer = newer</span>
        <RevealLabel axis={timeAxis} revealStore={revealStore} />
      </div>
    </div>
  )
}
