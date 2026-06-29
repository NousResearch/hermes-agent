import { useStore } from '@nanostores/react'
import { scaleLinear } from 'd3-scale'
import { select } from 'd3-selection'
import {
  zoom as d3Zoom,
  type D3ZoomEvent,
  type ZoomBehavior,
  zoomIdentity,
  type ZoomTransform,
  zoomTransform
} from 'd3-zoom'
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { ToolIcon } from '@/components/ui/tool-icon'
import { cn } from '@/lib/utils'
import {
  $hoveredSpanId,
  $selectedSpanId,
  $traceLabelsCollapsed,
  clearHoveredSpan,
  flattenSpanTree,
  type TraceDoc,
  type TraceSpanNode
} from '@/store/trace'

import { fmtDuration } from './format'
import { barClass, spanIconName } from './span-style'
import { buildTimeMap, niceTicks } from './time-map'

export const ROW_HEIGHT = 26
const LABEL_WIDTH = 280
const LABEL_MAX_WIDTH = 240
const MAX_ZOOM = 5000

export function TraceWaterfall({ trace, viewKey }: { trace: TraceDoc; viewKey: string }) {
  const nodes = useMemo(() => flattenSpanTree(trace), [trace])
  const selectedId = useStore($selectedSpanId)
  const hoveredId = useStore($hoveredSpanId)
  const collapsed = useStore($traceLabelsCollapsed)
  const trackRef = useRef<HTMLDivElement>(null)
  const bodyRef = useRef<HTMLDivElement>(null)
  const zoomRef = useRef<ZoomBehavior<HTMLDivElement, unknown> | null>(null)
  const [size, setSize] = useState({ height: 0, width: 0 })
  const [availHeight, setAvailHeight] = useState(0)
  const [transform, setTransform] = useState<ZoomTransform>(zoomIdentity)

  // Measure the track so the time scale + zoom extents track its real width.
  useLayoutEffect(() => {
    const el = trackRef.current

    if (!el) {
      return
    }

    const ro = new ResizeObserver(([entry]) => {
      if (entry) {
        setSize({ height: entry.contentRect.height, width: entry.contentRect.width })
      }
    })

    ro.observe(el)

    return () => ro.disconnect()
  }, [])

  // Available height of the scroll area, so the track can fill it when there are
  // few rows (and still grow + scroll when there are many).
  useLayoutEffect(() => {
    const el = bodyRef.current

    if (!el) {
      return
    }

    const ro = new ResizeObserver(([entry]) => {
      if (entry) {
        setAvailHeight(entry.contentRect.height)
      }
    })

    ro.observe(el)

    return () => ro.disconnect()
  }, [])

  const tmap = useMemo(
    () => buildTimeMap(nodes, trace.start, trace.end || trace.start + 1),
    [nodes, trace.start, trace.end]
  )

  const xScale = useMemo(
    () => scaleLinear().domain([0, tmap.totalV]).range([0, Math.max(1, size.width)]),
    [tmap.totalV, size.width]
  )

  // d3-zoom owns drag-pan + click/drag separation; we drive the wheel ourselves.
  // Attached ONCE on mount (not gated on measured size) so the gesture is live
  // immediately — gating on size.width was the regression that "lost" zoom when
  // the grid delayed the first measurement.
  useEffect(() => {
    const el = trackRef.current

    if (!el) {
      return
    }

    const behavior = d3Zoom<HTMLDivElement, unknown>()
      .scaleExtent([1, MAX_ZOOM])
      .clickDistance(4)
      .filter((event: Event) => event.type !== 'wheel' && !(event as MouseEvent).button)
      .on('zoom', (event: D3ZoomEvent<HTMLDivElement, unknown>) => setTransform(event.transform))

    zoomRef.current = behavior
    const sel = select(el)
    sel.call(behavior)
    sel.on('dblclick.zoom', null)

    // Wheel routing on the track itself:
    //   ⌘/Ctrl + wheel → zoom toward the cursor · horizontal/shift wheel → pan
    //   time · plain vertical wheel → fall through so the row list scrolls.
    const onWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault()
        e.stopPropagation()
        const px = e.clientX - el.getBoundingClientRect().left
        behavior.scaleBy(select(el), Math.exp(-e.deltaY * 0.002), [px, 0])

        return
      }

      const horizontal = Math.abs(e.deltaX) > Math.abs(e.deltaY)
      const dx = horizontal ? e.deltaX : e.shiftKey ? e.deltaY : 0

      if (!dx) {
        return // plain vertical → let the list scroll
      }

      e.preventDefault()
      e.stopPropagation()
      behavior.translateBy(select(el), -dx / zoomTransform(el).k, 0)
    }

    el.addEventListener('wheel', onWheel, { passive: false })

    return () => {
      sel.on('.zoom', null)
      el.removeEventListener('wheel', onWheel)
    }
  }, [])

  // Vertical drag scrubs the actual scroll container (the body owns row
  // position; labels + canvas stay in sync because they share this scroller).
  useEffect(() => {
    const scroller = bodyRef.current

    if (!scroller) {
      return
    }

    let lastY: null | number = null

    const stop = () => {
      lastY = null
      window.removeEventListener('mousemove', move, { capture: true })
      window.removeEventListener('mouseup', stop, { capture: true })
    }

    const move = (e: MouseEvent) => {
      if (!(e.buttons & 1) || lastY === null) {
        stop()

        return
      }

      scroller.scrollTop -= e.clientY - lastY
      lastY = e.clientY
    }

    const start = (e: MouseEvent) => {
      if (e.button !== 0) {
        return
      }

      lastY = e.clientY
      window.addEventListener('mousemove', move, { capture: true })
      window.addEventListener('mouseup', stop, { capture: true })
    }

    scroller.addEventListener('mousedown', start, { capture: true })

    return () => {
      scroller.removeEventListener('mousedown', start, { capture: true })
      stop()
    }
  }, [])

  // Keep the pan/zoom extents in sync with the measured track size.
  useEffect(() => {
    const behavior = zoomRef.current

    if (!behavior || size.width === 0) {
      return
    }

    behavior
      .extent([
        [0, 0],
        [size.width, size.height]
      ])
      .translateExtent([
        [0, 0],
        [size.width, size.height]
      ])
  }, [size.width, size.height])

  // Reset the viewport only on an explicit navigation (session switch or pinning
  // a different turn) — NOT when the followed turn settles live→DB, and not on
  // live ticks. Keeps the user's zoom/pan; no auto-nav.
  useEffect(() => {
    const el = trackRef.current

    if (el && zoomRef.current) {
      select(el).call(zoomRef.current.transform, zoomIdentity)
    } else {
      setTransform(zoomIdentity)
    }
  }, [viewKey])

  // Preserve the visible *real-time* window across tmap changes that AREN'T a
  // nav (same viewKey): live ticks growing the trace, and the live→DB fold where
  // the compressed axis (totalV) shifts. Without this, a zoomed-in view would
  // drift each tick and the fold would reframe/jump. We translate the OLD view's
  // edges back to real time, re-project them through the NEW map, and re-apply
  // the transform — so the same span stays put while the bars get more exact.
  const prevTmapRef = useRef(tmap)
  const prevViewKeyRef = useRef(viewKey)
  useLayoutEffect(() => {
    const el = trackRef.current
    const behavior = zoomRef.current
    const width = size.width

    // Nav reset owns viewKey changes; just resync our baselines and bail.
    if (viewKey !== prevViewKeyRef.current) {
      prevViewKeyRef.current = viewKey
      prevTmapRef.current = tmap

      return
    }

    const oldMap = prevTmapRef.current

    if (oldMap === tmap || !el || !behavior || width === 0) {
      prevTmapRef.current = tmap

      return
    }

    const t = zoomTransform(el)
    prevTmapRef.current = tmap

    // Full view (identity) → keep showing the whole trace as it grows; nothing
    // to preserve, and re-projecting would fight the natural "fit all".
    if (t.k <= 1.0001 && Math.abs(t.x) < 0.5) {
      return
    }

    const clampV = (v: number, total: number) => Math.max(0, Math.min(total, v))
    const oldXs = scaleLinear().domain([0, oldMap.totalV]).range([0, width])
    const zxOld = t.rescaleX(oldXs)
    const realStart = oldMap.toReal(clampV(zxOld.invert(0), oldMap.totalV))
    const realEnd = oldMap.toReal(clampV(zxOld.invert(width), oldMap.totalV))

    const newXs = scaleLinear().domain([0, tmap.totalV]).range([0, width])
    const bx0 = newXs(tmap.toV(realStart))
    const bx1 = newXs(tmap.toV(realEnd))
    const k = Math.max(1, Math.min(MAX_ZOOM, width / Math.max(1, bx1 - bx0)))
    const x = Math.min(0, Math.max(width * (1 - k), -k * bx0))

    select(el).call(behavior.transform, zoomIdentity.scale(k).translate(x / k, 0))
  }, [tmap, viewKey, size.width])

  // Latest nodes/tmap read via refs so the zoom-to-span effect can fire ONLY on
  // a selection change — never on live ticks (which would re-snap the view).
  const nodesRef = useRef(nodes)
  nodesRef.current = nodes
  const tmapRef = useRef(tmap)
  tmapRef.current = tmap

  // Selecting a span (clicking a row) zooms the timeline to frame it (~70%) and
  // scrolls to it. Keyed on the selection alone so it never fights live updates.
  useEffect(() => {
    const el = trackRef.current
    const behavior = zoomRef.current

    if (!el || !behavior || !selectedId) {
      return
    }

    const span = nodesRef.current.find(n => n.id === selectedId)
    const width = el.clientWidth

    if (!span || !width) {
      return
    }

    const map = tmapRef.current
    const xs = scaleLinear().domain([0, map.totalV]).range([0, width])
    const bx0 = xs(map.toV(span.start))
    const bx1 = xs(map.toV(span.end))
    const bw = Math.max(2, bx1 - bx0)
    const k = Math.max(1, Math.min(MAX_ZOOM, (width * 0.7) / bw))
    const center = (bx0 + bx1) / 2
    // Clamp the translate so the view can't slide past the start (left gap) or
    // the end — matching d3's translateExtent, but on the target so there's no
    // "show gap then snap back".
    const x = Math.min(0, Math.max(width * (1 - k), width / 2 - k * center))
    const next = zoomIdentity.scale(k).translate(x / k, 0)
    select(el).transition().duration(250).call(behavior.transform, next)
  }, [selectedId])

  const view = useMemo(() => {
    if (size.width === 0) {
      return { end: tmap.totalV, start: 0 }
    }

    const zx = transform.rescaleX(xScale)

    return { end: zx.invert(size.width), start: zx.invert(0) }
  }, [transform, xScale, size.width, tmap.totalV])

  // view is in compressed (virtual) coordinates.
  const viewSpan = Math.max(1e-6, view.end - view.start)
  const ticks = useMemo(() => niceTicks(view.start, view.end), [view.start, view.end])
  // pctV: a virtual coord → %; pct: a real time → % (via the compression map).
  const pctV = (vv: number) => ((vv - view.start) / viewSpan) * 100
  const pct = (t: number) => pctV(tmap.toV(t))

  const resetView = () => {
    const el = trackRef.current

    if (el && zoomRef.current) {
      select(el).transition().duration(200).call(zoomRef.current.transform, zoomIdentity)
    }
  }

  // One column template shared by the ruler and the body so the time axis and
  // the bars can never drift out of alignment. Column 1 is the label tree (a
  // thin gutter when collapsed, just wide enough for the expand toggle); the
  // `gap-x-2` between the columns separates labels from the chart.
  const cols = `${collapsed ? '1.5rem' : `${LABEL_WIDTH}px`} minmax(0, 1fr)`
  // Fill the scroll area when rows are few; grow + scroll when they're many.
  const bodyHeight = Math.max(nodes.length * ROW_HEIGHT, availHeight)

  return (
    <div className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
      {/* Ruler — shares the body's column template; column 1 holds the collapse
          toggle (in flow so it never overlaps labels), column 2 the time axis. */}
      <div
        className="grid shrink-0 items-center gap-x-2 text-[0.62rem] text-muted-foreground/70"
        style={{ gridTemplateColumns: cols, height: ROW_HEIGHT }}
      >
        <div className="flex items-center justify-end">
          <button
            className="rounded p-0.5 text-muted-foreground/55 hover:bg-foreground/10 hover:text-foreground"
            onClick={() => $traceLabelsCollapsed.set(!collapsed)}
            title={collapsed ? 'Show labels' : 'Hide labels'}
            type="button"
          >
            <Codicon name={collapsed ? 'chevron-right' : 'chevron-left'} size="0.9rem" />
          </button>
        </div>
        <div className="relative h-full overflow-hidden">
          {ticks.map(t => {
            const left = pctV(t)

            if (left < 0 || left > 100) {
              return null
            }

            // Edge ticks align inward so the first/last label isn't sliced by the
            // track's clip; interior ticks center on their gridline.
            const edge = left < 4 ? 'left' : left > 96 ? 'right' : 'center'

            return (
              <span
                className={cn(
                  'absolute top-1/2 -translate-y-1/2 px-1 whitespace-nowrap tabular-nums',
                  edge === 'left' ? 'translate-x-0' : edge === 'right' ? '-translate-x-full' : '-translate-x-1/2'
                )}
                key={t}
                style={{ left: `${left}%` }}
              >
                {fmtDuration(tmap.toReal(t) - trace.start)}
              </span>
            )
          })}
        </div>
      </div>

      {/* Body — same column template; vertically scrolls labels + track together.
          scrollbar-gutter stays stable so switching between a short turn (no
          scrollbar) and the full session (scrollbar) never reflows the track. */}
      <div
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain [scrollbar-gutter:stable]"
        ref={bodyRef}
      >
        <div className="grid min-h-full gap-x-2" style={{ gridTemplateColumns: cols }}>
          {/* Column 1: label index (empty cell when collapsed keeps the grid 2-wide) */}
          {collapsed ? (
            <div />
          ) : (
            <div>
              {nodes.map(node => (
                <SpanLabel active={node.id === selectedId || node.id === hoveredId} key={node.id} node={node} />
              ))}
            </div>
          )}

          {/* Column 2: time track */}
          <div
            className="relative cursor-grab touch-none overflow-hidden select-none active:cursor-grabbing"
            onClick={() => $selectedSpanId.set(null)}
            onDoubleClick={resetView}
            ref={trackRef}
            style={{ height: bodyHeight }}
          >
            {ticks.map(t => {
              const left = pctV(t)

              // Skip the flush-left gridline at t=0 — it reads as a border.
              if (left <= 0.1 || left > 100) {
                return null
              }

              return (
                <div
                  className="pointer-events-none absolute top-0 bottom-0 w-px bg-border/30"
                  key={t}
                  style={{ left: `${left}%` }}
                />
              )
            })}

            {/* Collapsed-idle markers: a dashed seam where dead time was removed. */}
            {tmap.gaps.map(g => {
              const left = pctV(g.v0)
              const right = pctV(g.v1)

              if (right < 0 || left > 100) {
                return null
              }

              return (
                <div
                  className="pointer-events-none absolute top-0 bottom-0 flex items-start justify-center border-l border-dashed border-border/50 bg-foreground/[0.03]"
                  key={`gap-${g.v0}`}
                  style={{ left: `${left}%`, width: `${Math.max(0.4, right - left)}%` }}
                >
                  <span className="mt-0.5 rounded bg-background/70 px-1 text-[0.55rem] whitespace-nowrap text-muted-foreground/55">
                    {fmtDuration(g.r1 - g.r0)}
                  </span>
                </div>
              )
            })}

            {nodes.map((node, i) => {
              const left = pct(node.start)
              const width = (tmap.toV(node.end) - tmap.toV(node.start)) / viewSpan
              const active = node.id === selectedId || node.id === hoveredId

              return (
                <div
                  className={cn(
                    'absolute right-0 left-0 transition-colors duration-100 ease-out hover:bg-foreground/[0.035] hover:transition-none',
                    active && 'bg-foreground/[0.035]'
                  )}
                  key={node.id}
                  onMouseEnter={() => $hoveredSpanId.set(node.id)}
                  onMouseLeave={() => clearHoveredSpan(node.id)}
                  style={{ height: ROW_HEIGHT, top: i * ROW_HEIGHT }}
                >
                  <button
                    className={cn(
                      'absolute top-1/2 h-4 -translate-y-1/2 overflow-hidden rounded-[3px] transition-[filter] hover:brightness-125',
                      barClass(node),
                      node.status === 'running' && 'animate-pulse',
                      active && 'ring-1 ring-foreground/70'
                    )}
                    onClick={e => {
                      e.stopPropagation()
                      $selectedSpanId.set(node.id)
                    }}
                    style={{ left: `${left}%`, minWidth: 2, width: `${width * 100}%` }}
                    type="button"
                  />
                  {/* On-lane label: name + duration, anchored at the bar start.
                      Only when the start is in view AND the bar is wide enough to
                      host it (or it's active) — otherwise sliver bars (e.g. a
                      sub-second span pinned at t=0) leave a floating duration
                      stuck at the left edge. The left tree always has the full
                      label, so slivers lose nothing. */}
                  {left >= 0 && left < 100 && (active || width * size.width >= 32) ? (
                    <span
                      className="pointer-events-none absolute inset-y-0 flex items-center gap-1.5 truncate pl-1.5 text-[0.6rem] text-white/85"
                      style={{ left: `${left}%`, maxWidth: LABEL_MAX_WIDTH }}
                    >
                      <span className="truncate">{node.name}</span>
                      <span className="shrink-0 text-white/55">{fmtDuration(node.duration)}</span>
                    </span>
                  ) : null}
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

function SpanLabel({ active, node }: { active: boolean; node: TraceSpanNode }) {
  return (
    <button
      className={cn(
        'flex w-full items-center gap-1.5 truncate pr-2 pl-2 text-left text-[0.72rem] text-(--ui-text-secondary) transition-colors duration-100 ease-out hover:bg-(--ui-row-hover-background) hover:text-foreground hover:transition-none',
        active && 'bg-(--ui-row-active-background) text-foreground'
      )}
      onClick={() => $selectedSpanId.set(node.id)}
      onMouseEnter={() => $hoveredSpanId.set(node.id)}
      onMouseLeave={() => clearHoveredSpan(node.id)}
      style={{ height: ROW_HEIGHT, paddingLeft: 8 + node.depth * 14 }}
      type="button"
    >
      <ToolIcon
        className={cn('shrink-0', node.status === 'error' ? 'text-red-500' : 'text-muted-foreground/60')}
        name={spanIconName(node)}
        size="0.8rem"
      />
      <span className="min-w-0 flex-1 truncate text-foreground/85">{node.name}</span>
      <span className="shrink-0 tabular-nums text-[0.62rem] text-muted-foreground/55">{fmtDuration(node.duration)}</span>
    </button>
  )
}
