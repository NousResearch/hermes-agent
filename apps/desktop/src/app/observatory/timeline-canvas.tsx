import { useCallback, useMemo, useRef, useState } from 'react'

import { cn } from '@/lib/utils'
import type { RunTimeline, TimelineLane, ToolBlock, ToolFamily } from '@/store/run-timeline'

// Layout constants (px).
const LANE_HEIGHT = 44
const LANE_GAP = 6
const BLOCK_HEIGHT = 22
const BLOCK_TOP = (LANE_HEIGHT - BLOCK_HEIGHT) / 2
const AXIS_HEIGHT = 24
const LABEL_WIDTH = 220
const MIN_BLOCK_PX = 3
const DEPTH_INDENT = 16

// Zoom bounds: ms represented by one pixel.
const MIN_MS_PER_PX = 2
const MAX_MS_PER_PX = 5_000
const DEFAULT_MS_PER_PX = 40

const FAMILY_COLOR: Record<ToolFamily, string> = {
  terminal: '#6aa7ff',
  file: '#38d996',
  browser: '#a78bfa',
  think: '#22d3ee',
  delegation: '#f5b85b',
  web: '#f472b6',
  other: '#8b98ad'
}

export interface VisibleBlock extends ToolBlock {
  /** px offset from the lane's content origin. */
  x: number
  /** px width (clamped to a minimum so 0-duration/tiny blocks stay clickable). */
  width: number
}

/**
 * Pure: project a lane's blocks onto pixels for the visible time window, and
 * drop blocks fully outside it (the primary scale mitigation — we never render
 * thousands of offscreen blocks). Exported for tests.
 */
export function visibleBlocks(
  blocks: readonly ToolBlock[],
  runStartMs: number,
  msPerPx: number,
  viewStartMs: number,
  viewEndMs: number,
  nowMs: number
): VisibleBlock[] {
  const out: VisibleBlock[] = []

  for (const block of blocks) {
    const endMs = block.startMs + (block.durationMs ?? Math.max(0, nowMs - block.startMs))

    // Skip blocks entirely outside the visible window.
    if (endMs < viewStartMs || block.startMs > viewEndMs) {
      continue
    }

    const x = (block.startMs - runStartMs) / msPerPx
    const rawWidth = (endMs - block.startMs) / msPerPx

    out.push({ ...block, x, width: Math.max(MIN_BLOCK_PX, rawWidth) })
  }

  return out
}

interface TimelineCanvasProps {
  timeline: RunTimeline
  /** Backend-epoch "now" for live (running) durations. */
  nowMs: number
  className?: string
}

export function TimelineCanvas({ timeline, nowMs, className }: TimelineCanvasProps) {
  const [msPerPx, setMsPerPx] = useState(DEFAULT_MS_PER_PX)
  const [scrollLeft, setScrollLeft] = useState(0)
  const [viewportWidth, setViewportWidth] = useState(900)
  const scrollRef = useRef<HTMLDivElement>(null)

  const runStartMs = timeline.startMs
  const runEndMs = Math.max(timeline.endMs, nowMs)
  const totalWidth = Math.max(1, (runEndMs - runStartMs) / msPerPx)

  const viewStartMs = runStartMs + scrollLeft * msPerPx
  const viewEndMs = viewStartMs + viewportWidth * msPerPx

  const onScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollLeft(e.currentTarget.scrollLeft)
    setViewportWidth(e.currentTarget.clientWidth)
  }, [])

  // Wheel + modifier zooms around the cursor; plain wheel scrolls.
  const onWheel = useCallback(
    (e: React.WheelEvent<HTMLDivElement>) => {
      if (!e.ctrlKey && !e.metaKey) {
        return
      }

      e.preventDefault()
      const factor = e.deltaY > 0 ? 1.2 : 1 / 1.2

      setMsPerPx(prev => Math.min(MAX_MS_PER_PX, Math.max(MIN_MS_PER_PX, prev * factor)))
    },
    []
  )

  const axisTicks = useMemo(() => {
    const ticks: { x: number; label: string }[] = []
    const spanMs = runEndMs - runStartMs
    const targetTicks = 8
    const rawStep = spanMs / targetTicks
    const step = niceStep(rawStep)

    for (let t = 0; t <= spanMs; t += step) {
      ticks.push({ x: t / msPerPx, label: formatOffset(t) })
    }

    return ticks
  }, [runStartMs, runEndMs, msPerPx])

  return (
    <div className={cn('flex min-h-0 flex-1 flex-col', className)} data-testid="timeline-canvas">
      <div className="flex min-h-0 flex-1">
        {/* Fixed lane-label gutter. */}
        <div className="shrink-0 border-r border-(--ui-stroke-tertiary)" style={{ width: LABEL_WIDTH }}>
          <div style={{ height: AXIS_HEIGHT }} />
          {timeline.lanes.map(lane => (
            <LaneLabel key={lane.id} lane={lane} />
          ))}
        </div>

        {/* Scrollable time area. */}
        <div
          className="min-w-0 flex-1 overflow-x-auto overflow-y-hidden"
          onScroll={onScroll}
          onWheel={onWheel}
          ref={scrollRef}
        >
          <div className="relative" style={{ width: totalWidth }}>
            {/* Axis */}
            <div className="sticky top-0 z-10 bg-background/80" style={{ height: AXIS_HEIGHT }}>
              {axisTicks.map(tick => (
                <div
                  className="absolute top-0 border-l border-(--ui-stroke-tertiary) pl-1 text-[0.6rem] text-muted-foreground/70"
                  key={tick.label}
                  style={{ left: tick.x, height: AXIS_HEIGHT }}
                >
                  {tick.label}
                </div>
              ))}
            </div>

            {/* Lanes */}
            {timeline.lanes.map(lane => (
              <LaneRow
                key={lane.id}
                lane={lane}
                msPerPx={msPerPx}
                nowMs={nowMs}
                runStartMs={runStartMs}
                viewEndMs={viewEndMs}
                viewStartMs={viewStartMs}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function LaneLabel({ lane }: { lane: TimelineLane }) {
  return (
    <div
      className="flex flex-col justify-center overflow-hidden px-2"
      style={{ height: LANE_HEIGHT, marginBottom: LANE_GAP, paddingLeft: 8 + lane.depth * DEPTH_INDENT }}
      title={lane.label}
    >
      <span className="truncate text-[0.72rem] font-medium text-foreground/90">{lane.label}</span>
      <span className="truncate text-[0.6rem] text-muted-foreground/60">
        {lane.kind === 'blocks' ? 'parent · per-tool' : `${lane.bar?.toolCount ?? 0} calls`}
        {lane.running ? ' · running' : ''}
      </span>
    </div>
  )
}

function LaneRow({
  lane,
  runStartMs,
  msPerPx,
  viewStartMs,
  viewEndMs,
  nowMs
}: {
  lane: TimelineLane
  runStartMs: number
  msPerPx: number
  viewStartMs: number
  viewEndMs: number
  nowMs: number
}) {
  return (
    <div
      className="relative border-b border-(--ui-stroke-tertiary)/40"
      data-testid={`lane-${lane.id}`}
      style={{ height: LANE_HEIGHT, marginBottom: LANE_GAP }}
    >
      {lane.kind === 'blocks'
        ? visibleBlocks(lane.blocks, runStartMs, msPerPx, viewStartMs, viewEndMs, nowMs).map(block => (
            <div
              className={cn('absolute rounded-[5px]', block.isOutlier && 'ring-2 ring-red-400')}
              data-family={block.family}
              data-outlier={block.isOutlier ? 'true' : undefined}
              data-testid={`block-${block.toolCallId}`}
              key={block.toolCallId}
              style={{
                left: block.x,
                width: block.width,
                top: BLOCK_TOP,
                height: BLOCK_HEIGHT,
                background: FAMILY_COLOR[block.family]
              }}
              title={`${block.name}${block.durationMs !== null ? ` · ${Math.round(block.durationMs)}ms` : ' · running'}`}
            />
          ))
        : renderBar(lane, runStartMs, msPerPx, nowMs)}
    </div>
  )
}

function renderBar(lane: TimelineLane, runStartMs: number, msPerPx: number, nowMs: number) {
  const endMs = lane.endMs ?? nowMs
  const x = (lane.startMs - runStartMs) / msPerPx
  const width = Math.max(MIN_BLOCK_PX, (endMs - lane.startMs) / msPerPx)

  return (
    <div
      className={cn(
        'absolute rounded-[5px] border border-(--ui-stroke-secondary) bg-[#2e3f5c]',
        lane.bar?.isOutlier && 'ring-2 ring-red-400'
      )}
      data-outlier={lane.bar?.isOutlier ? 'true' : undefined}
      data-testid={`bar-${lane.id}`}
      style={{ left: x, width, top: BLOCK_TOP, height: BLOCK_HEIGHT }}
      title={`${lane.label} · ${lane.bar?.toolCount ?? 0} calls`}
    />
  )
}

function niceStep(rawMs: number): number {
  const steps = [1_000, 2_000, 5_000, 10_000, 30_000, 60_000, 120_000, 300_000, 600_000, 1_800_000, 3_600_000]

  for (const s of steps) {
    if (rawMs <= s) {
      return s
    }
  }

  return steps[steps.length - 1]
}

function formatOffset(ms: number): string {
  const totalSec = Math.floor(ms / 1000)
  const m = Math.floor(totalSec / 60)
  const s = totalSec % 60

  return `${m}:${String(s).padStart(2, '0')}`
}
