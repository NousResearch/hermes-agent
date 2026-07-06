import { useCallback, useRef } from 'react'

import { cn } from '@/lib/utils'
import type { RunTimeline, ToolFamily } from '@/store/run-timeline'

const FAMILY_COLOR: Record<ToolFamily, string> = {
  terminal: '#6aa7ff',
  file: '#38d996',
  browser: '#a78bfa',
  think: '#22d3ee',
  delegation: '#f5b85b',
  web: '#f472b6',
  other: '#8b98ad'
}

export interface MinimapMark {
  x: number
  width: number
  color: string
  outlier: boolean
}

/**
 * Pure: compress the whole run into normalized [0,1] marks. Exported for tests.
 * Parent blocks keep their family color; child bars use a neutral color.
 */
export function minimapMarks(timeline: RunTimeline, nowMs: number): MinimapMark[] {
  const span = Math.max(1, Math.max(timeline.endMs, nowMs) - timeline.startMs)
  const marks: MinimapMark[] = []

  for (const lane of timeline.lanes) {
    if (lane.kind === 'blocks') {
      for (const block of lane.blocks) {
        const endMs = block.startMs + (block.durationMs ?? Math.max(0, nowMs - block.startMs))

        marks.push({
          x: (block.startMs - timeline.startMs) / span,
          width: Math.max(0.002, (endMs - block.startMs) / span),
          color: FAMILY_COLOR[block.family],
          outlier: block.isOutlier
        })
      }
    } else {
      const endMs = lane.endMs ?? nowMs

      marks.push({
        x: (lane.startMs - timeline.startMs) / span,
        width: Math.max(0.002, (endMs - lane.startMs) / span),
        color: '#2e3f5c',
        outlier: lane.bar?.isOutlier ?? false
      })
    }
  }

  return marks
}

interface MinimapProps {
  timeline: RunTimeline
  nowMs: number
  /** Visible window as normalized [0,1] fractions of the run. */
  windowStart: number
  windowEnd: number
  /** Called with a normalized [0,1] center when the user drags/clicks. */
  onSeek: (fraction: number) => void
  className?: string
}

export function Minimap({ timeline, nowMs, windowStart, windowEnd, onSeek, className }: MinimapProps) {
  const trackRef = useRef<HTMLDivElement>(null)
  const marks = minimapMarks(timeline, nowMs)

  const seekFromEvent = useCallback(
    (clientX: number) => {
      const el = trackRef.current

      if (!el) {
        return
      }

      const rect = el.getBoundingClientRect()
      const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width))

      onSeek(fraction)
    },
    [onSeek]
  )

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.currentTarget.setPointerCapture(e.pointerId)
      seekFromEvent(e.clientX)
    },
    [seekFromEvent]
  )

  const onPointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (e.buttons === 1) {
        seekFromEvent(e.clientX)
      }
    },
    [seekFromEvent]
  )

  const clampedStart = Math.min(1, Math.max(0, windowStart))
  const clampedWidth = Math.min(1, Math.max(0.02, windowEnd - windowStart))

  return (
    <div
      className={cn('relative h-9 overflow-hidden rounded-lg border border-(--ui-stroke-tertiary) bg-muted/20', className)}
      data-testid="timeline-minimap"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      ref={trackRef}
    >
      {marks.map((mark, i) => (
        <div
          className={cn('absolute top-1/2 h-2 -translate-y-1/2 rounded-[2px]', mark.outlier && 'ring-1 ring-red-400')}
          // Marks are positional; index key is stable for a given render.
          key={i}
          style={{ left: `${mark.x * 100}%`, width: `${mark.width * 100}%`, background: mark.color }}
        />
      ))}
      <div
        className="absolute top-0 h-full rounded-md border-2 border-primary/70 bg-primary/10"
        data-testid="minimap-window"
        style={{ left: `${clampedStart * 100}%`, width: `${clampedWidth * 100}%` }}
      />
    </div>
  )
}
