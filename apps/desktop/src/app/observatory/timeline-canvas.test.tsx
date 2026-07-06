import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { RunTimeline, TimelineLane, ToolBlock } from '@/store/run-timeline'

import { minimapMarks } from './minimap'
import { TimelineCanvas, visibleBlocks } from './timeline-canvas'

const RUN_START = 1_700_000_000_000
const NOW = RUN_START + 60_000

function block(id: string, offsetMs: number, durationMs: number | null, isOutlier = false): ToolBlock {
  return {
    toolCallId: id,
    name: 'terminal',
    family: 'terminal',
    startMs: RUN_START + offsetMs,
    durationMs,
    isOutlier
  }
}

function parentLane(blocks: ToolBlock[]): TimelineLane {
  return {
    id: 'parent',
    parentLaneId: null,
    depth: 0,
    label: 'Parent session',
    kind: 'blocks',
    startMs: RUN_START,
    endMs: null,
    running: true,
    blocks,
    bar: null
  }
}

function childLane(id: string, offsetMs: number, durationMs: number | null, outlier = false): TimelineLane {
  return {
    id,
    parentLaneId: 'parent',
    depth: 1,
    label: `child ${id}`,
    kind: 'bar',
    startMs: RUN_START + offsetMs,
    endMs: durationMs === null ? null : RUN_START + offsetMs + durationMs,
    running: durationMs === null,
    blocks: [],
    bar: { toolCount: 4, isOutlier: outlier }
  }
}

describe('visibleBlocks (horizontal windowing)', () => {
  it('drops blocks fully outside the visible window', () => {
    const blocks = [
      block('a', 0, 1000), // 0-1s
      block('b', 30_000, 1000), // 30-31s
      block('c', 55_000, 1000) // 55-56s
    ]

    // Window: 25s..40s at 40 ms/px
    const vis = visibleBlocks(blocks, RUN_START, 40, RUN_START + 25_000, RUN_START + 40_000, NOW)

    expect(vis.map(v => v.toolCallId)).toEqual(['b'])
  })

  it('clamps tiny/zero-duration blocks to a minimum width', () => {
    const vis = visibleBlocks([block('z', 0, 0)], RUN_START, 40, RUN_START, RUN_START + 60_000, NOW)
    expect(vis[0].width).toBeGreaterThanOrEqual(3)
  })

  it('computes x from the run start and the ms/px scale', () => {
    const vis = visibleBlocks([block('a', 4000, 2000)], RUN_START, 40, RUN_START, RUN_START + 60_000, NOW)
    // 4000ms / 40 mspp = 100px
    expect(vis[0].x).toBe(100)
    expect(vis[0].width).toBe(50) // 2000/40
  })

  it('gives a running block (null duration) a live width up to now', () => {
    const vis = visibleBlocks([block('run', 0, null)], RUN_START, 40, RUN_START, RUN_START + 60_000, NOW)
    // now - start = 60_000ms / 40 = 1500px
    expect(vis[0].width).toBeCloseTo(1500, 0)
  })

  it('handles a 2000-block lane by only returning the windowed slice', () => {
    const many: ToolBlock[] = Array.from({ length: 2000 }, (_, i) => block(`b${i}`, i * 1000, 500))
    // Narrow window: 100s..110s -> ~10 blocks
    const vis = visibleBlocks(many, RUN_START, 40, RUN_START + 100_000, RUN_START + 110_000, RUN_START + 2_000_000)

    expect(vis.length).toBeGreaterThan(0)
    expect(vis.length).toBeLessThan(30)
  })
})

describe('minimapMarks', () => {
  it('emits one mark per parent block plus one per child bar, outliers flagged', () => {
    const timeline: RunTimeline = {
      lanes: [
        parentLane([block('a', 0, 1000), block('b', 30_000, 25_000, true)]),
        childLane('c1', 5000, 10_000)
      ],
      startMs: RUN_START,
      endMs: RUN_START + 60_000
    }

    const marks = minimapMarks(timeline, NOW)
    expect(marks).toHaveLength(3) // 2 parent blocks + 1 child bar
    expect(marks.some(m => m.outlier)).toBe(true)

    // all normalized into [0,1]
    for (const m of marks) {
      expect(m.x).toBeGreaterThanOrEqual(0)
      expect(m.x).toBeLessThanOrEqual(1)
    }
  })
})

describe('TimelineCanvas render', () => {
  afterEach(cleanup)

  it('renders lanes, parent blocks, and child bars', () => {
    const timeline: RunTimeline = {
      lanes: [parentLane([block('a', 0, 1000), block('b', 5000, 2000)]), childLane('c1', 2000, 8000, true)],
      startMs: RUN_START,
      endMs: RUN_START + 60_000
    }

    render(<TimelineCanvas nowMs={NOW} timeline={timeline} />)

    expect(screen.getByTestId('timeline-canvas')).toBeTruthy()
    expect(screen.getByTestId('lane-parent')).toBeTruthy()
    expect(screen.getByTestId('lane-c1')).toBeTruthy()
    expect(screen.getByTestId('block-a')).toBeTruthy()
    expect(screen.getByTestId('bar-c1')).toBeTruthy()
    // outlier child bar carries the flag
    expect(screen.getByTestId('bar-c1').getAttribute('data-outlier')).toBe('true')
  })
})
