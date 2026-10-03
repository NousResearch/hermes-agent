import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SubagentNode } from '@/store/subagents'

import { STALE_EVIDENCE_MS, SubagentRow } from './index'

// An 8h-old frame and a live one are the same server state — running — but a
// frozen updatedAt is evidence the row is no longer actually moving. The row
// keeps reporting status/duration/age; only the liveness animation stops.
const NOW = 10_000_000

vi.mock('@/lib/use-enter-animation', () => ({ useEnterAnimation: () => undefined }))

vi.stubGlobal(
  'ResizeObserver',
  class {
    disconnect() {}
    observe() {}
    unobserve() {}
  }
)

const node = (overrides: Partial<SubagentNode> = {}): SubagentNode => ({
  id: 'worker',
  parentId: null,
  goal: 'Frozen task',
  status: 'running',
  taskCount: 1,
  taskIndex: 0,
  startedAt: NOW - 3_600_000,
  updatedAt: NOW - 1_000,
  durationSeconds: 4200,
  filesRead: [],
  filesWritten: [],
  stream: [{ at: NOW - 1_000, kind: 'progress', text: 'last words' }],
  children: [],
  ...overrides
})

afterEach(cleanup)

describe('SubagentRow evidence-aware animation', () => {
  it('stops animating a running row whose evidence is frozen past the threshold, keeping status, duration and age', () => {
    render(<SubagentRow node={node({ updatedAt: NOW - STALE_EVIDENCE_MS - 1_001 })} nowMs={NOW} />)

    // No live animation: no spinner (status glyph or stream tail), no shimmer.
    expect(screen.queryAllByRole('status')).toHaveLength(0)
    expect(screen.queryByLabelText('Streaming')).toBeNull()
    expect(screen.getByText('Frozen task').className).not.toContain('shimmer')

    // The row still reports its server status and its frozen bookkeeping.
    expect(screen.getByLabelText('Running')).toBeTruthy()
    expect(screen.getByText(/70m 0s/)).toBeTruthy()
    expect(screen.getByText(/updated 25m ago/)).toBeTruthy()
  })

  it('keeps a fresh running row animating exactly as before', () => {
    render(<SubagentRow node={node()} nowMs={NOW} />)

    expect(screen.getAllByRole('status').length).toBeGreaterThan(0)
    expect(screen.getByLabelText('Streaming')).toBeTruthy()
    expect(screen.getByText('Frozen task').className).toContain('shimmer')
  })

  it('treats the threshold as a boundary: at exactly STALE_EVIDENCE_MS the row is still live', () => {
    const view = render(<SubagentRow node={node({ updatedAt: NOW - STALE_EVIDENCE_MS })} nowMs={NOW} />)

    expect(screen.getAllByRole('status').length).toBeGreaterThan(0)
    expect(view.container.querySelector('[data-stale-evidence="true"]')).toBeNull()
  })
})
