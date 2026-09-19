import { describe, expect, it } from 'vitest'

import type { KanbanTask } from './types'
import { fmtSecs, runtimeCapBadge, staleBlocked } from './ui'

const task = (over: Partial<KanbanTask>): KanbanTask => ({ id: 't_1', status: 'todo', title: 'card', ...over })

describe('runtime cap badge', () => {
  it('stays off without a cap, without a start clock, or off the running lane', () => {
    expect(runtimeCapBadge(task({ max_runtime_seconds: 900, started_at: 1_000 }), 9_999)).toBeNull()
    expect(runtimeCapBadge(task({ started_at: 1_000, status: 'running' }), 9_999)).toBeNull()
    expect(runtimeCapBadge(task({ max_runtime_seconds: 900, status: 'running' }), 9_999)).toBeNull()
    expect(runtimeCapBadge(task({ max_runtime_seconds: 900, started_at: 1_000, status: 'blocked' }), 9_999)).toBeNull()
  })

  it('warns past half the cap and flags over-cap runs', () => {
    const capped = task({ max_runtime_seconds: 900, started_at: 1_000, status: 'running' })

    expect(runtimeCapBadge(capped, 1_450)).toBeNull() // exactly half — not yet
    expect(runtimeCapBadge(capped, 1_451)).toMatchObject({ cap: 900, elapsed: 451, kind: 'near' })
    expect(runtimeCapBadge(capped, 1_900)).toMatchObject({ kind: 'near' }) // at the cap it flips next tick
    expect(runtimeCapBadge(capped, 1_901)).toMatchObject({ kind: 'over' })
  })

  it('formats durations off the shared bucketing', () => {
    expect(fmtSecs(45)).toBe('45s')
    expect(fmtSecs(1_081)).toBe('18m')
    expect(fmtSecs(7_200)).toBe('2h')
  })
})

describe('stale blocked dot', () => {
  const now = 2_000_000

  it('shows only for blocked cards quiet for more than a day', () => {
    expect(staleBlocked(task({ last_event_at: now - 86_401, status: 'blocked' }), now)).toBe(true)
    expect(staleBlocked(task({ last_event_at: now - 86_400, status: 'blocked' }), now)).toBe(false)
    expect(staleBlocked(task({ last_event_at: now - 86_401, status: 'ready' }), now)).toBe(false)
    expect(staleBlocked(task({ status: 'blocked' }), now)).toBe(false)
  })
})
