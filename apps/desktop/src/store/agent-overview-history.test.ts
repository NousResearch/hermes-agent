import { expect, it } from 'vitest'

import type { OverviewSource } from '@/global'

import { mergeOverview } from './agent-overview'

const source = (overrides: Partial<OverviewSource> = {}): OverviewSource => ({
  connectionId: 'b',
  label: 'B',
  kind: 'remote',
  state: 'ready',
  sessions: [],
  canonical: [],
  live: [],
  profiles: [],
  total: 0,
  offset: 0,
  complete: true,
  liveCoverage: 'process',
  errors: [],
  ...overrides
})

it('projects only explicit valid counts into history knowledge, not preview or runtime presence', () => {
  for (const [message_count, history] of [
    [undefined, 'unknown'],
    [0, 'known-empty'],
    [2, 'known-history'],
    [-1, 'unknown'],
    [NaN, 'unknown'],
    [Infinity, 'unknown']
  ] as const) {
    const rows = mergeOverview(undefined, {
      fetchedAt: 1,
      sources: [source({ live: [{ id: 'empty', profile: 'analyst', runtime_id: 'run', preview: '', message_count }] })]
    }).rows

    expect(rows[0]?.history).toBe(history)
  }
})

it('preserves positive history over zero live counts within aliases and across refreshes without crossing owners', () => {
  const persisted = { id: 'tip', profile: 'analyst', message_count: 4 }
  const live = { id: 'tip', profile: 'analyst', runtime_id: 'run', message_count: 0 }
  const canonical = [{ id: 'root', resolved_id: 'tip', profile: 'analyst', message_count: 0 }]

  const first = mergeOverview(undefined, {
    fetchedAt: 1,
    sources: [source({ sessions: [persisted], canonical, live: [live] })]
  })

  expect(first.rows[0]).toMatchObject({ id: 'root', resolvedId: 'tip', history: 'known-history' })

  const next = mergeOverview(first, {
    fetchedAt: 2,
    sources: [
      source({ canonical, live: [live] }),
      source({ connectionId: 'c', live: [live] }),
      source({ connectionId: 'd', live: [{ ...live, profile: 'other' }] })
    ]
  })

  expect(next.rows.map(row => [row.connectionId, row.history])).toEqual([
    ['b', 'known-history'],
    ['c', 'known-empty'],
    ['d', 'known-empty']
  ])
})
