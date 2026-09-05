import { expect, it } from 'vitest'

import { type AgentRow, mergeOverview } from './agent-overview'
import { overlayObservedStatuses } from './agent-overview-status'

it('uses stable Desktop status only for the exact observed runtime owner and never a same-id foreign row', () => {
  const makeSource = (connectionId: string) => ({
    connectionId,
    label: connectionId,
    kind: 'remote',
    state: 'unsupported' as const,
    sessions: [{ id: 'same', profile: 'default' }],
    live: [],
    canonical: [],
    profiles: [],
    total: 1,
    offset: 1,
    complete: true,
    liveCoverage: 'unavailable' as const,
    errors: []
  })

  const rows = mergeOverview(undefined, { fetchedAt: 1, sources: [makeSource('a'), makeSource('b')] }).rows

  const observed = [
    {
      connectionId: 'a',
      profile: 'default',
      id: 'same',
      runtimeId: 'run-a',
      status: 'needs-input' as const,
      socketOpen: true
    }
  ]

  const result = overlayObservedStatuses(rows, observed)
  expect(result[0]).toMatchObject({ attention: 'needs-you', runtimeId: 'run-a', stale: false })
  expect(result[1]).toBe(rows[1])
  expect(result[1]?.attention).toBe('idle')
  expect(overlayObservedStatuses(result, observed)).toBe(result)
  expect(overlayObservedStatuses(rows, [{ ...observed[0]!, status: 'background' }])[0]?.attention).toBe('working')
})

it.each([false, true])(
  'never revives an explicitly stale source from a retained event ledger (socket open %s)',
  socketOpen => {
    const row: AgentRow = {
      key: '["a","default","same"]',
      connectionId: 'a',
      sourceLabel: 'A',
      profile: 'default',
      owner: 'default',
      description: '',
      id: 'same',
      resolvedId: 'same',
      history: 'unknown',
      title: '',
      preview: '',
      model: '',
      provider: '',
      platform: '',
      attention: 'idle',
      canonical: false,
      stale: true,
      lastActive: 1
    }

    const rows = [row]

    const result = overlayObservedStatuses(rows, [
      { connectionId: 'a', profile: 'default', id: 'same', runtimeId: 'old-runtime', status: 'needs-input', socketOpen }
    ])

    expect(result).toBe(rows)
    expect(result[0]).toMatchObject({ stale: true, attention: 'idle' })
    expect(result[0]?.runtimeId).toBeUndefined()
  }
)

it('ignores a disconnected owner ledger even when HTTP history is available', () => {
  const row: AgentRow = {
    key: '["a","default","same"]',
    connectionId: 'a',
    sourceLabel: 'A',
    profile: 'default',
    owner: 'default',
    description: '',
    id: 'same',
    resolvedId: 'same',
    history: 'unknown',
    title: '',
    preview: '',
    model: '',
    provider: '',
    platform: '',
    attention: 'idle',
    canonical: false,
    stale: false,
    lastActive: 1
  }

  const rows = [row]
  expect(
    overlayObservedStatuses(rows, [
      {
        connectionId: 'a',
        profile: 'default',
        id: 'same',
        runtimeId: 'old-runtime',
        status: 'working',
        socketOpen: false
      }
    ])
  ).toBe(rows)
})
