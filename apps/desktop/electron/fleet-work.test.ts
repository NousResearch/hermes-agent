import { describe, expect, it } from 'vitest'

import {
  FLEET_WORKER_ACTIVE_WINDOW_S,
  FLEET_WORKER_SESSIONS_PATH,
  FLEET_WORKER_SOURCES,
  fleetWorkerSessionsFromBody
} from './fleet-work'

const now = 1_800_000_000

const row = (over: Record<string, unknown> = {}) => ({
  ended_at: null,
  id: 'sess-1',
  last_active: now - 10,
  profile: 'engineer1',
  source: 'kanban',
  title: 'Implement the card',
  ...over
})

describe('fleetWorkerSessionsFromBody', () => {
  it('keeps the newest open worker session per profile', () => {
    const work = fleetWorkerSessionsFromBody(
      {
        sessions: [
          row({ id: 'older', last_active: now - 60 }),
          row({ id: 'newer', last_active: now - 5 }),
          row({ id: 'other', profile: 'qa', last_active: now - 30 })
        ]
      },
      now
    )

    expect(work.engineer1).toMatchObject({ sessionId: 'newer', source: 'kanban', lastActive: now - 5 })
    expect(work.qa).toMatchObject({ sessionId: 'other', lastActive: now - 30 })
  })

  it('drops a session whose heartbeat is outside the liveness window', () => {
    const work = fleetWorkerSessionsFromBody(
      { sessions: [row({ id: 'dead', last_active: now - FLEET_WORKER_ACTIVE_WINDOW_S - 1 })] },
      now
    )

    expect(work).toEqual({})
  })

  it('keeps a session whose heartbeat is exactly on the window edge', () => {
    const work = fleetWorkerSessionsFromBody(
      { sessions: [row({ id: 'edge', last_active: now - (FLEET_WORKER_ACTIVE_WINDOW_S - 1) })] },
      now
    )

    expect(work.engineer1?.sessionId).toBe('edge')
  })

  it('ignores a finished session even when its last_active looks fresh', () => {
    const work = fleetWorkerSessionsFromBody(
      { sessions: [row({ ended_at: now - 5, id: 'done', last_active: now - 5 })] },
      now
    )

    expect(work).toEqual({})
  })

  it('skips malformed rows instead of inventing a profile', () => {
    const work = fleetWorkerSessionsFromBody(
      {
        sessions: [
          null,
          'nope',
          row({ id: '' }),
          row({ profile: '   ' }),
          row({ last_active: 'not-a-number' }),
          row({ profile: 'ok', id: 'kept' })
        ]
      },
      now
    )

    expect(Object.keys(work)).toEqual(['ok'])
  })

  it('omits an empty title rather than carrying a blank label', () => {
    const work = fleetWorkerSessionsFromBody({ sessions: [row({ title: '  ' })] }, now)

    expect(work.engineer1).not.toHaveProperty('title')
  })

  it('returns an empty record for a body that is not a session page', () => {
    expect(fleetWorkerSessionsFromBody(null, now)).toEqual({})
    expect(fleetWorkerSessionsFromBody({}, now)).toEqual({})
    expect(fleetWorkerSessionsFromBody({ sessions: 'nope' }, now)).toEqual({})
  })

  it('scopes the request to the worker sources that listings deny-list', () => {
    expect(FLEET_WORKER_SOURCES).toEqual(['kanban', 'tool', 'oneshot'])
    expect(FLEET_WORKER_SESSIONS_PATH).toContain('sources=kanban,tool,oneshot')
    expect(FLEET_WORKER_SESSIONS_PATH).toContain('order=recent')
  })
})
