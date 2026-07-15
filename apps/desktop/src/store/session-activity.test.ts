import { beforeEach, describe, expect, it } from 'vitest'

import { $sessions, $workingSessionIds } from './session'
import { $sessionActivityIds, deriveSessionActivityIds } from './session-activity'
import { $subagentsBySession, type SubagentProgress, upsertSubagent } from './subagents'

const child = (overrides: Partial<SubagentProgress> = {}): SubagentProgress => ({
  id: 'child-1',
  parentId: null,
  goal: 'Independent review',
  status: 'running',
  taskCount: 1,
  taskIndex: 0,
  startedAt: 1,
  updatedAt: 1,
  filesRead: [],
  filesWritten: [],
  stream: [],
  ...overrides
})

describe('session activity ids', () => {
  beforeEach(() => {
    $sessions.set([])
    $workingSessionIds.set([])
    $subagentsBySession.set({})
  })

  it('keeps the parent and child sessions active while an independent review runs', () => {
    expect(
      deriveSessionActivityIds([], {
        'runtime-parent': [
          child({ ownerSessionId: 'stored-parent', sessionId: 'stored-review' }),
          child({ id: 'child-done', ownerSessionId: 'stored-parent', sessionId: 'stored-done', status: 'completed' })
        ]
      })
    ).toEqual(['stored-parent', 'stored-review'])
  })

  it('falls back to the event session id for older subagent events without a durable owner id', () => {
    expect(deriveSessionActivityIds([], { 'stored-parent': [child()] })).toEqual(['stored-parent'])
  })

  it('projects an active parent lineage root onto its current continuation row', () => {
    expect(
      deriveSessionActivityIds(
        [],
        {
          'runtime-parent': [child({ ownerSessionId: 'lineage-root' })]
        },
        [{ _lineage_root_id: 'lineage-root', id: 'current-tip' }]
      )
    ).toEqual(['lineage-root', 'current-tip'])
  })

  it('reactively combines foreground turns with background subagents until they complete', () => {
    $workingSessionIds.set(['foreground-session'])
    upsertSubagent(
      'runtime-parent',
      { child_session_id: 'stored-review', goal: 'Review', status: 'running', subagent_id: 'review-1' },
      true,
      'subagent.start',
      'stored-parent'
    )

    expect($sessionActivityIds.get()).toEqual(['foreground-session', 'stored-parent', 'stored-review'])

    upsertSubagent(
      'runtime-parent',
      { child_session_id: 'stored-review', goal: 'Review', status: 'completed', subagent_id: 'review-1' },
      true,
      'subagent.complete',
      'stored-parent'
    )

    expect($sessionActivityIds.get()).toEqual(['foreground-session'])
  })
})
