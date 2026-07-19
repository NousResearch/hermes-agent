import { beforeEach, describe, expect, it } from 'vitest'

import { $sessions, $workingSessionIds, $workingSessionProfiles, setSessionWorking } from './session'
import {
  $sessionActivityIds,
  $sessionActivityKeys,
  deriveSessionActivityIds,
  deriveSessionActivityKeys,
  sessionActivityKey
} from './session-activity'
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
    $workingSessionProfiles.set({})
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

  it('holds a detached terminal result active until its parent handoff starts', () => {
    expect(
      deriveSessionActivityIds([], {
        'runtime-parent': [
          child({ handoff: true, ownerSessionId: 'stored-parent', sessionId: 'stored-review', status: 'completed' })
        ]
      })
    ).toEqual(['stored-parent', 'stored-review'])
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

  it('keeps identical session ids scoped to their source profile', () => {
    const keys = deriveSessionActivityKeys(
      [],
      {
        same: [
          child({ id: 'same-child', ownerSessionId: 'same-owner', profile: 'alpha' }),
          child({ id: 'same-child', ownerSessionId: 'same-owner', profile: 'beta' })
        ]
      },
      [
        { id: 'same-owner', profile: 'alpha' },
        { id: 'same-owner', profile: 'beta' }
      ]
    )

    expect(new Set(keys)).toEqual(
      new Set([sessionActivityKey('alpha', 'same-owner'), sessionActivityKey('beta', 'same-owner')])
    )
  })

  it('does not attribute an unknown foreground session to the default profile', () => {
    $sessions.set([{ id: 'unknown', profile: 'default' } as never])
    setSessionWorking('unknown', true)

    expect($sessionActivityKeys.get()).toEqual([])
  })

  it('keeps colliding foreground session ids scoped to their producing profiles', () => {
    $sessions.set([{ id: 'same', profile: 'alpha' } as never, { id: 'same', profile: 'beta' } as never])

    setSessionWorking('same', true, 'alpha')
    expect($sessionActivityKeys.get()).toEqual([sessionActivityKey('alpha', 'same')])

    setSessionWorking('same', true, 'beta')
    expect(new Set($sessionActivityKeys.get())).toEqual(
      new Set([sessionActivityKey('alpha', 'same'), sessionActivityKey('beta', 'same')])
    )

    setSessionWorking('same', false, 'alpha')
    expect($sessionActivityKeys.get()).toEqual([sessionActivityKey('beta', 'same')])
  })

  it('does not attribute a profileless subagent event to the default profile', () => {
    upsertSubagent(
      'runtime-parent',
      { goal: 'Unknown source', status: 'running', subagent_id: 'unknown-review' },
      true,
      'subagent.start',
      'stored-parent'
    )

    expect($sessionActivityKeys.get()).not.toContain(sessionActivityKey('default', 'stored-parent'))
  })

  it('reactively combines foreground turns with background subagents until they complete', () => {
    $sessions.set([{ id: 'foreground-session', profile: 'default' } as never])
    $workingSessionIds.set(['foreground-session'])
    upsertSubagent(
      'runtime-parent',
      { child_session_id: 'stored-review', goal: 'Review', status: 'running', subagent_id: 'review-1' },
      true,
      'subagent.start',
      'stored-parent',
      'default'
    )

    expect($sessionActivityIds.get()).toEqual(['foreground-session', 'stored-parent', 'stored-review'])

    upsertSubagent(
      'runtime-parent',
      { child_session_id: 'stored-review', goal: 'Review', status: 'completed', subagent_id: 'review-1' },
      true,
      'subagent.complete',
      'stored-parent',
      'default'
    )

    expect($sessionActivityIds.get()).toEqual(['foreground-session'])
  })
})
