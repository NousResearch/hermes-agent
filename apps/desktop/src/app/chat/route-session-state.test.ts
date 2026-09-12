import { describe, expect, it } from 'vitest'

import { routeSessionId, sessionRoute } from '../routes'

import { isRouteSessionMismatch } from './route-session-state'

describe('isRouteSessionMismatch', () => {
  it('keeps the composer mounted when auto-compression rotates root to tip', () => {
    const sessions = [{ id: 'tip-2', _lineage_root_id: 'root-1' }]

    expect(isRouteSessionMismatch('root-1', 'tip-2', sessions)).toBe(false)
    expect(isRouteSessionMismatch('tip-2', 'root-1', sessions)).toBe(false)
  })

  it('still suppresses the previous chat while a genuinely different route loads', () => {
    const sessions = [{ id: 'selected', _lineage_root_id: null }]

    expect(isRouteSessionMismatch('next-session', 'selected', sessions)).toBe(true)
    expect(isRouteSessionMismatch('next-session', null, sessions)).toBe(true)
  })

  it('does not require a session row when the selected and routed ids already agree', () => {
    expect(isRouteSessionMismatch('same', 'same', [])).toBe(false)
    expect(isRouteSessionMismatch(null, 'same', [])).toBe(false)
  })

  it('keeps only the routed chat whose active view owns an existing transcript during selection churn', () => {
    const routedSessionId = routeSessionId(sessionRoute('session-a'))

    const sessions = [
      { id: 'session-a', _lineage_root_id: null },
      { id: 'session-b', _lineage_root_id: null }
    ]

    const activeTranscript = {
      activeRuntimeId: 'runtime-a',
      contextSwitching: false,
      messagesEmpty: false,
      transcriptStoredSessionId: 'session-a'
    }

    expect(isRouteSessionMismatch(routedSessionId, null, sessions, activeTranscript)).toBe(false)
    expect(isRouteSessionMismatch(routedSessionId, 'session-b', sessions, activeTranscript)).toBe(false)

    expect(
      isRouteSessionMismatch('session-b', 'session-a', sessions, activeTranscript),
      'genuine navigation must suppress session A'
    ).toBe(true)
    expect(
      isRouteSessionMismatch(routedSessionId, null, sessions, { ...activeTranscript, contextSwitching: true }),
      'profile or connection switches must not retain the prior context'
    ).toBe(true)
    expect(
      isRouteSessionMismatch(routedSessionId, null, sessions, { ...activeTranscript, messagesEmpty: true }),
      'a route with no prior transcript must keep loading'
    ).toBe(true)
    expect(
      isRouteSessionMismatch(routedSessionId, null, sessions, {
        ...activeTranscript,
        activeRuntimeId: 'runtime-b',
        transcriptStoredSessionId: 'session-b'
      }),
      'a background chat must never publish into the routed foreground'
    ).toBe(true)
  })
})
