import { describe, expect, it } from 'vitest'

import { visibleWorkspaceSessions } from './workspace-group'

const sessions = ['one', 'two', 'three', 'four', 'five'].map(id => ({ id }))

describe('visibleWorkspaceSessions', () => {
  it('exposes every loaded profile row in Manual mode so rows beyond the preview can reorder', () => {
    expect(visibleWorkspaceSessions(sessions, true, 3, true).map(session => session.id)).toEqual([
      'one',
      'two',
      'three',
      'four',
      'five'
    ])
  })

  it('preserves the bounded profile preview outside Manual mode', () => {
    expect(visibleWorkspaceSessions(sessions, true, 3, false).map(session => session.id)).toEqual([
      'one',
      'two',
      'three'
    ])
  })

  it('keeps workspace paging independent from profile Manual mode', () => {
    expect(visibleWorkspaceSessions(sessions, false, 4, true).map(session => session.id)).toEqual([
      'one',
      'two',
      'three',
      'four'
    ])
  })
})
