import { describe, expect, it } from 'vitest'

import { desktopDeepLinkAction } from './deep-links'

describe('desktopDeepLinkAction', () => {
  it('opens a stored session from a hermes://session link', () => {
    expect(
      desktopDeepLinkAction({
        kind: 'session',
        name: '20260801_142533_1183a0',
        params: {}
      })
    ).toEqual({ kind: 'session', sessionId: '20260801_142533_1183a0' })
  })

  it('preserves blueprint deep-link command construction', () => {
    expect(
      desktopDeepLinkAction({
        kind: 'blueprint',
        name: 'morning-brief',
        params: { audience: 'Philippe LeBel', time: '08:00' }
      })
    ).toEqual({
      kind: 'blueprint',
      command: '/blueprint morning-brief audience="Philippe LeBel" time=08:00'
    })
  })

  it('ignores empty and unknown deep links', () => {
    expect(desktopDeepLinkAction(null)).toBeNull()
    expect(desktopDeepLinkAction({ kind: 'session', name: '', params: {} })).toBeNull()
    expect(desktopDeepLinkAction({ kind: 'unknown', name: 'value', params: {} })).toBeNull()
  })
})
