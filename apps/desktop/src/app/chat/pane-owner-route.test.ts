import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'

import { paneOwnerRoute } from './pane-owner-route'

const CONN = 'local'

describe('paneOwnerRoute', () => {
  beforeEach(() => {
    _resetSessionOwnerHintsForTests({ storage: true })
  })

  afterEach(() => {
    _resetSessionOwnerHintsForTests({ storage: true })
  })

  it('resolves the ambient-keyed hint (an in-place open under the active profile)', () => {
    setSessionOwnerHint('stored-1', { connectionId: CONN, profile: 'default' })

    expect(paneOwnerRoute('stored-1', { connectionId: CONN, profile: 'default' })).toMatchObject({
      connectionId: CONN,
      profile: 'default'
    })
  })

  it("falls back to the session's single filed hint for an in-place Bot chat (#137801)", () => {
    setSessionOwnerHint('stored-bot', { connectionId: CONN, profile: 'david' })

    // Chrome stays on the launch profile while the Bot chat owns the pane, so
    // the ambient-keyed lookup misses and the Bot's own hint must answer.
    expect(paneOwnerRoute('stored-bot', { connectionId: CONN, profile: 'default' })).toMatchObject({
      connectionId: CONN,
      profile: 'david'
    })
  })

  it('stays unowned when several profiles filed hints for the session', () => {
    setSessionOwnerHint('stored-multi', { connectionId: CONN, profile: 'david' })
    setSessionOwnerHint('stored-multi', { connectionId: CONN, profile: 'lia' })

    expect(paneOwnerRoute('stored-multi', { connectionId: CONN, profile: 'default' })).toBeUndefined()
  })

  it('stays unowned with no filed hint or no stored id', () => {
    expect(paneOwnerRoute('stored-none', { connectionId: CONN, profile: 'default' })).toBeUndefined()
    expect(paneOwnerRoute(null, { connectionId: CONN, profile: 'default' })).toBeUndefined()
    expect(paneOwnerRoute('stored-none', null)).toBeUndefined()
  })
})
