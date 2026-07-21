import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $selectedStoredSessionId, setSelectedStoredSessionId } from './session'
import { $sessionTiles, focusOpenSession } from './session-states'

const SESSION_A = 'session-a'

beforeEach(() => {
  // Start clean: no selected session, no tiles, path at root.
  setSelectedStoredSessionId(null)
  $sessionTiles.set([])
  window.history.pushState({}, '', '/')
})

afterEach(() => {
  setSelectedStoredSessionId(null)
  $sessionTiles.set([])
  window.history.pushState({}, '', '/')
})

describe('focusOpenSession', () => {
  it('returns true when the session is an open tile', () => {
    $sessionTiles.set([
      { storedSessionId: SESSION_A }
    ])

    expect(focusOpenSession(SESSION_A)).toBe(true)
  })

  it('returns true when the session is selected and current route is a session route', () => {
    setSelectedStoredSessionId(SESSION_A)
    window.history.pushState({}, '', `/${SESSION_A}`)

    expect(focusOpenSession(SESSION_A)).toBe(true)
  })

  it('returns false when the session is selected but current route is NOT a session route', () => {
    setSelectedStoredSessionId(SESSION_A)
    window.history.pushState({}, '', '/messaging')

    expect(focusOpenSession(SESSION_A)).toBe(false)
  })

  it('returns false when the session is selected but route is a reserved page like /skills', () => {
    setSelectedStoredSessionId(SESSION_A)
    window.history.pushState({}, '', '/skills')

    expect(focusOpenSession(SESSION_A)).toBe(false)
  })

  it('returns false when the session does not match anything', () => {
    setSelectedStoredSessionId(SESSION_A)

    expect(focusOpenSession('other-session')).toBe(false)
  })

  it('returns false when no session is selected at all', () => {
    window.history.pushState({}, '', '/messaging')

    expect(focusOpenSession(SESSION_A)).toBe(false)
  })

  it('returns true for a tile session even when on a non-session route', () => {
    $sessionTiles.set([
      { storedSessionId: SESSION_A }
    ])
    window.history.pushState({}, '', '/messaging')

    // Tiles are always navigated via revealTreePane, never via navigate.
    expect(focusOpenSession(SESSION_A)).toBe(true)
  })
})
