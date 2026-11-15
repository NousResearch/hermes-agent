import { describe, expect, it } from 'vitest'

import { shouldShowIntro } from './intro-visibility'

const showing = {
  activeSessionId: null,
  auxiliaryWindow: false,
  draftSession: false,
  enabled: true,
  freshDraftReady: true,
  messagesEmpty: true,
  primary: true,
  routedSessionView: false,
  selectedSessionId: null
} as const

describe('shouldShowIntro', () => {
  it('shows on a fresh draft in the primary window', () => {
    expect(shouldShowIntro(showing)).toBe(true)
  })

  it('hides when the Appearance toggle is off', () => {
    expect(shouldShowIntro({ ...showing, enabled: false })).toBe(false)
  })

  it('keeps the toggle authoritative over every other clause', () => {
    // Off means off: no window, session, or draft state re-enables the splash.
    const inputs = [
      { ...showing, auxiliaryWindow: true, enabled: false },
      { ...showing, enabled: false, freshDraftReady: false },
      { ...showing, enabled: false, primary: false },
      { ...showing, enabled: false, messagesEmpty: false },
      { ...showing, draftSession: true, enabled: false }
    ]

    for (const input of inputs) {
      expect(shouldShowIntro(input)).toBe(false)
    }
  })

  it('hides on surfaces that are not an empty primary draft', () => {
    expect(shouldShowIntro({ ...showing, primary: false })).toBe(false)
    expect(shouldShowIntro({ ...showing, auxiliaryWindow: true })).toBe(false)
    expect(shouldShowIntro({ ...showing, freshDraftReady: false })).toBe(false)
    expect(shouldShowIntro({ ...showing, routedSessionView: true })).toBe(false)
    expect(shouldShowIntro({ ...showing, selectedSessionId: 'session-1' })).toBe(false)
    expect(shouldShowIntro({ ...showing, activeSessionId: 'session-1' })).toBe(false)
    expect(shouldShowIntro({ ...showing, messagesEmpty: false })).toBe(false)
  })
})

describe('shouldShowIntro: draft sessions (tab-strip "+", ⌘T, ⌘N)', () => {
  // A draft session is a REAL, unlisted, never-sent session bound to a view —
  // the shape openNewSessionTab mints. Unlike a pure view-reset draft it is
  // selected, routed and active, so only the marker and emptiness matter.
  const draftSession = {
    ...showing,
    activeSessionId: 'runtime-1',
    draftSession: true,
    freshDraftReady: false,
    primary: false,
    routedSessionView: true,
    selectedSessionId: 'stored-1'
  } as const

  it('shows on an empty draft session in a tile', () => {
    expect(shouldShowIntro(draftSession)).toBe(true)
  })

  it('shows on an empty draft session bound to the primary view', () => {
    expect(shouldShowIntro({ ...draftSession, primary: true })).toBe(true)
  })

  it('hides once the conversation started', () => {
    expect(shouldShowIntro({ ...draftSession, messagesEmpty: false })).toBe(false)
  })

  it('never shows in an auxiliary window, even for a draft session', () => {
    expect(shouldShowIntro({ ...draftSession, auxiliaryWindow: true })).toBe(false)
  })

  it('hides when the Appearance toggle is off', () => {
    expect(shouldShowIntro({ ...draftSession, enabled: false })).toBe(false)
  })

  it('does not let a non-draft session sneak past the primary gates', () => {
    // draftSession=false with the rest of the tile shape: none of the
    // view-reset clauses hold, so nothing paints.
    expect(shouldShowIntro({ ...draftSession, draftSession: false })).toBe(false)
  })
})
