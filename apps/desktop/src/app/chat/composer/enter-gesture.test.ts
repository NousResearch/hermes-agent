import { describe, expect, it } from 'vitest'

import { composerEnterPressOwner, composerSendDelays, isComposerDoubleTap } from './enter-gesture'

const base = {
  doubleTap: false,
  pausedEnough: false,
  sendOnDoubleTap: true,
  sendOnHold: true,
  sendOnPause: true
}

describe('composerEnterPressOwner', () => {
  it('gives a double tap the press, even after the typing pause has elapsed', () => {
    // The reported bug: the pause rule ran first, so a deliberate double press
    // after a pause was treated as a delayed Enter and took the pause's grace
    // window instead of sending.
    expect(composerEnterPressOwner({ ...base, doubleTap: true, pausedEnough: true })).toBe('doubleTap')
  })

  it('leaves the press to the pause when the double tap is switched off', () => {
    expect(
      composerEnterPressOwner({ ...base, doubleTap: true, pausedEnough: true, sendOnDoubleTap: false })
    ).toBe('pauseOnRelease')
  })

  it('commits the pause on the press when no hold can supersede it', () => {
    expect(composerEnterPressOwner({ ...base, pausedEnough: true, sendOnHold: false })).toBe('pause')
  })

  it('waits for the release when the hold is armed, so a hold can still claim the press', () => {
    expect(composerEnterPressOwner({ ...base, pausedEnough: true })).toBe('pauseOnRelease')
  })

  it('claims nothing while the user is still typing', () => {
    expect(composerEnterPressOwner(base)).toBe('none')
  })

  it('claims nothing when the pause gesture is switched off, however long the pause', () => {
    expect(composerEnterPressOwner({ ...base, pausedEnough: true, sendOnPause: false })).toBe('none')
  })

  it('prefers the double tap over a pause that would otherwise wait for the release', () => {
    expect(composerEnterPressOwner({ ...base, doubleTap: true, pausedEnough: true })).toBe('doubleTap')
  })
})

describe('isComposerDoubleTap', () => {
  it('counts a press inside the window', () => {
    expect(isComposerDoubleTap(1_000, 700, 400)).toBe(true)
  })

  it('counts the boundary itself', () => {
    expect(isComposerDoubleTap(1_400, 1_000, 400)).toBe(true)
  })

  it('does not count a press past the window', () => {
    expect(isComposerDoubleTap(1_401, 1_000, 400)).toBe(false)
  })

  it('does not count the first press of a session', () => {
    expect(isComposerDoubleTap(1_000, 0, 400)).toBe(false)
  })
})

describe('composerSendDelays', () => {
  it('delays only the situations the user listed', () => {
    expect(composerSendDelays('pause', ['pause'])).toBe(true)
    expect(composerSendDelays('doubleTap', ['pause'])).toBe(false)
  })
})
