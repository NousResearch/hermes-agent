import { describe, expect, it } from 'vitest'

import { createTouchSubmitGesture } from './touch-submit-gesture'

describe('the touch Send gesture', () => {
  it('submits once on a release inside the button and swallows its click', () => {
    const g = createTouchSubmitGesture()

    g.down()

    expect(g.up()).toBe(true)
    expect(g.takeClick()).toBe(true)

    // The flag is spent by that click — a later, real click is a real press.
    expect(g.takeClick()).toBe(false)
  })

  // Regression: submitting on pointerdown sent the message even though the
  // press had been dragged off the button.
  it('does not submit when the press leaves the button before release', () => {
    const g = createTouchSubmitGesture()

    g.down()
    g.leave()

    expect(g.up()).toBe(false)

    // A gesture that never submitted still owns its click.
    expect(g.takeClick()).toBe(true)
  })

  // Regression: a press held past the old 700 ms window submitted a second time
  // on release, and the busy composer read that as Stop — cancelling the turn it
  // had just started. There is no timer left to outlast.
  it('submits exactly once however long the press is held', () => {
    const g = createTouchSubmitGesture()

    g.down()

    // …the press is held for an arbitrary time; nothing here is time-based…

    expect(g.up()).toBe(true)
    expect(g.takeClick()).toBe(true)
    expect(g.takeClick()).toBe(false)
  })

  it('never submits after a cancelled gesture, and still swallows the click', () => {
    const g = createTouchSubmitGesture()

    g.down()
    g.cancel()

    expect(g.up()).toBe(false)
    expect(g.takeClick()).toBe(true)
  })

  it('re-arms when the finger comes back onto the button', () => {
    const g = createTouchSubmitGesture()

    g.down()
    g.leave()
    g.enter()

    expect(g.up()).toBe(true)
  })

  it('clears a stale swallow on the next press', () => {
    const g = createTouchSubmitGesture()

    g.down()
    g.leave()
    g.up()

    g.down()

    // The previous gesture's click never arrived; the new press must not have
    // its own click eaten.
    expect(g.takeClick()).toBe(false)
  })

  it('ignores a release that never had a press', () => {
    const g = createTouchSubmitGesture()

    expect(g.up()).toBe(false)
    expect(g.takeClick()).toBe(false)
  })
})
