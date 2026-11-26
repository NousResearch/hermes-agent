import { describe, expect, it, vi } from 'vitest'

import { enterLocked, pausedReplyControls, sendButtonDisabled } from './paused-reply'

// A reply paused for its continuation holds a send in the queue (use-composer-submit.ts), also while a
// restart has the backend away: the composer's Enter and Send must reach it.
describe('paused-reply composer controls', () => {
  it('lets Enter through while disabled only when a send would be held', () => {
    expect(enterLocked(true, false)).toBe(true)
    expect(enterLocked(true, true)).toBe(false)
    expect(enterLocked(false, false)).toBe(false)
  })

  it('keeps Send usable while disabled only when a send would be held', () => {
    expect(sendButtonDisabled(true, true)).toBe(true)
    expect(sendButtonDisabled(true, true, true)).toBe(false)
    expect(sendButtonDisabled(true, false, true)).toBe(true)
    expect(sendButtonDisabled(false, true)).toBe(false)
  })

  it('reads Queue and offers no Steer while the reply is paused', () => {
    const steer = vi.fn()

    expect(pausedReplyControls(true, 'steer', steer)).toEqual({
      busyAction: 'queue',
      onSteer: undefined,
      sendWhileDisabled: true
    })
    expect(pausedReplyControls(false, 'steer', steer)).toEqual({
      busyAction: 'steer',
      onSteer: steer,
      sendWhileDisabled: false
    })
  })
})
