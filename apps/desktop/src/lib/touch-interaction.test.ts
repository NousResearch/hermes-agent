import { afterEach, describe, expect, it, vi } from 'vitest'

import { isTouchInteraction } from './touch-interaction'

afterEach(() => vi.unstubAllGlobals())

describe('touch input ownership', () => {
  it('uses coarse capability only when the event has no input evidence', () => {
    vi.stubGlobal('matchMedia', vi.fn().mockReturnValue({ matches: true }))
    expect(isTouchInteraction(new Event('contextmenu'))).toBe(true)
    expect(isTouchInteraction(Object.assign(new Event('contextmenu'), { pointerType: 'mouse' }))).toBe(false)
    expect(
      isTouchInteraction(Object.assign(new Event('contextmenu'), { sourceCapabilities: { firesTouchEvents: false } }))
    ).toBe(false)
  })

  it('recognizes touch and pen on a hybrid device', () => {
    vi.stubGlobal('matchMedia', vi.fn().mockReturnValue({ matches: false }))

    for (const pointerType of ['touch', 'pen']) {
      expect(isTouchInteraction(Object.assign(new Event('contextmenu'), { pointerType }))).toBe(true)
    }

    expect(
      isTouchInteraction(Object.assign(new Event('dblclick'), { sourceCapabilities: { firesTouchEvents: true } }))
    ).toBe(true)
  })
})
