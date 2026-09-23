import { describe, expect, it } from 'vitest'

import { toWebviewInputSpace } from './preview-input'

// #116281: the act engine measures targets in guest CSS pixels, but the
// webview's input space is css × zoom. At the shipped 90 % default an
// unscaled click landed 1/0.9 too far from the origin and missed silently.
describe('toWebviewInputSpace', () => {
  it('scales pointer events by the guest zoom so the reporter\'s 500,310 target is hit at 90 %', () => {
    const down = toWebviewInputSpace({ button: 'left', clickCount: 1, type: 'mouseDown', x: 500, y: 310 }, 0.9)

    expect(down).toEqual({ button: 'left', clickCount: 1, type: 'mouseDown', x: 450, y: 279 })
    expect(toWebviewInputSpace({ deltaX: 0, deltaY: 600, type: 'mouseWheel', x: 380, y: 467 }, 1.25)).toEqual({
      deltaX: 0,
      deltaY: 600,
      type: 'mouseWheel',
      x: 475,
      y: 584
    })
  })

  it('leaves events untouched at 100 %, with an unknown zoom, and for keys', () => {
    const move = { type: 'mouseMove', x: 500, y: 382 } as const
    const key = { keyCode: 'Enter', type: 'keyDown' } as const

    expect(toWebviewInputSpace(move, 1)).toBe(move)
    expect(toWebviewInputSpace(move, undefined)).toBe(move)
    expect(toWebviewInputSpace(move, Number.NaN)).toBe(move)
    expect(toWebviewInputSpace(key, 0.9)).toBe(key)
  })
})
