import { describe, expect, it } from 'vitest'

import { normalizeWakeIndicatorState, selectWakeIndicatorDisplay, wakeIndicatorWindowBounds } from './wake-indicator'

describe('wake indicator window', () => {
  it('centers the helper window at the top of the selected display', () => {
    expect(
      wakeIndicatorWindowBounds({
        bounds: { height: 982, width: 1512, x: -120, y: 40 }
      })
    ).toEqual({
      height: 52,
      width: 176,
      x: 548,
      y: 40
    })
  })

  it('prefers the internal display and falls back to the primary display', () => {
    const primary = {
      bounds: { height: 1080, width: 1920, x: 0, y: 0 },
      id: 'primary',
      internal: false
    }

    const internal = {
      bounds: { height: 982, width: 1512, x: 1920, y: 0 },
      id: 'internal',
      internal: true
    }

    expect(selectWakeIndicatorDisplay([primary, internal], primary)).toBe(internal)

    expect(selectWakeIndicatorDisplay([primary], primary)).toBe(primary)
  })

  it('rejects unknown renderer states', () => {
    expect(normalizeWakeIndicatorState('capturing')).toBe('capturing')
    expect(normalizeWakeIndicatorState('other')).toBe('hidden')
    expect(normalizeWakeIndicatorState(null)).toBe('hidden')
  })
})
