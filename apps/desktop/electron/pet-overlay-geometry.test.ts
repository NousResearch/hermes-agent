import { describe, expect, it } from 'vitest'

import { overlayWindowBoundsToDip } from './pet-overlay-geometry'

describe('pet overlay desktop geometry', () => {
  it('converts physical Win32 window bounds to Electron DIP coordinates', () => {
    const converted = overlayWindowBoundsToDip(
      { height: 600, width: 1000, x: 250, y: 750 },
      'win32',
      bounds => ({
        height: bounds.height / 1.25,
        width: bounds.width / 1.25,
        x: bounds.x / 1.25,
        y: bounds.y / 1.25
      })
    )

    expect(converted).toEqual({ height: 480, width: 800, x: 200, y: 600 })
  })

  it('keeps non-Windows enumerator coordinates unchanged', () => {
    const bounds = { height: 600, width: 1000, x: 250, y: 750 }
    let conversionCalls = 0

    expect(
      overlayWindowBoundsToDip(bounds, 'darwin', value => {
        conversionCalls += 1

        return value
      })
    ).toBe(bounds)
    expect(conversionCalls).toBe(0)
  })
})
