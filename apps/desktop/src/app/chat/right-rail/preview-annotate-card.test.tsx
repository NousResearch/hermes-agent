import { describe, expect, it } from 'vitest'

import { ANNOTATE_CARD_WIDTH } from '@/lib/preview-annotate'

import { placeAnnotateCard } from './preview-annotate-card'

describe('placeAnnotateCard', () => {
  it('sits to the right of the pin instead of past a full-width selection', () => {
    const placed = placeAnnotateCard({
      paneHeight: 640,
      paneWidth: 420,
      rect: { height: 220, width: 400, x: 8, y: 48 }
    })

    expect(placed.left).toBeLessThan(80)
    expect(placed.left).toBeGreaterThan(12)
    expect(placed.top).toBeGreaterThanOrEqual(12)
  })

  it('stays inside the pane when the pin is on the right edge', () => {
    const placed = placeAnnotateCard({
      paneHeight: 400,
      paneWidth: 360,
      rect: { height: 40, width: 80, x: 300, y: 20 }
    })

    expect(placed.left + ANNOTATE_CARD_WIDTH).toBeLessThanOrEqual(360)
    expect(placed.left).toBeGreaterThanOrEqual(12)
    expect(placed.top).toBeGreaterThanOrEqual(12)
  })
})
